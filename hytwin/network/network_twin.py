"""
Network Twin
============
Top-level digital twin of a **multi-site H2 energy network**.  It orchestrates
one :class:`~hytwin.digital_twin.grid_twin.GridTwin` per site (the "SiteTwin"),
a per-site :class:`~hytwin.weather.weather_field.WeatherField`, and the
inter-site link physics models, then runs an **explicit** network dispatch each
step so the energy balance closes on real flows (no implicit residual).

Step sequence
-------------
  1. WeatherField.step(ts)                      → per-site weather
  2. each SiteTwin.step(dt, weather, actions)   → local production/consumption
  3. local H2 storage update per site           → el charges / fc discharges tanks
  4. compute per-site electric net (gen − dem)
  5. H2 pipeline dispatch (storage balancing)   → tank transfers + compressor load
  6. electric line dispatch (cover deficits)    → inter-node power flows
  7. national grid slack per site + cost/CO₂
  8. assemble NetworkState (per-node + system KPIs)

The national grid connection acts as the per-site slack computed by the
dispatch (each site keeps its own market-zone price and carbon factor); the
site-local controllers still drive electrolyzers / fuel cells / demand-response
via ``actions_by_site``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..digital_twin.grid_twin import GridTwin, GridState
from ..models import (
    ElectricLineModel, H2PipelineModel,
    HydrogenTankModel, GridConnectionModel,
    WindTurbineModel, PhotovoltaicModel, FuelCellModel, EnergyLoadModel,
    ElectrolyzerModel,
)
from ..weather.weather_field import WeatherField
from ..core.event_bus import EventBus
from ..core.rng import spawn_generators, spawn_one
from .topology import LinkType, NetworkTopology
from .network_state import NodeState, NetworkState
from .dispatch import (
    dispatch_h2_pipelines, dispatch_electric_lines,
    site_soc, site_charge, site_discharge,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Virtual-sensor auto-generation (one sensor per component)
# ------------------------------------------------------------------

def _autogen_sensor_config(grid_config: Dict[str, Any], fault_prob: float) -> List[Dict[str, Any]]:
    """
    Derive a sensible virtual-sensor set from a site's component list so the
    digital-twin fusion layer (health / anomaly / quality) is active in network
    mode without hand-writing sensors for every device.
    """
    cfg = grid_config
    out: List[Dict[str, Any]] = []

    # Only these sensor types expose a fault_probability kwarg.
    _FAULTABLE = {"power", "pressure", "flow"}

    def add(cid, s_type, key, noise, drift=0.0):
        s = {"id": f"{cid}.{s_type}", "type": s_type, "model_key": f"{cid}.{key}",
             "noise_std": noise, "drift_rate": drift}
        if s_type in _FAULTABLE:
            s["fault_probability"] = fault_prob
        out.append(s)

    # Drift left at 0 so health stays stable over long continuous runs;
    # liveness comes from Gaussian noise + occasional (auto-clearing) spikes.
    for wt in cfg.get("wind_turbines", []):
        add(wt["id"], "power", "power_kw", 3.0)
    for pv in cfg.get("pv_arrays", []):
        add(pv["id"], "power", "power_ac_kw", 2.5)
    for el in cfg.get("electrolyzers", []):
        add(el["id"], "power", "power_kw", 2.5)
    for fc in cfg.get("fuel_cells", []):
        add(fc["id"], "power", "power_kw", 2.0)
    for tk in cfg.get("hydrogen_tanks", []):
        add(tk["id"], "pressure", "pressure_bar", 1.5)
        add(tk["id"], "h2_level", "soc", 0.003)
    for ld in cfg.get("loads", []):
        add(ld["id"], "power", "load_kw", 3.0)
    for gc in cfg.get("grid_connections", []):
        add(gc["id"], "power", "power_kw", 3.0)
    return out

ActionsProvider = Callable[[int, datetime, Optional[NetworkState]], Dict[str, Dict[str, Any]]]


class NetworkTwin:
    """
    Digital twin of a complete multi-site H2 network.

    Parameters
    ----------
    topology : NetworkTopology
    soc_target : float
        Target H₂ SOC used by the pipeline storage-balancing dispatch.
    seed : int, optional
        If given, this NetworkTwin (and its whole component subtree — sites,
        weather, sensors, links) draws randomness from an **independent**
        ``numpy.random.Generator`` seeded from *seed*, isolated from any other
        NetworkTwin instance running concurrently in the same process (e.g. a
        live simulation and a background RL training job).  ``seed=None``
        (default) keeps the legacy behaviour of drawing from the shared
        global ``numpy.random`` state — safe only when a single NetworkTwin
        is ever stepped at a time.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        soc_target: float = 0.5,
        enable_sensors: bool = True,
        sensor_fault_prob: float = 0.0006,
        seed: Optional[int] = None,
    ) -> None:
        self._topo = topology
        self._soc_target = float(soc_target)
        self._enable_sensors = bool(enable_sensors)
        self._seed = seed
        rng = np.random.default_rng(seed) if seed is not None else None

        site_ids = list(topology.sites.keys())
        site_rngs = dict(zip(site_ids, spawn_generators(rng, len(site_ids))))

        # Per-site digital twins (reuse GridTwin unchanged).
        self._sites: Dict[str, GridTwin] = {}
        for sid, spec in topology.sites.items():
            twin = GridTwin(spec.grid_config, rng=spawn_one(site_rngs[sid]))
            twin.build()
            self._sites[sid] = twin

        # Per-site virtual sensor layer (digital-twin fusion → health/anomaly).
        self._sensor_mgrs: Dict[str, Any] = {}
        if self._enable_sensors:
            from ..simulation.engine import _build_sensor_manager
            for sid, spec in topology.sites.items():
                scfg = spec.sensor_config or _autogen_sensor_config(spec.grid_config, sensor_fault_prob)
                if scfg:
                    self._sensor_mgrs[sid] = _build_sensor_manager(
                        scfg, EventBus(), rng=spawn_one(site_rngs[sid]))

        # Per-site weather.
        self._weather = WeatherField.from_topology(topology, rng=spawn_one(rng))

        # Inter-site link physics models.
        self._pipe_models: Dict[str, H2PipelineModel] = {}
        self._line_models: Dict[str, ElectricLineModel] = {}
        for lid, link in topology.links.items():
            if link.link_type == LinkType.H2_PIPELINE:
                self._pipe_models[lid] = H2PipelineModel(lid, link.params)
            elif link.link_type == LinkType.ELECTRIC_LINE:
                self._line_models[lid] = ElectricLineModel(lid, link.params, rng=spawn_one(rng))

        # Cache per-site helpers.
        self._tanks_by_site: Dict[str, List[HydrogenTankModel]] = {}
        self._grid_conn_by_site: Dict[str, List[GridConnectionModel]] = {}
        self._grid_info: Dict[str, Dict[str, float]] = {}
        for sid, twin in self._sites.items():
            tanks = [n.model for n in twin.nodes.values()
                     if isinstance(n.model, HydrogenTankModel)]
            gcs = [n.model for n in twin.nodes.values()
                   if isinstance(n.model, GridConnectionModel)]
            self._tanks_by_site[sid] = tanks
            self._grid_conn_by_site[sid] = gcs
            self._grid_info[sid] = {
                "max_import_kw": sum(getattr(g, "max_import_kw", 0.0) for g in gcs) or 1.0e9,
                "max_export_kw": sum(getattr(g, "max_export_kw", 0.0) for g in gcs),
                "carbon_factor": float(np.mean([
                    g.params.get("grid_carbon_factor", 0.4) for g in gcs
                ])) if gcs else 0.4,
                "has_grid": 1.0 if gcs else 0.0,
            }

        self._cumulative_cost = 0.0
        self._current: Optional[NetworkState] = None
        self._history: List[NetworkState] = []
        self._max_history = 8_640
        self._step_count = 0
        self._last_weather: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        timestamp: datetime,
        actions_by_site: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> NetworkState:
        ts = timestamp
        dt_h = dt / 3600.0
        actions_by_site = actions_by_site or {}

        # 1. Weather for every site.
        weather = self._weather.step(ts)
        self._last_weather = weather

        # 2. Step each site twin with its local weather + control actions.
        #    The virtual sensors read the previous step's state snapshot and
        #    feed the twin fusion (health / anomaly / sensor quality).
        gs_by_site: Dict[str, GridState] = {}
        for sid, twin in self._sites.items():
            actions = actions_by_site.get(sid) or self._naive_site_actions(sid)
            # Fuel cells may only burn H2 that is in the site tanks.
            # site_discharge splits evenly, so each tank yields at most the emptiest one's mass.
            tanks = self._tanks_by_site[sid]
            fcs = [nid for nid, n in twin.nodes.items() if isinstance(n.model, FuelCellModel)]
            if tanks and fcs:
                h2_kg = len(tanks) * min(t.soc * t.max_capacity_kg for t in tanks) / len(fcs)
                actions = {**actions, **{f: {**actions.get(f, {}), "h2_available_kg": h2_kg} for f in fcs}}
            readings = None
            mgr = self._sensor_mgrs.get(sid)
            if mgr is not None:
                readings = mgr.update(twin.snapshot(), timestamp=ts)
            gs = twin.step(dt, weather[sid], actions, sensor_readings=readings, timestamp=ts)
            gs_by_site[sid] = gs

        # 3. Local H2 storage update (electrolyzer charges, fuel cell discharges).
        for sid, gs in gs_by_site.items():
            self._update_local_h2(sid, gs, dt)

        # 4. Base electric net per site (surplus > 0, deficit < 0).
        net: Dict[str, float] = {}
        gen: Dict[str, float] = {}
        dem: Dict[str, float] = {}
        for sid, gs in gs_by_site.items():
            g = gs.wind_power_kw + gs.pv_power_kw + gs.fuel_cell_power_kw
            d = gs.load_kw + gs.electrolyzer_power_kw
            gen[sid], dem[sid], net[sid] = g, d, g - d

        # 5. H2 pipeline dispatch (mutates tanks, steps pipes).
        pipe_flows, compressor_kw, pipe_out, pipe_in = dispatch_h2_pipelines(
            self._topo, self._pipe_models, self._tanks_by_site,
            dt, ts, self._soc_target,
        )
        for sid, c_kw in compressor_kw.items():
            dem[sid] += c_kw
            net[sid] -= c_kw

        # 6. Electric line dispatch (mutates net, steps lines).
        line_flows, link_in, link_out = dispatch_electric_lines(
            self._topo, self._line_models, net, dt, ts,
        )

        # 7. National grid slack + economics per site.
        nodes: Dict[str, NodeState] = {}
        total_cost = 0.0
        total_co2 = 0.0
        for sid, gs in gs_by_site.items():
            info = self._grid_info[sid]
            grid_avail = any(g.available for g in self._grid_conn_by_site[sid]) \
                if self._grid_conn_by_site[sid] else False

            net_i = net[sid]
            grid_import = grid_export = curtailed = unmet = 0.0
            if net_i >= 0:
                grid_export = min(net_i, info["max_export_kw"])
                curtailed = net_i - grid_export
            else:
                deficit = -net_i
                grid_import = min(deficit, info["max_import_kw"]) if grid_avail else 0.0
                unmet = deficit - grid_import

            price = gs.energy_price_eur_kwh
            cost_model = self._sites[sid].cost_model
            sell_price = cost_model.get_sell_price(ts) if cost_model is not None else price * 0.28
            step_cost = grid_import * dt_h * price - grid_export * dt_h * sell_price
            step_co2 = grid_import * dt_h * info["carbon_factor"]
            total_cost += step_cost
            total_co2 += step_co2

            tanks = self._tanks_by_site[sid]
            nodes[sid] = NodeState(
                site_id=sid, timestamp=ts,
                renewable_kw=gs.wind_power_kw + gs.pv_power_kw,
                fuel_cell_kw=gs.fuel_cell_power_kw,
                load_kw=gs.load_kw,
                electrolyzer_kw=gs.electrolyzer_power_kw,
                compressor_load_kw=compressor_kw[sid],
                generation_kw=gen[sid],
                demand_kw=dem[sid],
                link_import_kw=link_in[sid],
                link_export_kw=link_out[sid],
                grid_import_kw=grid_import,
                grid_export_kw=grid_export,
                curtailed_kw=curtailed,
                unmet_kw=unmet,
                grid_available=grid_avail,
                h2_soc=site_soc(tanks),
                h2_storage_kg=sum(t.mass_kg for t in tanks),
                h2_pipeline_in_kg_h=pipe_in[sid],
                h2_pipeline_out_kg_h=pipe_out[sid],
                price_eur_kwh=price,
                step_cost_eur=step_cost,
                step_co2_kg=step_co2,
                grid_state=gs,
            )

        self._cumulative_cost += total_cost
        ns = self._assemble_network_state(ts, nodes, {**pipe_flows, **line_flows}, total_cost, total_co2)

        self._current = ns
        self._history.append(ns)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._step_count += 1
        return ns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_local_h2(self, sid: str, gs: GridState, dt: float) -> None:
        """Charge site tanks with electrolyzer output, discharge for fuel cell."""
        tanks = self._tanks_by_site[sid]
        if not tanks:
            return
        dt_h = dt / 3600.0
        produced_kg = gs.h2_production_kg_h * dt_h
        consumed_kg = gs.h2_consumption_kg_h * dt_h
        site_charge(tanks, produced_kg, dt)
        site_discharge(tanks, consumed_kg, dt)

    def _naive_site_actions(self, sid: str) -> Dict[str, Dict[str, Any]]:
        """
        Default per-node control when no controller is attached (F3): run each
        electrolyzer at 30 % and each fuel cell at 20 % of rated power so H₂
        production/consumption and inter-node flows are exercised.  Replaced by
        real controllers in F4/F5.
        """
        twin = self._sites[sid]
        actions: Dict[str, Dict[str, Any]] = {}
        for nid, node in twin.nodes.items():
            if isinstance(node.model, ElectrolyzerModel):
                rated = float(node.model.params.get("rated_power_kw", 0.0))
                actions[nid] = {"power_setpoint_kw": 0.30 * rated}
            elif isinstance(node.model, FuelCellModel):
                rated = float(node.model.params.get("rated_power_kw", 0.0))
                actions[nid] = {"power_setpoint_kw": 0.20 * rated}
        return actions

    def _assemble_network_state(self, ts, nodes, links, total_cost, total_co2) -> NetworkState:
        total_load = sum(n.load_kw for n in nodes.values())
        total_renew = sum(n.renewable_kw for n in nodes.values())
        total_gen = sum(n.generation_kw for n in nodes.values())
        total_import = sum(n.grid_import_kw for n in nodes.values())
        total_export = sum(n.grid_export_kw for n in nodes.values())
        total_curtail = sum(n.curtailed_kw for n in nodes.values())
        total_unmet = sum(n.unmet_kw for n in nodes.values())
        inter_p = sum(l.delivered for l in links.values() if l.link_type == LinkType.ELECTRIC_LINE.value)
        inter_h2 = sum(l.delivered for l in links.values() if l.link_type == LinkType.H2_PIPELINE.value)
        socs = [n.h2_soc for n in nodes.values() if self._tanks_by_site[n.site_id]]

        return NetworkState(
            timestamp=ts, nodes=nodes, links=links,
            total_load_kw=total_load,
            total_renewable_kw=total_renew,
            total_generation_kw=total_gen,
            total_grid_import_kw=total_import,
            total_grid_export_kw=total_export,
            total_curtailed_kw=total_curtail,
            unmet_demand_kw=total_unmet,
            total_cost_eur_step=total_cost,
            cumulative_cost_eur=self._cumulative_cost,
            total_co2_kg_step=total_co2,
            inter_node_power_kw=inter_p,
            inter_node_h2_kg_h=inter_h2,
            network_renewable_fraction=float(np.clip(total_renew / (total_load + 1e-9), 0.0, 1.0)),
            network_self_sufficiency=float(np.clip(1.0 - total_import / (total_load + 1e-9), 0.0, 1.0)),
            reliability_index=float(np.clip(1.0 - total_unmet / (total_load + 1e-9), 0.0, 1.0)),
            avg_h2_soc=float(np.mean(socs)) if socs else 0.0,
        )

    # ------------------------------------------------------------------
    # Convenience run loop
    # ------------------------------------------------------------------

    def run(
        self,
        steps: int,
        start_time: datetime,
        dt_seconds: float,
        actions_provider: Optional[ActionsProvider] = None,
    ) -> List[NetworkState]:
        """Run *steps* steps and return the list of NetworkState snapshots."""
        results: List[NetworkState] = []
        ts = start_time
        for i in range(steps):
            actions = actions_provider(i, ts, self._current) if actions_provider else None
            ns = self.step(dt_seconds, ts, actions)
            results.append(ns)
            ts = ts + _timedelta_seconds(dt_seconds)
        return results

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def topology(self) -> NetworkTopology:
        return self._topo

    @property
    def network_state(self) -> Optional[NetworkState]:
        return self._current

    @property
    def last_weather(self) -> Dict[str, Dict[str, Any]]:
        return self._last_weather

    def site(self, site_id: str) -> GridTwin:
        return self._sites[site_id]

    def history(self, last: Optional[int] = None) -> List[NetworkState]:
        return self._history[-last:] if last else list(self._history)

    def reset(self) -> None:
        for twin in self._sites.values():
            twin.reset()
        for mgr in self._sensor_mgrs.values():
            mgr.reset_all()
        for m in self._pipe_models.values():
            m.reset()
        for m in self._line_models.values():
            m.reset()
        self._weather.reset()
        # refresh cached tank/grid references (reset may rebuild internal state)
        self._cumulative_cost = 0.0
        self._current = None
        self._history.clear()
        self._step_count = 0


def _timedelta_seconds(seconds: float):
    from datetime import timedelta
    return timedelta(seconds=seconds)
