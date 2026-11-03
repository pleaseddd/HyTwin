"""
Grid Digital Twin
=================
Top-level digital twin for an H2-based energy grid.  This is the
"virtual replica" layer that:

  1. Orchestrates all component TwinNodes each step
  2. Computes grid-level KPIs (energy balance, H2 balance)
  3. Exposes the observation vector consumed by the RL agent
  4. Maintains the authoritative grid state for the simulation engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.event_bus import Event, EventBus
from ..core.state_manager import StateManager
from ..models import (
    WindTurbineModel, PhotovoltaicModel,
    ElectrolyzerModel, FuelCellModel,
    HydrogenTankModel, EnergyLoadModel,
    GridConnectionModel,
)
from ..models.energy_cost import EnergyCostModel
from ..sensors.base_sensor import SensorReading
from ..core.rng import spawn_one
from .twin_node import TwinNode, TwinNodeState

logger = logging.getLogger(__name__)


@dataclass
class GridState:
    """Aggregated grid-level state snapshot."""
    timestamp: datetime
    # Power flows [kW]
    wind_power_kw: float = 0.0
    pv_power_kw: float = 0.0
    electrolyzer_power_kw: float = 0.0
    fuel_cell_power_kw: float = 0.0
    load_kw: float = 0.0
    grid_exchange_kw: float = 0.0      # positive = import (implicit balance)
    # Explicit grid connection (only when GridConnectionModel present)
    grid_connection_kw: float = 0.0    # actual power from explicit grid model
    grid_available: bool = True        # grid outage flag
    # Economics
    energy_price_eur_kwh: float = 0.15 # current buy price
    step_cost_eur: float = 0.0         # cost of grid import this step
    cumulative_cost_eur: float = 0.0   # running total cost
    step_co2_kg: float = 0.0           # CO₂ from grid this step
    # H2 flows [kg/h]
    h2_production_kg_h: float = 0.0
    h2_consumption_kg_h: float = 0.0
    h2_storage_kg: float = 0.0
    h2_soc: float = 0.0
    h2_pressure_bar: float = 0.0
    # KPIs
    renewable_fraction: float = 0.0
    grid_self_sufficiency: float = 0.0
    system_efficiency: float = 0.0
    # Health / anomaly
    overall_health: float = 1.0
    anomaly_nodes: List[str] = field(default_factory=list)
    # The power / H2 fields above are the twin's view (model fused with the sensor
    # readings, which lag one step): what a controller can know.  ``physical`` holds the
    # same quantities straight from the models, i.e. what actually flows; the network
    # balance, the tanks and the costs use these.
    physical: Dict[str, float] = field(default_factory=dict)


class GridTwin:
    """
    Digital twin of the complete H2-based energy grid.

    Usage::

        twin = GridTwin(config, event_bus, state_manager)
        twin.build()     # instantiate models from config
        # each step:
        grid_state = twin.step(dt, weather, control_actions, sensor_readings)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[StateManager] = None,
        cost_model: Optional[EnergyCostModel] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config
        self._bus = event_bus
        self._sm = state_manager if state_manager is not None else StateManager()
        self._nodes: Dict[str, TwinNode] = {}
        self._current_grid_state: Optional[GridState] = None
        self._history: List[GridState] = []
        self._max_history = 8_640
        self._cost_model: Optional[EnergyCostModel] = cost_model
        self._cumulative_cost: float = 0.0
        # Only the components that actually draw randomness need a child
        # generator (wind turbines, loads, grid connections, cost model) —
        # see hytwin.core.rng for why this matters once a live simulation and
        # a background RL training job can run concurrently.
        self._rng = rng

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Instantiate all component models and twin nodes from config."""
        cfg = self._config

        # Wind turbines
        for wt_cfg in cfg.get("wind_turbines", []):
            mid = wt_cfg["id"]
            model = WindTurbineModel(mid, wt_cfg["params"], rng=spawn_one(self._rng))
            self._nodes[mid] = TwinNode(mid, model)

        # PV arrays
        for pv_cfg in cfg.get("pv_arrays", []):
            mid = pv_cfg["id"]
            model = PhotovoltaicModel(mid, pv_cfg["params"])
            self._nodes[mid] = TwinNode(mid, model)

        # Electrolyzers
        for el_cfg in cfg.get("electrolyzers", []):
            mid = el_cfg["id"]
            model = ElectrolyzerModel(mid, el_cfg["params"])
            self._nodes[mid] = TwinNode(mid, model)

        # Fuel cells
        for fc_cfg in cfg.get("fuel_cells", []):
            mid = fc_cfg["id"]
            model = FuelCellModel(mid, fc_cfg["params"])
            self._nodes[mid] = TwinNode(mid, model)

        # H2 tanks
        for tk_cfg in cfg.get("hydrogen_tanks", []):
            mid = tk_cfg["id"]
            model = HydrogenTankModel(mid, tk_cfg["params"])
            self._nodes[mid] = TwinNode(mid, model)

        # Loads
        for ld_cfg in cfg.get("loads", []):
            mid = ld_cfg["id"]
            model = EnergyLoadModel(mid, ld_cfg["params"], rng=spawn_one(self._rng))
            self._nodes[mid] = TwinNode(mid, model)

        # Grid connections (optional — backward-compatible)
        for gc_cfg in cfg.get("grid_connections", []):
            mid = gc_cfg["id"]
            model = GridConnectionModel(mid, gc_cfg["params"], rng=spawn_one(self._rng))
            self._nodes[mid] = TwinNode(mid, model)

        # Energy cost model (optional)
        # Only create a new one if not already provided via __init__
        if self._cost_model is None and "energy_cost" in cfg:
            self._cost_model = EnergyCostModel(cfg["energy_cost"], rng=spawn_one(self._rng))

        logger.info("GridTwin built with %d nodes", len(self._nodes))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        weather: Dict[str, Any],
        control_actions: Dict[str, Any],
        sensor_readings: Optional[List[SensorReading]] = None,
        timestamp: Optional[datetime] = None,
    ) -> GridState:
        """
        Advance the grid twin by one time step.

        Parameters
        ----------
        dt : float
            Step duration [s].
        weather : dict
            Weather context (from WeatherModel.step()).
        control_actions : dict
            Mapping of component_id → action dict passed to the model.
        sensor_readings : list of SensorReading, optional
        timestamp : datetime, optional

        Returns
        -------
        GridState
        """
        ts = timestamp or datetime.utcnow()
        ctx_base = dict(weather)
        ctx_base["timestamp"] = ts

        # Hold aggregated power and H2 flows
        wind_kw = 0.0
        pv_kw = 0.0
        load_kw = 0.0
        el_power_kw = 0.0
        fc_power_kw = 0.0
        grid_conn_kw = 0.0
        grid_available = True
        grid_co2_step = 0.0
        h2_produced_kg = 0.0
        h2_consumed_kg = 0.0
        h2_storage_kg = 0.0
        h2_soc = 0.0
        h2_pressure = 0.0
        _h2_tank_count = 0
        health_scores = []
        phys = dict.fromkeys(("wind_power_kw", "pv_power_kw", "electrolyzer_power_kw",
                              "fuel_cell_power_kw", "load_kw", "h2_kg", "h2_consumed_kg"), 0.0)

        # Advance cost model — feed it a renewable-favorability proxy (GHI +
        # wind speed, both already available from the weather context, no
        # ordering dependency on this step's actual generation) so the
        # day-ahead price reflects the real merit-order effect: abundant
        # zero-marginal-cost renewables push prices down.
        if self._cost_model is not None:
            pv_cf = min(1.0, max(0.0, weather.get("ghi_wm2", 0.0)) / 850.0)
            wind_cf = min(1.0, max(0.0, weather.get("wind_speed_ms", 0.0)) / 11.0)
            renewable_cf = 0.5 * pv_cf + 0.5 * wind_cf
            self._cost_model.step(ts, dt, renewable_cf=renewable_cf)
        energy_price = (
            self._cost_model.get_buy_price(ts)
            if self._cost_model is not None
            else 0.15
        )

        # Group sensor readings by node prefix
        node_readings: Dict[str, List[SensorReading]] = {}
        if sensor_readings:
            for r in sensor_readings:
                parts = r.sensor_id.split(".")
                nid = parts[0] if len(parts) >= 2 else r.sensor_id
                node_readings.setdefault(nid, []).append(r)

        # Step each node
        for node_id, node in self._nodes.items():
            model_type = type(node.model).__name__
            context = dict(ctx_base)

            # Merge control actions
            if node_id in control_actions:
                context.update(control_actions[node_id])

            # Model step
            model_state = node.model.step(dt, context)

            # Twin fusion
            s_readings = node_readings.get(node_id, [])
            twin_state = node.update(model_state, s_readings or None)
            health_scores.append(twin_state.health_score)

            # Update central state manager
            for k, v in model_state.values.items():
                self._sm.set(f"{node_id}.{k}", v, ts)

            mv = model_state.values
            if isinstance(node.model, WindTurbineModel):
                phys["wind_power_kw"] += mv.get("power_kw", 0.0)
            elif isinstance(node.model, PhotovoltaicModel):
                phys["pv_power_kw"] += mv.get("power_ac_kw", 0.0)
            elif isinstance(node.model, ElectrolyzerModel):
                phys["electrolyzer_power_kw"] += mv.get("power_kw", 0.0)
                phys["h2_kg"] += mv.get("h2_kg_step", 0.0)
            elif isinstance(node.model, FuelCellModel):
                phys["fuel_cell_power_kw"] += mv.get("power_kw", 0.0)
                phys["h2_consumed_kg"] += mv.get("h2_consumed_kg_step", 0.0)
            elif isinstance(node.model, EnergyLoadModel):
                phys["load_kw"] += mv.get("load_kw", 0.0)

            # Aggregate KPIs
            v = twin_state.fused_values
            if isinstance(node.model, WindTurbineModel):
                wind_kw += v.get("power_kw", 0.0)
            elif isinstance(node.model, PhotovoltaicModel):
                pv_kw += v.get("power_ac_kw", 0.0)
            elif isinstance(node.model, ElectrolyzerModel):
                el_power_kw += v.get("power_kw", 0.0)
                h2_produced_kg += v.get("h2_kg_step", 0.0)
            elif isinstance(node.model, FuelCellModel):
                fc_power_kw += v.get("power_kw", 0.0)
                h2_consumed_kg += v.get("h2_consumed_kg_step", 0.0)
            elif isinstance(node.model, HydrogenTankModel):
                h2_storage_kg += v.get("mass_kg", 0.0)
                h2_soc += v.get("soc", 0.0)
                h2_pressure += v.get("pressure_bar", 0.0)
                _h2_tank_count += 1
            elif isinstance(node.model, EnergyLoadModel):
                load_kw += v.get("load_kw", 0.0)
            elif isinstance(node.model, GridConnectionModel):
                grid_conn_kw += v.get("power_kw", 0.0)
                grid_available = bool(v.get("available", 1.0) > 0.5)
                grid_co2_step += v.get("co2_kg_step", 0.0)

        # Normalize multi-tank H2 SOC and pressure to per-tank averages
        if _h2_tank_count > 1:
            h2_soc /= _h2_tank_count
            h2_pressure /= _h2_tank_count

        # Grid balance: supply includes explicit grid connection
        renewable_kw = wind_kw + pv_kw
        supply_kw = renewable_kw + fc_power_kw + grid_conn_kw
        # Implicit residual (covers any unmodeled shortfall)
        grid_exchange = load_kw + el_power_kw - supply_kw   # import if positive

        # Economics
        dt_h = dt / 3600.0
        import_kw = max(0.0, grid_conn_kw)   # only count import
        step_cost = import_kw * dt_h * energy_price
        self._cumulative_cost += step_cost

        # KPIs
        renew_frac = min(1.0, renewable_kw / (load_kw + el_power_kw + 1e-9))
        self_suff = max(0.0, min(1.0, (renewable_kw + fc_power_kw) / (load_kw + 1e-9)))
        efficiency = supply_kw / (renewable_kw + 1e-9) if renewable_kw > 0 else 0.0

        overall_health = float(np.mean(health_scores)) if health_scores else 1.0
        anomaly_nodes = [
            nid for nid, node in self._nodes.items()
            if node.state and node.state.anomaly_score > 0.5
        ]

        gs = GridState(
            timestamp=ts,
            wind_power_kw=wind_kw,
            pv_power_kw=pv_kw,
            electrolyzer_power_kw=el_power_kw,
            fuel_cell_power_kw=fc_power_kw,
            load_kw=load_kw,
            grid_connection_kw=grid_conn_kw,
            grid_available=grid_available,
            energy_price_eur_kwh=energy_price,
            step_cost_eur=step_cost,
            cumulative_cost_eur=self._cumulative_cost,
            step_co2_kg=grid_co2_step,
            grid_exchange_kw=grid_exchange,
            h2_production_kg_h=h2_produced_kg * 3600.0 / dt if dt > 0 else 0.0,
            h2_consumption_kg_h=h2_consumed_kg * 3600.0 / dt if dt > 0 else 0.0,
            h2_storage_kg=h2_storage_kg,
            h2_soc=h2_soc,
            h2_pressure_bar=h2_pressure,
            renewable_fraction=renew_frac,
            grid_self_sufficiency=self_suff,
            system_efficiency=min(1.0, efficiency),
            overall_health=overall_health,
            anomaly_nodes=anomaly_nodes,
            physical={**{k: v for k, v in phys.items() if k.endswith("_kw")},
                      "h2_production_kg_h": phys["h2_kg"] * 3600.0 / dt if dt > 0 else 0.0,
                      "h2_consumption_kg_h": phys["h2_consumed_kg"] * 3600.0 / dt if dt > 0 else 0.0},
        )

        self._current_grid_state = gs
        self._history.append(gs)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if self._bus:
            self._bus.publish(Event(
                topic="twin.grid_state",
                payload=gs,
                source="grid_twin",
                timestamp=ts,
            ))

        return gs

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    @property
    def grid_state(self) -> Optional[GridState]:
        return self._current_grid_state

    @property
    def nodes(self) -> Dict[str, TwinNode]:
        """All twin nodes keyed by id (read-only use)."""
        return self._nodes

    @property
    def cost_model(self) -> Optional[EnergyCostModel]:
        return self._cost_model

    def snapshot(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Flat snapshot of the site's state manager (for the sensor layer)."""
        return self._sm.snapshot(prefix)

    def node(self, node_id: str) -> TwinNode:
        return self._nodes[node_id]

    def observe(self) -> np.ndarray:
        """Flat observation vector for RL agent."""
        gs = self._current_grid_state
        if gs is None:
            return np.zeros(14, dtype=np.float32)
        return np.array([
            gs.wind_power_kw,
            gs.pv_power_kw,
            gs.electrolyzer_power_kw,
            gs.fuel_cell_power_kw,
            gs.load_kw,
            gs.grid_exchange_kw,
            gs.h2_production_kg_h,
            gs.h2_consumption_kg_h,
            gs.h2_storage_kg,
            gs.h2_soc,
            gs.h2_pressure_bar,
            gs.renewable_fraction,
            gs.grid_self_sufficiency,
            gs.overall_health,
        ], dtype=np.float32)

    def history(self, last: Optional[int] = None) -> List[GridState]:
        return self._history[-last:] if last else list(self._history)

    def reset(self) -> None:
        for node in self._nodes.values():
            node.reset()
        self._current_grid_state = None
        self._history.clear()
        self._sm.clear()
        self._cumulative_cost = 0.0
        if self._cost_model is not None:
            self._cost_model.reset()
