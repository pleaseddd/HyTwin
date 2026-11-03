"""
HyTwin — Multi-Site Network Control Room (SCADA / EMS / Digital Twin)
========================================================================
Real-time supervisory dashboard for the whole Italian H2 network.

Architecture (presentation kept separate from the simulation)
  Main thread   : FastAPI (Uvicorn) — serves the SPA + REST/WS API
  Sim thread    : NetworkTwin loop — puts rich frames on a queue
  WS broadcaster: pulls frames — pushes JSON to all clients

The simulation core (``hytwin.network``) is untouched: this module only *reads*
its state and derives presentation payloads (per-component snapshots, weather,
commanded setpoints, an objective breakdown, and an event/alarm stream).

Controllers switchable at runtime: ``none`` | ``classical`` | ``rl`` (if a
trained model is available).

The ``NetworkSimulationWorker`` has no FastAPI dependency (unit-testable);
FastAPI/uvicorn are imported lazily in ``create_app``.  This module intentionally
does NOT use ``from __future__ import annotations`` (FastAPI must resolve the
``WebSocket`` handler annotation, which is imported locally in ``create_app``).
"""

import json
import logging
import queue
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

from hytwin.models import (
    WindTurbineModel, PhotovoltaicModel, ElectrolyzerModel, FuelCellModel,
    HydrogenTankModel, EnergyLoadModel, GridConnectionModel,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = str(ROOT / "config" / "italy_network_large.yaml")
DEFAULT_RL_MODEL = str(ROOT / "output" / "rl_models" / "net_ppo_large")


def _read_model_steps(p: Path) -> int:
    """Read num_timesteps from a SB3 zip without loading the policy weights."""
    try:
        import zipfile as _zf
        with _zf.ZipFile(p) as z:
            if "data" in z.namelist():
                return int(json.loads(z.read("data")).get("num_timesteps", 0))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _resolve_rl_model(path: str, expected_obs_dim: Optional[int] = None) -> str:
    """Return *path* if the .zip exists; otherwise auto-discover the model with
    the highest number of training steps in the same directory, among the
    models built for this network when *expected_obs_dim* is given."""
    if Path(str(path) + ".zip").exists():
        return path
    model_dir = Path(path).parent
    candidates = list(model_dir.glob("*.zip"))
    if expected_obs_dim is not None:
        candidates = [c for c in candidates
                      if (d := _model_dims(c, {})[0]) is not None and _obs_compatible(d, expected_obs_dim)]
    if not candidates:
        return path
    best = max(candidates, key=_read_model_steps)
    best_stem = str(best.with_suffix(""))
    logger.info("RL model auto-discovered (most steps): %s", best_stem)
    return best_stem


# ============================================================================
# Component presentation mapping (device-appropriate metrics)
# ============================================================================

def _component_view(node) -> Optional[Dict[str, Any]]:
    """Build a device-appropriate presentation dict for one TwinNode."""
    model = node.model
    st = node.state
    vals = dict(st.fused_values) if st is not None else {}
    params = getattr(model, "params", {})

    def r(x, n=2):
        try:
            return round(float(x), n)
        except Exception:
            return 0.0

    if isinstance(model, WindTurbineModel):
        kind = "wind"
        metrics = {"power_kw": r(vals.get("power_kw")),
                   "wind_ms": r(vals.get("wind_speed_hub_ms")),
                   "rated_kw": r(params.get("rated_power_kw"))}
    elif isinstance(model, PhotovoltaicModel):
        kind = "pv"
        metrics = {"power_kw": r(vals.get("power_ac_kw")),
                   "cell_temp_c": r(vals.get("cell_temp_c")),
                   "irradiance": r(vals.get("g_poa_wm2")),
                   "rated_kw": r(params.get("rated_power_kw"))}
    elif isinstance(model, ElectrolyzerModel):
        kind = "electrolyzer"
        metrics = {"power_kw": r(vals.get("power_kw")),
                   "h2_kg_h": r(vals.get("h2_flow_kg_h"), 3),
                   "efficiency": r(vals.get("efficiency_hhv"), 3),
                   "rated_kw": r(params.get("rated_power_kw"))}
    elif isinstance(model, FuelCellModel):
        kind = "fuel_cell"
        metrics = {"power_kw": r(vals.get("power_kw")),
                   "h2_kg_h": r(vals.get("h2_flow_kg_h"), 3),
                   "efficiency": r(vals.get("efficiency_lhv"), 3),
                   "rated_kw": r(params.get("rated_power_kw"))}
    elif isinstance(model, HydrogenTankModel):
        kind = "tank"
        metrics = {"soc": r(vals.get("soc"), 4),
                   "pressure_bar": r(vals.get("pressure_bar")),
                   "mass_kg": r(vals.get("mass_kg")),
                   "capacity_kg": r(vals.get("max_capacity_kg"))}
    elif isinstance(model, EnergyLoadModel):
        kind = "load"
        metrics = {"load_kw": r(vals.get("load_kw")),
                   "dr_shed_kw": r(vals.get("dr_shed_kw"))}
    elif isinstance(model, GridConnectionModel):
        kind = "grid"
        metrics = {"power_kw": r(vals.get("power_kw")),
                   "available": bool(vals.get("available", 1.0) > 0.5)}
    else:
        return None

    return {
        "id": node.node_id,
        "kind": kind,
        "health": r(st.health_score, 3) if st is not None else 1.0,
        "anomaly": r(st.anomaly_score, 3) if st is not None else 0.0,
        "quality": r(st.sensor_quality, 3) if st is not None else 1.0,
        "fault_key": st.fault_key if st is not None else None,
        "fault_status": st.fault_status if st is not None else None,
        "fault_sensor_value": r(st.fault_sensor_value) if st is not None and st.fault_sensor_value is not None else None,
        "fault_model_value": r(st.fault_model_value) if st is not None and st.fault_model_value is not None else None,
        "m": metrics,
    }


# ============================================================================
# Objective breakdown (same shape the RL reward optimises — for the AI screen)
# ============================================================================

def _objective_terms(ns, topology) -> Dict[str, float]:
    """Live objective decomposition — same weights and reference scales as the env
    reward (NetworkRewardConfig with the network's rl_reward block)."""
    from hytwin.rl.network_environment import NetworkRewardConfig
    rc = NetworkRewardConfig(**topology.rl_reward)
    unmet_frac = ns.unmet_demand_kw / (ns.total_load_kw + 1e-9)
    soc_dev = abs(ns.avg_h2_soc - rc.soc_target)
    unmet = -(rc.w_unmet_linear * unmet_frac + rc.w_unmet_quadratic * unmet_frac ** 2)
    if unmet_frac > rc.unmet_tolerance:
        unmet -= rc.w_unmet_event
    return {
        "cost": round(-rc.w_cost * (ns.total_cost_eur_step / rc.ref_cost_eur_step), 3),
        "co2": round(-rc.w_co2 * (ns.total_co2_kg_step / rc.ref_co2_kg_step), 3),
        "unmet": round(unmet, 3),
        "self_sufficiency": round(rc.w_self_sufficiency * ns.network_self_sufficiency, 3),
        "renewable": round(rc.w_renewable * ns.network_renewable_fraction, 3),
        "soc": round(rc.w_soc_health * (1.0 - 2.0 * soc_dev), 3),
        "curtailment": round(-rc.w_curtailment * min(1.0, ns.total_curtailed_kw / rc.ref_curtail_kw), 3),
    }


# ============================================================================
# Simulation worker thread
# ============================================================================

class NetworkSimulationWorker:
    """Runs a NetworkTwin in a background thread and emits rich frames."""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        dt_seconds: float = 600.0,
        speed_factor: float = 0.0,
        seed: int = 42,
        rl_model_path: str = DEFAULT_RL_MODEL,
        queue_maxsize: int = 200,
    ) -> None:
        self._config_path = config_path
        self._dt = dt_seconds
        self._speed_factor = speed_factor
        self._seed = seed
        from hytwin.simulation.scenario import Scenario
        from hytwin.rl.network_environment import obs_dim_for_n_sites
        n_sites = len(Scenario.from_yaml(config_path).topology().site_ids)
        self._rl_model_path = _resolve_rl_model(rl_model_path, obs_dim_for_n_sites(n_sites))
        self._rl_available = Path(str(self._rl_model_path) + ".zip").exists()
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._running = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._controller_type = "classical"
        self._lock = threading.Lock()
        self._step = 0
        self._start_time = datetime(2024, 6, 15, 0, 0)

        self._topo = None
        self._twin = None
        self._classical = None
        self._rl = None
        self._prev_state = None
        self._sim_ts = self._start_time
        self._last_actions: Dict[str, Any] = {}

        self._node_history: Dict[str, List[Dict[str, Any]]] = {}
        self._comp_history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._kpi_history: Deque[Dict[str, Any]] = deque(maxlen=720)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=300)
        self._prev_flags: Dict[str, Dict[str, bool]] = {}
        self._history_lock = threading.Lock()

        self._topology_payload: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Build / lifecycle
    # ------------------------------------------------------------------

    def set_config_path(self, config_path: str) -> None:
        """Swap the scenario YAML used on the *next* build/reset."""
        self._config_path = config_path

    def _build(self) -> None:
        import sys
        sys.path.insert(0, str(ROOT))
        from hytwin.simulation.scenario import Scenario
        from hytwin.network.network_twin import NetworkTwin
        from hytwin.control.network_controller import NetworkClassicalController

        scenario = Scenario.from_yaml(self._config_path)
        self._topo = scenario.topology()
        # An explicit seed gives this live-simulation NetworkTwin its own
        # isolated RNG subtree, safe to run concurrently with a background
        # RL training job (which seeds its own NetworkTwin per-episode via
        # NetworkH2GridEnv.reset() — see hytwin.core.rng).
        self._twin = NetworkTwin(self._topo, seed=self._seed)
        self._classical = NetworkClassicalController.from_network(self._twin)
        self._rl = None
        self._prev_state = None
        self._sim_ts = scenario.start_time or self._start_time
        self._start_time = self._sim_ts
        self._last_actions = {}
        self._prev_flags = {}

        self._topology_payload = {
            "sites": [
                {"id": sid, "name": spec.location.name or sid,
                 "lat": spec.location.lat, "lon": spec.location.lon}
                for sid, spec in self._topo.sites.items()
            ],
            "links": [
                {"id": lid, "type": link.link_type.value,
                 "from": link.from_site, "to": link.to_site,
                 "length_km": round(link.length_km or 0.0, 1)}
                for lid, link in self._topo.links.items()
            ],
            "controllers": ["none", "classical", "rl"],
            "rl_available": self._rl_available,
            "rl_has_models": bool(list(Path(self._rl_model_path).parent.glob("*.zip"))),
            "rl_model_path": self._rl_model_path,
            "dt_seconds": self._dt,
            "start_time": self._sim_ts.isoformat(),
        }

    def _ensure_rl(self):
        if self._rl is None and self._rl_available:
            from hytwin.control.network_rl_controller import NetworkRLController
            self._rl = NetworkRLController.from_model_path(self._rl_model_path, self._topo, self._twin)
        return self._rl

    def prepare(self) -> None:
        """Build the twin (topology, weather, sensors) WITHOUT starting the
        stepping thread — the map/topology is visible immediately but the
        simulation stays paused until ``start()`` is explicitly called."""
        if self._twin is None:
            self._build()

    @property
    def is_built(self) -> bool:
        return self._twin is not None

    def set_rl_model(self, path: str) -> None:
        """Point the RL controller at a different trained model (e.g. one that
        just finished training) without restarting the dashboard."""
        self._rl_model_path = path
        self._rl_available = Path(str(path) + ".zip").exists()
        self._rl = None  # force reload on next use
        if self._topology_payload:
            self._topology_payload["controllers"] = ["none", "classical", "rl"]
            self._topology_payload["rl_available"] = self._rl_available
            self._topology_payload["rl_has_models"] = bool(list(Path(path).parent.glob("*.zip")))
            self._topology_payload["rl_model_path"] = self._rl_model_path

    def rl_model_path(self) -> str:
        return self._rl_model_path

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._running.set()
            return
        if self._twin is None:
            self._build()
        self._stop.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("NetworkSimulationWorker started")

    def stop(self) -> None:
        self._running.clear()

    def reset(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._step = 0
        with self._history_lock:
            self._node_history.clear()
            self._comp_history.clear()
            self._kpi_history.clear()
            self._events.clear()
        self._build()
        self.start()

    def set_controller(self, ctrl_type: str) -> None:
        with self._lock:
            self._controller_type = ctrl_type

    def set_speed_factor(self, sf: float) -> float:
        sf = max(0.0, min(300.0, float(sf)))
        with self._lock:
            self._speed_factor = sf
        return sf

    def get_speed_factor(self) -> float:
        with self._lock:
            return self._speed_factor

    def get_state_nowait(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def topology_payload(self) -> Dict[str, Any]:
        return self._topology_payload

    @property
    def topology(self):
        """The live ``NetworkTopology`` object (not the JSON payload)."""
        return self._topo

    def expected_obs_dim(self) -> Optional[int]:
        """Observation-space size an RL model must match to run on this net."""
        if self._topo is None:
            return None
        from hytwin.rl.network_environment import obs_dim_for_n_sites
        return obs_dim_for_n_sites(len(self._topo.site_ids))

    def node_history(self, site_id: str) -> List[Dict[str, Any]]:
        with self._history_lock:
            return list(self._node_history.get(site_id, []))

    def kpi_history(self) -> List[Dict[str, Any]]:
        with self._history_lock:
            return list(self._kpi_history)

    def events(self) -> List[Dict[str, Any]]:
        with self._history_lock:
            return list(self._events)

    def node_components(self, site_id: str) -> List[Dict[str, Any]]:
        if self._twin is None or site_id not in self._twin.topology.site_ids:
            return []
        return [c for c in (_component_view(n) for n in self._twin.site(site_id).nodes.values())
                if c is not None]

    def component_detail(self, site_id: str, comp_id: str) -> Optional[Dict[str, Any]]:
        comps = {c["id"]: c for c in self.node_components(site_id)}
        c = comps.get(comp_id)
        if c is None:
            return None
        with self._history_lock:
            hist = list(self._comp_history.get(f"{site_id}.{comp_id}", []))
        return {"site": site_id, "id": comp_id, "kind": c["kind"],
                "current": c, "history": hist}

    # ------------------------------------------------------------------
    # Control resolution
    # ------------------------------------------------------------------

    def _compute_actions(self):
        with self._lock:
            ctrl_type = self._controller_type
        if ctrl_type == "classical" and self._classical is not None:
            return self._classical.compute_actions(self._prev_state, self._sim_ts)
        if ctrl_type == "rl":
            rl = self._ensure_rl()
            if rl is not None:
                return rl.compute_actions(self._prev_state, self._sim_ts)
        return None

    def _commanded_setpoints(self, actions) -> Dict[str, Dict[str, float]]:
        """Summarise commanded setpoints per site (EL/FC total kW + DR)."""
        out: Dict[str, Dict[str, float]] = {}
        if not actions:
            return out
        for sid, comps in actions.items():
            el = fc = dr = 0.0
            for cid, params in comps.items():
                sp = float(params.get("power_setpoint_kw", 0.0))
                if "el" in cid:
                    el += sp
                elif "fc" in cid:
                    fc += sp
                if "demand_response" in params:
                    dr = max(dr, float(params["demand_response"]))
            out[sid] = {"el_setpoint_kw": round(el, 1), "fc_setpoint_kw": round(fc, 1),
                        "dr": round(dr, 3)}
        return out

    # ------------------------------------------------------------------
    # Frame + step
    # ------------------------------------------------------------------

    def step_once(self) -> Dict[str, Any]:
        actions = self._compute_actions()
        self._last_actions = self._commanded_setpoints(actions)
        ns = self._twin.step(self._dt, self._sim_ts, actions)
        self._prev_state = ns
        self._sim_ts = self._sim_ts + timedelta(seconds=self._dt)

        frame = ns.as_dict()
        frame["step"] = self._step
        with self._lock:
            frame["controller"] = self._controller_type
        frame["weather"] = self._weather_view()
        frame["components"] = {sid: self.node_components(sid) for sid in self._twin.topology.site_ids}
        frame["actions"] = self._last_actions
        frame["objective"] = _objective_terms(ns, self._topo)

        new_events = self._detect_events(ns, frame["components"])
        frame["new_events"] = new_events
        self._step += 1

        with self._history_lock:
            for ev in new_events:
                self._events.appendleft(ev)
            self._kpi_history.append(self._kpi_row(frame, ns))
            ts = frame["ts"]
            for sid, comps in frame["components"].items():
                for c in comps:
                    key = f"{sid}.{c['id']}"
                    h = self._comp_history.get(key)
                    if h is None:
                        h = self._comp_history[key] = deque(maxlen=200)
                    h.append({"ts": ts, "health": c["health"], "anomaly": c["anomaly"], **c["m"]})
            for sid, nd in frame["nodes"].items():
                hist = self._node_history.setdefault(sid, [])
                hist.append({
                    "ts": frame["ts"], "soc": nd["h2_soc"], "load_kw": nd["load_kw"],
                    "renewable_kw": nd["renewable_kw"], "fuel_cell_kw": nd["fuel_cell_kw"],
                    "electrolyzer_kw": nd["electrolyzer_kw"], "grid_import_kw": nd["grid_import_kw"],
                    "price_eur_kwh": nd["price_eur_kwh"],
                })
                if len(hist) > 288:
                    hist.pop(0)
        return frame

    def _weather_view(self) -> Dict[str, Any]:
        out = {}
        for sid, w in (self._twin.last_weather or {}).items():
            out[sid] = {
                "wind_ms": round(float(w.get("wind_speed_ms", 0.0)), 2),
                "ghi": round(float(w.get("ghi_wm2", 0.0)), 0),
                "temp_c": round(float(w.get("temperature_c", 0.0)), 1),
                "cloud": round(float(w.get("cloud_cover", 0.0)), 2),
            }
        return out

    def _kpi_row(self, frame, ns) -> Dict[str, Any]:
        return {
            "ts": frame["ts"], "step": frame["step"],
            "load": ns.total_load_kw, "renewable": ns.total_renewable_kw,
            "generation": ns.total_generation_kw, "grid_import": ns.total_grid_import_kw,
            "grid_export": ns.total_grid_export_kw, "curtailed": ns.total_curtailed_kw,
            "cost_step": ns.total_cost_eur_step, "cum_cost": ns.cumulative_cost_eur,
            "co2_step": ns.total_co2_kg_step, "self_sufficiency": ns.network_self_sufficiency,
            "renewable_fraction": ns.network_renewable_fraction, "reliability": ns.reliability_index,
            "avg_soc": ns.avg_h2_soc, "inter_node_power": ns.inter_node_power_kw,
            "inter_node_h2": ns.inter_node_h2_kg_h, "unmet": ns.unmet_demand_kw,
        }

    # Human-readable phrasing for each raw SensorStatus fault kind, keyed by
    # the enum's ``.name`` (see hytwin/sensors/base_sensor.py:SensorStatus).
    _FAULT_PHRASES = {
        "FAULT_SPIKE": "outlier spike",
        "FAULT_STUCK": "stuck reading (not updating)",
        "FAULT_OFFLINE": "sensor offline (no data)",
        "DEGRADED": "reading drifted from model estimate",
    }

    def _detect_events(self, ns, components) -> List[Dict[str, Any]]:
        """
        Edge-triggered event/alarm derivation from state transitions.

        Each condition uses a dead-band between its "raise" and "clear"
        thresholds so a value oscillating right at the boundary produces one
        event per genuine episode, not one per step-crossing.
        """
        events: List[Dict[str, Any]] = []
        ts = ns.timestamp.isoformat()

        def emit(node, sev, kind, msg):
            events.append({"ts": ts, "step": self._step, "node": node,
                           "severity": sev, "kind": kind, "msg": msg})

        for sid, n in ns.nodes.items():
            prev = self._prev_flags.setdefault(sid, {})
            # Grid outage
            out_now = not n.grid_available
            if out_now and not prev.get("outage"):
                emit(sid, "critical", "outage", "National grid unavailable")
            if (not out_now) and prev.get("outage"):
                emit(sid, "info", "outage_clear", "National grid restored")
            prev["outage"] = out_now

            # Unmet demand — raise above 1.0 kW, clear only once below 0.3 kW
            unmet_now = n.unmet_kw > 1.0
            unmet_clear = n.unmet_kw < 0.3
            if unmet_now and not prev.get("unmet"):
                emit(sid, "critical", "unmet", f"Unmet demand ({n.unmet_kw:.0f} kW)")
                prev["unmet"] = True
            elif unmet_clear and prev.get("unmet"):
                emit(sid, "info", "unmet_clear", "Demand fully served again")
                prev["unmet"] = False

            # SOC thresholds — 5-point dead-band between raise and clear
            soc_pct = n.h2_soc * 100
            if soc_pct < 15 and not prev.get("soc_crit"):
                emit(sid, "critical", "soc", f"Critical H₂ SOC ({soc_pct:.0f}%)")
                prev["soc_crit"], prev["soc_low"] = True, True
            elif soc_pct < 25 and not prev.get("soc_low") and not prev.get("soc_crit"):
                emit(sid, "warning", "soc", f"Low H₂ SOC ({soc_pct:.0f}%)")
                prev["soc_low"] = True
            elif soc_pct > 30 and (prev.get("soc_crit") or prev.get("soc_low")):
                emit(sid, "info", "soc_clear", f"H₂ SOC recovered ({soc_pct:.0f}%)")
                prev["soc_crit"], prev["soc_low"] = False, False

            # Price spike — raise above 0.40 €/kWh, clear below 0.32 €/kWh
            if n.price_eur_kwh > 0.40 and not prev.get("spike"):
                emit(sid, "warning", "price", f"High price ({n.price_eur_kwh:.2f} €/kWh)")
                prev["spike"] = True
            elif n.price_eur_kwh < 0.32 and prev.get("spike"):
                emit(sid, "info", "price_clear", "Price back to normal range")
                prev["spike"] = False

            # Component fault / anomaly — report the specific worst-offending
            # sensor (component id, fault kind, expected-vs-actual reading)
            # rather than a generic "anomaly detected" at site level.
            comps = components.get(sid, [])
            worst = max(comps, key=lambda c: c["anomaly"], default=None)
            anom_now = worst is not None and worst["anomaly"] > 0.5
            if anom_now and not prev.get("anom"):
                phrase = self._FAULT_PHRASES.get(worst.get("fault_status"), "reading diverges from model estimate")
                detail = ""
                if worst.get("fault_sensor_value") is not None and worst.get("fault_model_value") is not None:
                    detail = f" — measured {worst['fault_sensor_value']:g}, expected ~{worst['fault_model_value']:g}"
                emit(sid, "warning", "anomaly",
                     f"{worst['kind']} ({worst['id']}): {phrase}{detail}")
                prev["anom"] = True
            elif not anom_now and prev.get("anom"):
                prev["anom"] = False
        return events

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._running.is_set():
                time.sleep(0.05)
                continue
            try:
                t0 = time.perf_counter()
                frame = self.step_once()
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put_nowait(frame)
                speed = self.get_speed_factor()
                if speed > 0:
                    target = self._dt / speed
                    elapsed = time.perf_counter() - t0
                    if elapsed < target:
                        time.sleep(target - elapsed)
            except Exception as exc:  # noqa: BLE001
                logger.exception("NetworkSimulationWorker error: %s", exc)
                time.sleep(1)


# ============================================================================
# Scenario comparison (reproducible, runs off the sim thread)
# ============================================================================

def run_comparison(config_path: str, steps: int, seed: int, rl_model_path: str) -> Dict[str, Any]:
    from hytwin.simulation.scenario import Scenario
    from hytwin.network.compare import compare_controllers
    from hytwin.control.network_rl_controller import NetworkRLController

    topo = Scenario.from_yaml(config_path).topology()
    strategies: Dict[str, Any] = {"none": "none", "classical": "classical"}
    if Path(str(rl_model_path) + ".zip").exists():
        strategies["rl"] = NetworkRLController.factory(rl_model_path)
    return compare_controllers(topo, steps=steps, seed=seed, strategies=strategies)


# ============================================================================
# RL model registry — every model trained via the dashboard gets a JSON
# sidecar with training metadata; models trained via the CLI (no sidecar)
# still show up with basic file info so nothing is hidden from the operator.
# ============================================================================

# Cache of {path: (mtime, obs_dim, act_dim)} for models with no metadata
# sidecar (e.g. trained via the CLI before this field existed, or by an older
# version of the dashboard) — avoids re-loading the SB3 policy (torch weights)
# on every registry fetch just to read its observation-space shape.
_dim_probe_cache: Dict[str, tuple] = {}


def _model_dims(p: Path, meta: Dict[str, Any]) -> "tuple[Optional[int], Optional[int]]":
    """Obs/action dim for a model: from metadata if present, else a cached probe."""
    if meta.get("obs_dim") is not None:
        return meta["obs_dim"], meta.get("act_dim")
    mtime = p.stat().st_mtime
    cached = _dim_probe_cache.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    try:
        from hytwin.control.network_rl_controller import probe_model_dims
        obs_dim, act_dim = probe_model_dims(str(p.with_suffix("")))
    except Exception:  # noqa: BLE001
        return None, None
    _dim_probe_cache[str(p)] = (mtime, obs_dim, act_dim)
    return obs_dim, act_dim


def _obs_compatible(obs_dim: int, expected_obs_dim: int) -> bool:
    """A model fits the network if trained on the same sites, with or without site forecasts."""
    from hytwin.rl.network_environment import infer_n_sites_from_obs_dim, obs_dim_for_n_sites
    n = infer_n_sites_from_obs_dim(expected_obs_dim)
    return n is not None and obs_dim in (obs_dim_for_n_sites(n), obs_dim_for_n_sites(n, forecast=True))


def list_models(model_dir: Path, expected_obs_dim: Optional[int] = None) -> List[Dict[str, Any]]:
    if not model_dir.exists():
        return []
    from hytwin.rl.network_environment import infer_n_sites_from_obs_dim

    out = []
    for p in sorted(model_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True):
        meta: Dict[str, Any] = {}
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                meta = {}
        obs_dim, act_dim = _model_dims(p, meta)
        n_sites = meta.get("n_sites") or (infer_n_sites_from_obs_dim(obs_dim) if obs_dim else None)
        compatible = _obs_compatible(obs_dim, expected_obs_dim) if (obs_dim and expected_obs_dim) else None
        # steps: prefer sidecar JSON, fall back to reading from zip internals
        steps = meta.get("timesteps_completed") or _read_model_steps(p)
        out.append({
            "name": p.stem,
            "path": str(p.with_suffix("")),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            "size_kb": round(p.stat().st_size / 1024.0, 1),
            "timesteps_completed": steps or None,
            "trained_at": meta.get("trained_at"),
            "stopped_early": meta.get("stopped_early"),
            "has_metadata": bool(meta),
            "obs_dim": obs_dim,
            "n_sites": n_sites,
            "compatible": compatible,
        })
    return out


# ============================================================================
# Live comparison worker — steps 2-3 NetworkTwin instances (one per control
# strategy) in lockstep, ALL seeded identically so they see the exact same
# weather / price / outage stream (see hytwin.core.rng) — only their control
# decisions differ.  This is the live, real-time-paced counterpart to
# /api/compare (which runs to completion instantly and reports only the final
# aggregate KPIs): here the operator watches Traditional vs AI (vs no-control)
# diverge step by step, which is what makes the difference tangible instead of
# just a number.  Runs independently of (and, thanks to the RNG isolation from
# improvement #2, safely concurrently with) the main live-simulation worker
# and any background training job.
# ============================================================================

class LiveCompareWorker:
    """Runs several control strategies on identical conditions, live."""

    STRATEGY_LABELS = {"none": "None", "classical": "Traditional", "rl": "AI (RL)"}

    def __init__(
        self,
        config_path: str,
        rl_model_path: str,
        dt_seconds: float = 600.0,
        queue_maxsize: int = 200,
    ) -> None:
        self._config_path = config_path
        self._rl_model_path = rl_model_path
        self._dt = dt_seconds
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._running = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._speed_factor = 0.0
        self._step = 0
        self._strategies: List[str] = []
        self._seed = 42
        self._twins: Dict[str, Any] = {}
        self._controllers: Dict[str, Any] = {}
        self._prev_states: Dict[str, Any] = {}
        self._sim_ts: Optional[datetime] = None
        self._topo = None
        self._error: Optional[str] = None

    # ------------------------------------------------------------------

    def set_config_path(self, config_path: str) -> None:
        """Swap the scenario YAML used on the *next* configure_and_start()."""
        self._config_path = config_path

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "configured": bool(self._strategies),
                "running": self._running.is_set(),
                "strategies": list(self._strategies),
                "seed": self._seed,
                "step": self._step,
                "speed_factor": self._speed_factor,
                "error": self._error,
            }

    def configure_and_start(self, strategies: List[str], seed: int, rl_model_path: Optional[str] = None) -> None:
        """(Re)build fresh twins for *strategies* under a shared *seed* and start stepping."""
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=5)
        if rl_model_path:
            self._rl_model_path = rl_model_path
        self._build(strategies, seed)
        self._stop.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _build(self, strategies: List[str], seed: int) -> None:
        from hytwin.simulation.scenario import Scenario
        from hytwin.network.network_twin import NetworkTwin
        from hytwin.control.network_controller import NetworkClassicalController
        from hytwin.control.network_rl_controller import NetworkRLController

        scenario = Scenario.from_yaml(self._config_path)
        self._topo = scenario.topology()
        self._sim_ts = scenario.start_time or datetime(2024, 6, 15, 0, 0)
        self._step = 0
        self._error = None
        self._strategies = list(strategies)
        self._seed = int(seed)
        self._twins = {}
        self._controllers = {}
        self._prev_states = {}
        self._cum_co2: Dict[str, float] = {s: 0.0 for s in self._strategies}

        for strat in self._strategies:
            # Same seed for every strategy -> identical weather/price/outage
            # stream across twins (proven in improvement #2); only the
            # control decisions differ, which is what makes the comparison fair.
            twin = NetworkTwin(self._topo, seed=self._seed)
            self._twins[strat] = twin
            self._prev_states[strat] = None
            if strat == "classical":
                self._controllers[strat] = NetworkClassicalController.from_network(twin)
            elif strat == "rl":
                self._controllers[strat] = NetworkRLController.from_model_path(self._rl_model_path, self._topo, twin)
            else:
                self._controllers[strat] = None

    def stop(self) -> None:
        self._running.clear()

    def get_state_nowait(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def set_speed_factor(self, sf: float) -> float:
        sf = max(0.0, min(300.0, float(sf)))
        with self._lock:
            self._speed_factor = sf
        return sf

    def get_speed_factor(self) -> float:
        with self._lock:
            return self._speed_factor

    # ------------------------------------------------------------------

    def step_once(self) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        for strat in self._strategies:
            twin = self._twins[strat]
            ctrl = self._controllers.get(strat)
            actions = ctrl.compute_actions(self._prev_states[strat], self._sim_ts) if ctrl is not None else None
            ns = twin.step(self._dt, self._sim_ts, actions)
            self._prev_states[strat] = ns
            self._cum_co2[strat] += ns.total_co2_kg_step
            kpis[strat] = self._kpi_row(ns, self._cum_co2[strat])
        ts_iso = self._sim_ts.isoformat()
        self._sim_ts = self._sim_ts + timedelta(seconds=self._dt)
        self._step += 1
        return {"ts": ts_iso, "step": self._step, "strategies": list(self._strategies), "kpis": kpis}

    @staticmethod
    def _kpi_row(ns, cum_co2: float) -> Dict[str, float]:
        return {
            "cost_step": round(ns.total_cost_eur_step, 4),
            "cum_cost": round(ns.cumulative_cost_eur, 2),
            "co2_step": round(ns.total_co2_kg_step, 4),
            "cum_co2": round(cum_co2, 3),
            "self_sufficiency": round(ns.network_self_sufficiency, 4),
            "renewable_fraction": round(ns.network_renewable_fraction, 4),
            "reliability": round(ns.reliability_index, 4),
            "avg_soc": round(ns.avg_h2_soc, 4),
            "grid_import": round(ns.total_grid_import_kw, 1),
            "unmet": round(ns.unmet_demand_kw, 2),
            "load": round(ns.total_load_kw, 1),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._running.is_set():
                time.sleep(0.05)
                continue
            try:
                t0 = time.perf_counter()
                frame = self.step_once()
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put_nowait(frame)
                speed = self.get_speed_factor()
                if speed > 0:
                    target = self._dt / speed
                    elapsed = time.perf_counter() - t0
                    if elapsed < target:
                        time.sleep(target - elapsed)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LiveCompareWorker error: %s", exc)
                with self._lock:
                    self._error = str(exc)
                self._running.clear()


# ============================================================================
# Training worker — runs PPO training in a background thread so an operator
# can launch it from the dashboard without blocking the live simulation.
# Progress is reported via a stable-baselines3 callback; a stop request makes
# the callback return False, which aborts ``model.learn()`` early (SB3 still
# saves whatever policy it has at that point).
#
# The training env and the live-simulation env each own an *isolated*
# numpy.random.Generator subtree (see hytwin.core.rng / NetworkTwin(seed=...)
# / NetworkH2GridEnv.reset()), so running both concurrently no longer shares
# any mutable RNG state — each stays independently reproducible. The one
# residual caveat is stable-baselines3 itself: constructing PPO(seed=...)
# calls SB3's own set_random_seed(), which reseeds the global numpy/python/
# torch RNGs as a one-off at training start (outside hytwin's control) — this
# has no effect on the per-instance generators used by the physics/weather/
# sensor stack above, so it does not reintroduce cross-thread interference.
# ============================================================================

class TrainingWorker:
    """Background PPO training job with live progress and early-stop support."""

    #: cap on how many completed-episode reward points we retain/report —
    #: generous for any demo-scale training run in this project while keeping
    #: the /api/train/status payload bounded on very long jobs.
    MAX_REWARD_POINTS = 3000

    def __init__(self, config_path: str, model_dir: Path) -> None:
        self._config_path = config_path
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._status: Dict[str, Any] = {
            "state": "idle", "progress": 0.0, "current_step": 0, "total_steps": 0,
            "elapsed_s": 0.0, "eta_s": None, "save_path": None, "error": None,
            "name": None, "started_at": None,
        }
        self._reward_history: List[Dict[str, float]] = []

    def set_config_path(self, config_path: str) -> None:
        """Swap the scenario YAML used on the *next* start()."""
        self._config_path = config_path

    def start(self, timesteps: int, n_steps: int = 576, name: Optional[str] = None,
              seed: int = 0, n_envs: int = 1) -> Dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A training job is already running.")
            self._stop_flag.clear()
            run_name = name or f"net_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
            self._status = {
                "state": "running", "progress": 0.0, "current_step": 0,
                "total_steps": int(timesteps), "elapsed_s": 0.0, "eta_s": None,
                "save_path": None, "error": None, "name": run_name,
                "started_at": datetime.utcnow().isoformat() + "Z",
                "n_envs": int(n_envs),
            }
            self._reward_history = []
            self._thread = threading.Thread(
                target=self._run,
                args=(int(timesteps), int(n_steps), run_name, int(seed), int(n_envs)),
                daemon=True,
            )
            self._thread.start()
        return dict(self._status)

    def stop(self) -> None:
        self._stop_flag.set()

    def status(self, include_history: bool = True) -> Dict[str, Any]:
        with self._lock:
            d = dict(self._status)
            if include_history:
                d["reward_history"] = list(self._reward_history)
            return d

    def _run(self, timesteps: int, n_steps: int, run_name: str, seed: int, n_envs: int = 1) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
            from hytwin.simulation.scenario import Scenario
            from hytwin.rl.network_trainer import train_network_agent
            from hytwin.rl.network_environment import NetworkH2GridEnv

            topo = Scenario.from_yaml(self._config_path).topology()
            # Cheap (no torch/model) space construction, just to record the
            # obs/action dimensions this run's model will be locked to — lets
            # the dashboard warn before activating it on an incompatible net.
            _dims_env = NetworkH2GridEnv(topo)
            obs_dim = int(_dims_env.observation_space.shape[0])
            act_dim = int(_dims_env.action_space.shape[0])
            n_sites = len(topo.site_ids)
            t0 = time.time()
            outer = self

            class _ProgressCallback(BaseCallback):
                def _on_step(cb_self) -> bool:  # noqa: N805
                    if outer._stop_flag.is_set():
                        return False
                    elapsed = time.time() - t0
                    step = cb_self.num_timesteps
                    frac = min(1.0, step / max(1, timesteps))
                    eta = (elapsed / frac - elapsed) if frac > 0.03 else None
                    # Monitor (wrapping the env in train_network_agent) injects
                    # info["episode"] = {"r": return, "l": length, "t": time}
                    # into the step info dict exactly once, on the step an
                    # episode ends — the same mechanism SB3's own console
                    # ep_rew_mean logging uses. Capture each one as a point on
                    # the live learning curve.
                    for info in cb_self.locals.get("infos", []):
                        ep = info.get("episode")
                        if ep is not None:
                            with outer._lock:
                                outer._reward_history.append({
                                    "step": step, "reward": round(float(ep["r"]), 3),
                                })
                                if len(outer._reward_history) > outer.MAX_REWARD_POINTS:
                                    outer._reward_history.pop(0)
                    with outer._lock:
                        outer._status.update(current_step=step, progress=frac,
                                             elapsed_s=elapsed, eta_s=eta)
                    return True

            self._model_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(self._model_dir / run_name)
            model = train_network_agent(
                topo, timesteps=timesteps, save_path=save_path, seed=seed,
                n_steps=n_steps, n_envs=n_envs, callback=_ProgressCallback(),
            )
            stopped_early = self._stop_flag.is_set()
            meta = {
                "name": run_name,
                "timesteps_requested": timesteps,
                "timesteps_completed": int(model.num_timesteps),
                "n_steps": n_steps,
                "seed": seed,
                "trained_at": datetime.utcnow().isoformat() + "Z",
                "config": self._config_path,
                "stopped_early": stopped_early,
                "elapsed_s": round(time.time() - t0, 1),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_sites": n_sites,
            }
            Path(save_path + ".json").write_text(json.dumps(meta, indent=2))
            with self._lock:
                self._status.update(
                    state="stopped" if stopped_early else "completed",
                    progress=self._status["progress"] if stopped_early else 1.0,
                    save_path=save_path + ".zip",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training job failed: %s", exc)
            with self._lock:
                self._status.update(state="error", error=str(exc))


# ============================================================================
# FastAPI application factory
# ============================================================================

def create_app(
    config_path: str = DEFAULT_CONFIG,
    dt_seconds: float = 600.0,
    speed_factor: float = 0.0,
    seed: int = 42,
    rl_model_path: str = DEFAULT_RL_MODEL,
):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
        import uvicorn
    except ImportError as e:
        raise ImportError(
            f"Dashboard requires fastapi + uvicorn: pip install fastapi uvicorn\n{e}"
        )
    import asyncio
    from contextlib import asynccontextmanager

    worker = NetworkSimulationWorker(config_path, dt_seconds, speed_factor, seed, rl_model_path)
    training_worker = TrainingWorker(config_path, Path(rl_model_path).parent)
    compare_worker = LiveCompareWorker(config_path, rl_model_path, dt_seconds)
    active_ws: Set[WebSocket] = set()
    active_ws_compare: Set[WebSocket] = set()

    @asynccontextmanager
    async def lifespan(_app):
        # Build the topology so the map/config are available immediately, but
        # do NOT start stepping — the operator must press "Start simulation"
        # so the running/paused state is always an explicit, visible action.
        worker.prepare()
        yield
        worker.stop()
        compare_worker.stop()
        training_worker.stop()

    app = FastAPI(title="HyTwin — Network Control Room", lifespan=lifespan)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (Path(__file__).parent / "static" / "network.html").read_text(encoding="utf-8")

    @app.get("/network/topology")
    async def topology():
        return worker.topology_payload()

    @app.get("/api/config")
    async def api_config():
        return worker.topology_payload()

    # ------------------------------------------------------------------
    # Scenario configuration — list / read / validate+save / activate the
    # YAML files under config/. Editing must happen before a sim is (re)built:
    # "activate" only swaps the path used on the *next* /sim/reset, /api/train/start
    # or /api/live_compare/start, it never mutates a network mid-run.
    # ------------------------------------------------------------------

    config_dir = (ROOT / "config").resolve()

    def _scenario_file(name: str) -> Path:
        # Prevent path traversal — only bare filenames inside config/ are valid.
        p = (config_dir / name).resolve()
        if p.parent != config_dir or not name.endswith((".yaml", ".yml")):
            raise HTTPException(status_code=400, detail="Invalid scenario filename.")
        return p

    @app.get("/api/scenarios")
    async def api_scenarios_list():
        active = str(Path(worker._config_path).resolve())
        out = []
        for p in sorted(config_dir.glob("*.yaml")):
            out.append({"name": p.name, "active": str(p.resolve()) == active})
        return {"scenarios": out}

    @app.get("/api/scenarios/{name}")
    async def api_scenario_get(name: str):
        p = _scenario_file(name)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Scenario not found: {name}")
        return {"name": name, "yaml": p.read_text(encoding="utf-8")}

    @app.post("/api/scenarios/{name}")
    async def api_scenario_save(name: str, body: dict):
        import yaml as _yaml
        from hytwin.simulation.scenario import Scenario

        p = _scenario_file(name)
        text = body.get("yaml", "")
        try:
            data = _yaml.safe_load(text)
            if not isinstance(data, dict):
                raise ValueError("YAML root must be a mapping.")
            tmp = config_dir / f".{name}.validate.tmp"
            tmp.write_text(text, encoding="utf-8")
            try:
                Scenario.from_yaml(str(tmp)).topology()
            finally:
                tmp.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid scenario: {e}")
        p.write_text(text, encoding="utf-8")
        return {"status": "ok", "name": name}

    @app.post("/api/scenarios/{name}/activate")
    async def api_scenario_activate(name: str):
        p = _scenario_file(name)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Scenario not found: {name}")
        path = str(p)
        worker.set_config_path(path)
        training_worker.set_config_path(path)
        compare_worker.set_config_path(path)
        return {"status": "ok", "active": name,
                "note": "Active on next /sim/reset, training run, or comparison run."}

    @app.get("/api/history")
    async def api_history():
        return {"kpi": worker.kpi_history()}

    @app.get("/api/events")
    async def api_events():
        return {"events": worker.events()}

    @app.get("/node/{site_id}")
    async def node_detail(site_id: str):
        if site_id not in {s["id"] for s in worker.topology_payload().get("sites", [])}:
            raise HTTPException(status_code=404, detail=f"Site '{site_id}' not found")
        return {
            "id": site_id,
            "history": worker.node_history(site_id)[-144:],
            "components": worker.node_components(site_id),
        }

    @app.get("/api/component/{site_id}/{comp_id}")
    async def component_detail(site_id: str, comp_id: str):
        d = worker.component_detail(site_id, comp_id)
        if d is None:
            raise HTTPException(status_code=404, detail=f"Component '{comp_id}' not found in '{site_id}'")
        return d

    @app.post("/api/compare")
    async def api_compare(body: dict):
        steps = int(body.get("steps", 288))
        seed = int(body.get("seed", 42))
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_comparison, config_path, steps, seed, worker.rl_model_path())
        return {"steps": steps, "seed": seed, "result": result}

    # ------------------------------------------------------------------
    # RL model registry
    # ------------------------------------------------------------------

    @app.get("/api/models")
    async def api_models():
        return {"models": list_models(Path(rl_model_path).parent, worker.expected_obs_dim()),
                "active": worker.rl_model_path(),
                "expected_obs_dim": worker.expected_obs_dim()}

    @app.post("/api/models/select")
    async def api_models_select(body: dict):
        from hytwin.control.network_rl_controller import probe_model_dims
        from hytwin.rl.network_environment import infer_n_sites_from_obs_dim

        path = body.get("path")
        if not path or not Path(path + ".zip").exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {path}")
        expected = worker.expected_obs_dim()
        try:
            model_obs_dim, _ = probe_model_dims(path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Could not load model: {e}")
        if expected is not None and not _obs_compatible(model_obs_dim, expected):
            n_model = infer_n_sites_from_obs_dim(model_obs_dim) or "?"
            n_now = len(worker.topology.site_ids) if worker.topology else "?"
            raise HTTPException(status_code=409, detail=(
                f"Model incompatible with the active network: trained for {n_model} nodes "
                f"(observation {model_obs_dim}-D), but the current network has {n_now} "
                f"(expected {expected}-D). Select a model trained on this same network."
            ))
        worker.set_rl_model(path)
        return {"status": "ok", "active": path}

    @app.post("/api/models/delete")
    async def api_models_delete(body: dict):
        path = body.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="path required")
        zip_path = Path(path + ".zip")
        if not zip_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {path}")
        # Deactivate if this model is currently active
        if worker.rl_model_path() == path:
            worker.set_rl_model(str(zip_path.parent / "net_ppo_large"))  # reset to default
        zip_path.unlink(missing_ok=True)
        Path(path + ".json").unlink(missing_ok=True)  # remove sidecar if present
        return {"status": "deleted", "path": path}

    # ------------------------------------------------------------------
    # Training mode — launch/monitor/stop a background PPO training job
    # ------------------------------------------------------------------

    @app.post("/api/train/start")
    async def train_start(body: dict):
        timesteps = int(body.get("timesteps", 20000))
        n_steps = int(body.get("n_steps", 576))
        n_envs = int(body.get("n_envs", 1))
        name = body.get("name") or None
        seed_ = int(body.get("seed", 0))
        try:
            st = training_worker.start(timesteps, n_steps=n_steps, name=name, seed=seed_,
                                       n_envs=n_envs)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return st

    @app.post("/api/train/stop")
    async def train_stop():
        training_worker.stop()
        return {"status": "stopping"}

    @app.get("/api/train/status")
    async def train_status():
        return training_worker.status()

    WS_INTERVAL = 0.20

    # ------------------------------------------------------------------
    # Live comparison — Traditional vs AI (vs none), stepped side by side
    # under identical conditions, streamed over its own WebSocket channel.
    # ------------------------------------------------------------------

    @app.post("/api/live_compare/start")
    async def live_compare_start(body: dict):
        strategies = body.get("strategies", ["classical", "rl"])
        seed = int(body.get("seed", 42))
        if not isinstance(strategies, list) or not (2 <= len(strategies) <= 3):
            raise HTTPException(status_code=400, detail="Select 2 or 3 strategies to compare.")
        if any(s not in ("none", "classical", "rl") for s in strategies):
            raise HTTPException(status_code=400, detail="Unknown strategy.")
        if "rl" in strategies and not worker.expected_obs_dim():
            raise HTTPException(status_code=400, detail="Network not ready.")
        if "rl" in strategies and not Path(worker.rl_model_path() + ".zip").exists():
            raise HTTPException(status_code=400, detail="No AI model available — train one in the AI Training screen.")
        await asyncio.get_event_loop().run_in_executor(
            None, compare_worker.configure_and_start, strategies, seed, worker.rl_model_path())
        return compare_worker.status()

    @app.post("/api/live_compare/stop")
    async def live_compare_stop():
        compare_worker.stop()
        return {"status": "stopped"}

    @app.post("/api/live_compare/resume")
    async def live_compare_resume():
        compare_worker._running.set()
        return {"status": "resumed"}

    @app.post("/api/live_compare/speed")
    async def live_compare_speed(body: dict):
        return {"status": "ok", "speed_factor": compare_worker.set_speed_factor(body.get("speed_factor", 0.0))}

    @app.get("/api/live_compare/status")
    async def live_compare_status():
        return compare_worker.status()

    @app.websocket("/ws/compare")
    async def websocket_compare(websocket: WebSocket):
        await websocket.accept()
        active_ws_compare.add(websocket)
        last_send = 0.0
        try:
            while True:
                payload = compare_worker.get_state_nowait()
                now = asyncio.get_event_loop().time()
                if payload is not None and (now - last_send) >= WS_INTERVAL:
                    msg = json.dumps(payload)
                    dead = set()
                    for ws in list(active_ws_compare):
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            dead.add(ws)
                    active_ws_compare.difference_update(dead)
                    last_send = now
                else:
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            active_ws_compare.discard(websocket)

    @app.post("/sim/start")
    async def sim_start():
        worker.start()
        return {"status": "started"}

    @app.post("/sim/stop")
    async def sim_stop():
        worker.stop()
        return {"status": "stopped"}

    @app.post("/sim/reset")
    async def sim_reset():
        await asyncio.get_event_loop().run_in_executor(None, worker.reset)
        return {"status": "reset"}

    @app.post("/control")
    async def set_control(body: dict):
        ctrl = body.get("type", "classical")
        worker.set_controller(ctrl)
        return {"status": "ok", "controller": ctrl}

    @app.post("/sim/speed")
    async def sim_speed(body: dict):
        return {"status": "ok", "speed_factor": worker.set_speed_factor(body.get("speed_factor", 0.0))}

    @app.get("/status")
    async def status():
        return {"built": worker.is_built,
                "running": worker.is_built and worker._running.is_set(),
                "step": worker._step, "sim_time": worker._sim_ts.isoformat(),
                "controller": worker._controller_type, "speed_factor": worker.get_speed_factor(),
                "rl_available": worker._rl_available, "rl_model_path": worker.rl_model_path(),
                "training": training_worker.status(include_history=False)}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        active_ws.add(websocket)
        last_send = 0.0
        try:
            while True:
                payload = worker.get_state_nowait()
                now = asyncio.get_event_loop().time()
                if payload is not None and (now - last_send) >= WS_INTERVAL:
                    msg = json.dumps(payload)
                    dead = set()
                    for ws in list(active_ws):
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            dead.add(ws)
                    active_ws.difference_update(dead)
                    last_send = now
                else:
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            active_ws.discard(websocket)

    return app, uvicorn


def run_network_dashboard(
    config_path: str = DEFAULT_CONFIG,
    dt_seconds: float = 600.0,
    speed_factor: float = 0.0,
    port: int = 8060,
    seed: int = 42,
    rl_model_path: str = DEFAULT_RL_MODEL,
) -> None:
    print(f"\n{'='*60}")
    print(f"  HyTwin — Italian H2 Network Control Room")
    print(f"{'='*60}")
    print(f"  URL:    http://localhost:{port}")
    print(f"  Config: {config_path}")
    print(f"  RL:     {'available' if Path(str(rl_model_path)+'.zip').exists() else 'not found (train first)'}")
    print(f"  Speed:  {'max' if speed_factor == 0 else f'{speed_factor}x'}")
    print(f"  Press Ctrl-C to stop\n")
    app, uvicorn = create_app(config_path, dt_seconds, speed_factor, seed, rl_model_path)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HyTwin — Italian H2 Network Control Room")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="scenario YAML (multi-site)")
    ap.add_argument("--port", type=int, default=8060, help="HTTP port (default 8060)")
    ap.add_argument("--dt", type=float, default=600.0, help="step duration [s] (default 600)")
    ap.add_argument("--speed-factor", type=float, default=0.0,
                    help="wall-clock speed: 0 = as fast as possible, e.g. 60 = 1 sim-min/s")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--rl-model", default=DEFAULT_RL_MODEL, help="path to trained RL model (no .zip)")
    args = ap.parse_args()

    run_network_dashboard(config_path=args.config, dt_seconds=args.dt,
                          speed_factor=args.speed_factor, port=args.port,
                          seed=args.seed, rl_model_path=args.rl_model)
