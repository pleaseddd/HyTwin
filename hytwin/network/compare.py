"""
Network Controller Comparison
=============================
Reproducible side-by-side evaluation of control strategies on a multi-site
network under **identical real conditions** (same weather / price / outage RNG
stream).  This is the network-level analogue of the single-site
``experiments/control_comparison`` harness, and the foundation for the
Traditional-vs-AI comparison (F5).

Strategies
----------
* ``"none"``      — NetworkTwin naive defaults (dumb baseline).
* ``"classical"`` — per-site cost-aware :class:`NetworkClassicalController`.
* a callable ``factory(network_twin) -> controller`` with a
  ``compute_actions(prev_state, ts)`` method (e.g. a future RL controller).

Fairness: each run seeds the global RNG identically and rebuilds a fresh
NetworkTwin; controller price reads are RNG-neutral (see
:class:`~hytwin.control.network_controller._RngSafeCostModel`), so both runs see
the same weather and prices — only the control decisions differ.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .network_twin import NetworkTwin
from .network_state import NetworkState
from .topology import NetworkTopology
from ..control.network_controller import NetworkClassicalController


def aggregate_network_kpis(
    results: List[NetworkState],
    dt_seconds: float,
    h2_value_eur_per_kg: float = 1.5,
) -> Dict[str, float]:
    """
    Aggregate a run's per-step NetworkStates into scalar KPIs.

    ``storage_adjusted_cost_eur`` credits/debits the change in total stored H₂
    over the run at ``h2_value_eur_per_kg`` (default 1.5 €/kg, the same value as
    ``NetworkRewardConfig.h2_value_eur_per_kg``: what a kg saves when used).
    Without it, a strategy that simply *drains* H₂ storage looks artificially
    cheap — the raw ``cost_eur`` ignores the value of the reserve it burned.  This makes strategies with different terminal SOC
    comparable, which is essential for a fair Traditional-vs-AI verdict.
    """
    dt_h = dt_seconds / 3600.0
    load_kwh = sum(r.total_load_kw for r in results) * dt_h
    import_kwh = sum(r.total_grid_import_kw for r in results) * dt_h
    export_kwh = sum(r.total_grid_export_kw for r in results) * dt_h
    renew_kwh = sum(r.total_renewable_kw for r in results) * dt_h
    curtail_kwh = sum(r.total_curtailed_kw for r in results) * dt_h
    unmet_kwh = sum(r.unmet_demand_kw for r in results) * dt_h
    inter_p_kwh = sum(r.inter_node_power_kw for r in results) * dt_h
    inter_h2_kg = sum(r.inter_node_h2_kg_h for r in results) * dt_h
    cost = results[-1].cumulative_cost_eur if results else 0.0
    co2 = sum(r.total_co2_kg_step for r in results)

    # H₂ storage delta over the run (final − initial total stored mass).
    def _total_h2(ns):
        return sum(n.h2_storage_kg for n in ns.nodes.values())
    h2_delta = (_total_h2(results[-1]) - _total_h2(results[0])) if results else 0.0
    storage_adjusted_cost = cost - h2_delta * h2_value_eur_per_kg

    return {
        "cost_eur": float(cost),
        "storage_adjusted_cost_eur": float(storage_adjusted_cost),
        "net_h2_delta_kg": float(h2_delta),
        "co2_kg": float(co2),
        "grid_import_kwh": float(import_kwh),
        "grid_export_kwh": float(export_kwh),
        "renewable_kwh": float(renew_kwh),
        "curtailed_kwh": float(curtail_kwh),
        "unmet_kwh": float(unmet_kwh),
        "inter_node_power_kwh": float(inter_p_kwh),
        "inter_node_h2_kg": float(inter_h2_kg),
        "load_kwh": float(load_kwh),
        "self_sufficiency": float(np.mean([r.network_self_sufficiency for r in results])) if results else 0.0,
        "renewable_fraction": float(np.mean([r.network_renewable_fraction for r in results])) if results else 0.0,
        "reliability": float(np.mean([r.reliability_index for r in results])) if results else 0.0,
        "avg_h2_soc": float(np.mean([r.avg_h2_soc for r in results])) if results else 0.0,
    }


def run_network(
    topology: NetworkTopology,
    steps: int,
    start_time: datetime,
    dt_seconds: float,
    seed: int,
    controller: Any = "none",
    soc_target: float = 0.5,
) -> Tuple[Dict[str, float], List[NetworkState]]:
    """
    Run one network scenario with a given control strategy.

    Parameters
    ----------
    controller : "none" | "classical" | callable(NetworkTwin) -> controller
    seed : int
        Seeds an isolated RNG subtree for this run's NetworkTwin (weather,
        sensors, outages, ...) — identical *seed* across strategies gives
        identical conditions, without touching the shared global RNG (so this
        comparison stays reproducible even if something else, e.g. a
        concurrent RL training job, is drawing from the global state at the
        same time).

    Returns
    -------
    (kpis, results)
    """
    twin = NetworkTwin(topology, soc_target=soc_target, seed=seed)

    provider: Optional[Callable] = None
    if controller == "none" or controller is None:
        provider = None
    elif controller == "classical":
        ctrl = NetworkClassicalController.from_network(twin, soc_target=soc_target)
        provider = lambda i, ts, prev: ctrl.compute_actions(prev, ts)  # noqa: E731
    elif callable(controller):
        ctrl = controller(twin)
        provider = lambda i, ts, prev: ctrl.compute_actions(prev, ts)  # noqa: E731
    else:
        raise ValueError(f"Unknown controller {controller!r}")

    results = twin.run(steps, start_time, dt_seconds, actions_provider=provider)
    return aggregate_network_kpis(results, dt_seconds), results


def compare_controllers(
    topology: NetworkTopology,
    steps: int = 144,
    start_time: Optional[datetime] = None,
    dt_seconds: float = 600.0,
    seed: int = 42,
    strategies: Optional[Dict[str, Any]] = None,
    soc_target: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Run several strategies under identical conditions and return their KPIs.

    Returns ``{strategy_name: kpis_dict}``.
    """
    start_time = start_time or datetime(2024, 6, 15, 0, 0)
    strategies = strategies or {"none": "none", "classical": "classical"}
    out: Dict[str, Dict[str, float]] = {}
    for name, ctrl in strategies.items():
        kpis, _ = run_network(topology, steps, start_time, dt_seconds, seed, ctrl, soc_target)
        out[name] = kpis
    return out
