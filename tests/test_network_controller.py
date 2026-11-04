"""
test_network_controller.py
=========================
F4 tests: NetworkClassicalController drives per-site assets, the comparison
harness is reproducible (identical conditions across strategies), and the
energy balance still closes under rule-based control.
Run: pytest tests/test_network_controller.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.network.network_twin import NetworkTwin
from hytwin.network.compare import run_network, compare_controllers, aggregate_network_kpis
from hytwin.control.network_controller import NetworkClassicalController

PILOT_YAML = ROOT / "config" / "italy_network_pilot.yaml"
DT = 600.0
START = datetime(2024, 6, 15, 0, 0)


def _topo():
    return Scenario.from_yaml(PILOT_YAML).topology()


# ---------------------------------------------------------------------------
# Controller wiring
# ---------------------------------------------------------------------------

def test_controller_builds_one_per_site():
    np.random.seed(0)
    tw = NetworkTwin(_topo())
    ctrl = NetworkClassicalController.from_network(tw)
    assert set(ctrl._controllers.keys()) == {"foggia", "napoli", "milano"}


def test_controller_cold_start_returns_empty():
    np.random.seed(0)
    tw = NetworkTwin(_topo())
    ctrl = NetworkClassicalController.from_network(tw)
    assert ctrl.compute_actions(None, START) == {}


def test_controller_produces_actions_after_first_step():
    np.random.seed(0)
    tw = NetworkTwin(_topo())
    ctrl = NetworkClassicalController.from_network(tw)
    ns = tw.step(DT, START)                     # cold start (naive)
    actions = ctrl.compute_actions(ns, START + timedelta(seconds=DT))
    assert set(actions.keys()) == {"foggia", "napoli", "milano"}
    # each site should have at least one component setpoint
    assert any(actions[s] for s in actions)


# ---------------------------------------------------------------------------
# Balance still closes under rule-based control
# ---------------------------------------------------------------------------

def test_balance_closes_under_classical():
    kpis, results = run_network(_topo(), 144, START, DT, seed=1, controller="classical")
    for ns in results:
        for n in ns.nodes.values():
            supply = n.generation_kw + n.link_import_kw + n.grid_import_kw
            sink = n.demand_kw + n.link_export_kw + n.grid_export_kw + n.curtailed_kw - n.unmet_kw
            assert abs(supply - sink) < 1e-6


# ---------------------------------------------------------------------------
# Reproducibility: identical conditions across strategies
# ---------------------------------------------------------------------------

def test_reproducibility_same_seed_same_conditions():
    """
    With RNG-neutral controller price reads, two runs at the same seed must see
    the *same weather* — total renewable generation is identical whether the
    baseline or the classical controller is used.
    """
    _, res_none = run_network(_topo(), 72, START, DT, seed=99, controller="none")
    _, res_cls = run_network(_topo(), 72, START, DT, seed=99, controller="classical")
    renew_none = np.array([r.total_renewable_kw for r in res_none])
    renew_cls = np.array([r.total_renewable_kw for r in res_cls])
    assert np.allclose(renew_none, renew_cls, atol=1e-6)


def test_same_seed_repeated_run_is_deterministic():
    k1, _ = run_network(_topo(), 48, START, DT, seed=7, controller="classical")
    k2, _ = run_network(_topo(), 48, START, DT, seed=7, controller="classical")
    assert k1["cost_eur"] == pytest.approx(k2["cost_eur"], rel=1e-9)


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------

def test_compare_controllers_returns_kpis():
    out = compare_controllers(_topo(), steps=72, seed=5)
    assert set(out.keys()) == {"none", "classical"}
    for kpis in out.values():
        assert kpis["load_kwh"] > 0
        assert 0.0 <= kpis["reliability"] <= 1.0
        assert kpis["cost_eur"] == kpis["cost_eur"]  # not NaN


def test_load_served_identical_across_strategies():
    """Same real demand under both strategies (fair comparison sanity)."""
    out = compare_controllers(_topo(), steps=72, seed=5)
    assert out["none"]["load_kwh"] == pytest.approx(out["classical"]["load_kwh"], rel=1e-6)


def test_classical_optimises_vs_naive_over_week():
    """
    The network-aware rule-based controller must genuinely optimise: over a week
    it should cost less (storage-adjusted) and emit less CO₂ than the naive
    baseline, while not degrading reliability — the point of 'traditional control'.
    """
    out = compare_controllers(_topo(), steps=1008, seed=42)
    n, c = out["none"], out["classical"]
    assert c["storage_adjusted_cost_eur"] < n["storage_adjusted_cost_eur"]
    assert c["co2_kg"] <= n["co2_kg"]
    assert c["reliability"] >= n["reliability"] - 1e-9


def test_kpi_h2_value_matches_reward():
    """The dashboard's storage-adjusted cost must value H2 like the reward does."""
    import inspect
    from hytwin.rl.network_environment import NetworkRewardConfig
    default = inspect.signature(aggregate_network_kpis).parameters["h2_value_eur_per_kg"].default
    assert default == NetworkRewardConfig().h2_value_eur_per_kg
