"""
test_network_rl.py
=================
F5 tests: the multi-site RL environment (Gymnasium-compliant, per-node factored
obs/action, closed energy balance), plus a short end-to-end train → wrap →
compare loop proving the RL controller plugs into the reproducible harness.
Run: pytest tests/test_network_rl.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.rl.network_environment import NetworkH2GridEnv

PILOT = ROOT / "config" / "italy_network_pilot.yaml"
_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


def _topo():
    return Scenario.from_yaml(PILOT).topology()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def test_spaces_shape():
    env = NetworkH2GridEnv(_topo())
    # 3 sites -> 8*3 site obs + 2 time feats + 6 forecast feats (2 horizons x 3), 3*3 act
    assert env.observation_space.shape == (32,)
    assert env.action_space.shape == (9,)


def test_actions_only_for_existing_components():
    # large: trapani/cagliari no FC, roma/bologna/milano no EL -> 21 - 5 = 16 slots
    env = NetworkH2GridEnv(Scenario.from_yaml(ROOT / "config" / "italy_network_large.yaml").topology())
    assert env.action_space.shape == (16,)
    acts = env._decode(np.ones(16, dtype=np.float32))
    assert not any(k.startswith("fc") for k in acts["trapani"])
    assert sum(len(a) for a in acts.values()) == sum(
        len(env._el_ids[s]) + len(env._fc_ids[s]) + len(env._load_ids[s]) for s in env.site_ids)
    # legacy 3-per-site models: slot -> old index, el of trapani is 0, fc of roma is 4*3+1
    assert env._legacy_idx[0] == 0 and (("roma", "fc") in env._act_slots)
    assert env._legacy_idx[env._act_slots.index(("roma", "fc"))] == 13


def test_reset_and_step():
    env = NetworkH2GridEnv(_topo())
    obs, info = env.reset(seed=1)
    assert obs.shape == (32,) and np.all(np.isfinite(obs))
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (32,) and np.isfinite(r)
    assert term is False
    assert "network_state" in info


def test_gymnasium_compliant():
    from gymnasium.utils.env_checker import check_env
    check_env(NetworkH2GridEnv(_topo()), skip_render_check=True)


def test_episode_truncates_at_length():
    env = NetworkH2GridEnv(_topo(), episode_steps=10)
    env.reset(seed=0)
    trunc = False
    for _ in range(10):
        _, _, _, trunc, _ = env.step(env.action_space.sample())
    assert trunc is True


def test_env_balance_closes():
    env = NetworkH2GridEnv(_topo())
    env.reset(seed=3)
    for _ in range(48):
        _, _, _, _, info = env.step(env.action_space.sample())
        for n in info["network_state"].nodes.values():
            supply = n.generation_kw + n.link_import_kw + n.grid_import_kw
            sink = (n.demand_kw + n.link_export_kw
                    + n.grid_export_kw + n.curtailed_kw - n.unmet_kw)
            assert abs(supply - sink) < 1e-6


def test_reproducible_reset_seed():
    env = NetworkH2GridEnv(_topo())
    env.reset(seed=7); a = env.step(np.zeros(9, dtype=np.float32))[1]
    env.reset(seed=7); b = env.step(np.zeros(9, dtype=np.float32))[1]
    assert a == pytest.approx(b, rel=1e-9)


# ---------------------------------------------------------------------------
# Train → wrap → compare (short smoke)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
def test_train_wrap_and_compare(tmp_path):
    from hytwin.rl.network_trainer import train_network_agent
    from hytwin.control.network_rl_controller import NetworkRLController
    from hytwin.network.compare import run_network
    import logging
    logging.disable(logging.INFO)

    topo = _topo()
    save = str(tmp_path / "net_ppo_test")
    train_network_agent(topo, timesteps=1024, episode_steps=48, save_path=save, seed=0, n_steps=128)

    # Wrapped controller produces well-formed per-site actions.
    ctrl = NetworkRLController.from_model_path(save, topo)
    _, results = run_network(topo, 48, Scenario.from_yaml(PILOT).start_time, 600.0,
                             seed=42, controller=NetworkRLController.factory(save))
    assert len(results) == 48
    # Balance still closes under RL control.
    for n in results[-1].nodes.values():
        supply = n.generation_kw + n.link_import_kw + n.grid_import_kw
        sink = (n.demand_kw + n.link_export_kw
                + n.grid_export_kw + n.curtailed_kw - n.unmet_kw)
        assert abs(supply - sink) < 1e-6
    # cold-start action shape sanity
    from datetime import datetime
    a0 = ctrl.compute_actions(None, datetime(2024, 6, 15))
    assert set(a0.keys()) == {"foggia", "napoli", "milano"}
