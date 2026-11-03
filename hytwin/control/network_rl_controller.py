"""
Network RL Controller
=====================
Wraps a trained Stable-Baselines3 policy (trained on
:class:`~hytwin.rl.network_environment.NetworkH2GridEnv`) so it can be used as a
drop-in network controller — same ``compute_actions(prev_state, ts)`` interface
as :class:`NetworkClassicalController`, so it plugs straight into the
reproducible comparison harness (``hytwin.network.compare``).

It rebuilds the exact per-site-factored observation the env produced and decodes
the policy's per-site action block back into ``actions_by_site`` for
``NetworkTwin.step``.  Models trained with per-site forecasts (longer observation)
need the running ``NetworkTwin``: the forecasts come from a shadow copy of its
exogenous models, taken at the first call, exactly as the env does at reset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..rl.network_environment import (
    NetworkH2GridEnv, N_SITE_ACT, FORECAST_WINDOWS_H, _ExogShadow,
)


class NetworkRLController:
    """Trained-policy network controller."""

    name = "NetworkRLController"

    def __init__(self, model, topology, twin=None, forecast: str = "noisy") -> None:
        self._model = model
        self._topo = topology
        self._twin = twin
        # Reuse the env purely for its obs-building / normalisation constants.
        self._ref = NetworkH2GridEnv(topology)
        n_obs = int(model.observation_space.shape[0])
        if n_obs != self._ref.observation_space.shape[0]:
            self._ref = NetworkH2GridEnv(topology, forecast=forecast)
            if n_obs != self._ref.observation_space.shape[0]:
                raise ValueError(f"model observation {n_obs}-D does not fit this network")
            if twin is None:
                raise ValueError("a model with site forecasts needs the running NetworkTwin (twin=...)")
        self._t0: Optional[datetime] = None
        self._site_ids: List[str] = list(topology.site_ids)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model_path(cls, model_path, topology, twin=None) -> "NetworkRLController":
        model = _load_sb3(model_path)
        return cls(model, topology, twin)

    @classmethod
    def factory(cls, model_path):
        """Return a ``factory(network_twin) -> controller`` for compare.run_network."""
        def _make(network_twin):
            return cls.from_model_path(model_path, network_twin.topology, network_twin)
        return _make

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_actions(self, prev_state, timestamp: datetime):
        obs = self._obs_from_state(prev_state, timestamp)
        action, _ = self._model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.size == N_SITE_ACT * len(self._site_ids):  # model trained with 3 slots per site
            action = action[self._ref._legacy_idx]
        return self._ref._decode(np.clip(action, 0.0, 1.0))

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation reconstruction (mirrors NetworkH2GridEnv)
    # ------------------------------------------------------------------

    def _obs_from_state(self, ns, ts: datetime) -> np.ndarray:
        env = self._ref
        if env._forecast != "none":
            if self._t0 is None:      # forecasts start from the twin as it is now, like env.reset
                self._t0 = ts
                env._fc_rng = np.random.default_rng(0)
                env._fc_err = env._fc_rng.normal(size=(len(FORECAST_WINDOWS_H), env._n, 3))
                env._shadow = _ExogShadow(self._twin)
                env._fut, env._fut_t0 = [], ts
            env._step_count = int(round((ts - self._t0).total_seconds() / env._dt))
        if ns is None:
            feats: List[float] = []
            for sid in self._site_ids:
                feats += [0.5, 0.0, 0.0, 0.30 / 0.5, 1.0, 0.0, 0.0, 0.0]
            feats += env._time_feats(ts)
            feats += env._forecast_feats(ts)
            feats += env._site_forecast_feats()
            return np.clip(np.array(feats, dtype=np.float32), -2.0, 2.0)
        return env._observe(ns)


# ------------------------------------------------------------------
def _load_sb3(model_path):
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ImportError as e:
        raise ImportError("stable-baselines3 is required for NetworkRLController") from e
    last_err = None
    for cls in (PPO, SAC, TD3):
        try:
            return cls.load(model_path)
        except Exception as e:  # noqa: BLE001
            last_err = e
    try:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(model_path)
    except Exception:
        pass
    raise RuntimeError(f"Could not load SB3 model from {model_path}: {last_err}")


def probe_model_dims(model_path) -> Tuple[int, int]:
    """
    Load a saved model just far enough to read the observation/action space it
    was trained with — used to check topology compatibility (site count)
    before activating a model as the live controller, without needing its
    original :class:`NetworkTopology`.
    """
    model = _load_sb3(model_path)
    return int(model.observation_space.shape[0]), int(model.action_space.shape[0])
