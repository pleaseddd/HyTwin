"""
Network RL Trainer
==================
Thin helper to train a PPO agent on :class:`NetworkH2GridEnv` and save it, so a
:class:`~hytwin.control.network_rl_controller.NetworkRLController` can then be
compared against the traditional controller on identical conditions.

GPU acceleration
----------------
SB3 automatically uses CUDA when available (``device="auto"``).  For the
small MLPs used here the main throughput bottleneck is environment stepping,
not the network update.  Pass ``n_envs > 1`` to spawn that many parallel
worker processes via ``SubprocVecEnv`` — each runs its own simulation in a
separate CPU core, multiplying sample throughput proportionally.

Recommended settings on a GPU machine with ≥4 CPU cores::

    train_network_agent(topo, timesteps=500_000, n_envs=4, n_steps=576)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .network_environment import NetworkH2GridEnv, NetworkRewardConfig

logger = logging.getLogger(__name__)


def make_env(topology, dt_seconds: float = 600.0, episode_steps: int = 144,
             reward_config: Optional[NetworkRewardConfig] = None,
             forecast: str = "none") -> NetworkH2GridEnv:
    return NetworkH2GridEnv(
        topology, dt_seconds=dt_seconds, episode_steps=episode_steps,
        reward_config=reward_config, forecast=forecast,
    )


def train_network_agent(
    topology,
    timesteps: int = 100_000,
    dt_seconds: float = 600.0,
    episode_steps: int = 144,
    save_path: Optional[str] = None,
    seed: int = 0,
    n_steps: int = 288,
    n_envs: int = 1,
    reward_config: Optional[NetworkRewardConfig] = None,
    verbose: int = 0,
    callback=None,
    device: str = "auto",
    normalize_reward: bool = False,
    eval_every: int = 25_000,
    eval_seeds=range(900, 910),
    forecast: str = "none",
):
    """
    Train a PPO agent on the network env.  Returns the trained model.

    Requires stable-baselines3.  ``save_path`` (without extension) saves a
    ``.zip`` loadable by :class:`NetworkRLController`.

    A ``best_model.zip`` is saved alongside whenever the deterministic policy
    reaches a new best mean reward on a fixed set of evaluation days
    (``eval_seeds``, checked every ``eval_every`` steps).  The ``_find_rl_model``
    auto-discovery function in the demo script and dashboard prioritises this
    file over the most-recent timestamped checkpoint.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments (default 1).  Values > 1 launch
        ``SubprocVecEnv`` workers — one per CPU core — and multiply sample
        throughput without changing the learning algorithm.  Each worker gets
        a distinct seed (``seed + i``).
    normalize_reward : bool
        Wrap the env in ``VecNormalize(norm_obs=False, norm_reward=True)``:
        rewards are divided by a running std of the discounted return.
        Observations are untouched, so the saved model needs no stats.
    device : str
        PyTorch device string passed to SB3 (default ``"auto"`` → CUDA if
        available, else CPU).
    forecast : str
        ``"none"`` (default), ``"noisy"`` or ``"perfect"``: per-site multi-day
        forecasts in the observation (see :class:`NetworkH2GridEnv`).
    callback : stable_baselines3.common.callbacks.BaseCallback, optional
        Passed through to ``model.learn()`` — used by the dashboard's
        ``TrainingWorker`` to report live progress and to abort early
        (a callback whose ``_on_step`` returns ``False`` stops training).
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except ImportError as e:  # pragma: no cover
        raise ImportError("stable-baselines3 is required to train the network agent") from e

    n_envs = max(1, int(n_envs))

    if n_envs == 1:
        env = Monitor(make_env(topology, dt_seconds, episode_steps, reward_config, forecast))
    else:
        def _make_factory(i: int):
            def _factory():
                e = make_env(topology, dt_seconds, episode_steps, reward_config, forecast)
                e.reset(seed=seed + i)
                return Monitor(e)
            return _factory

        # "fork" does not exist on Windows; "spawn" needs the caller's __main__ guard.
        import multiprocessing as mp
        start = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        env = SubprocVecEnv([_make_factory(i) for i in range(n_envs)], start_method=start)
        logger.info("SubprocVecEnv: %d parallel workers", n_envs)

    if normalize_reward:
        if n_envs == 1:
            env = DummyVecEnv([lambda e=env: e])
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)

    total_buffer = n_steps * n_envs
    batch_size = min(256, total_buffer)

    model = PPO(
        "MlpPolicy", env,
        n_steps=n_steps, batch_size=batch_size,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
        learning_rate=3e-4, seed=seed, verbose=verbose,
        device=device,
    )

    # Build final callback list: optional dashboard callback + best-model saver
    # ._inner: SB3 requires a real BaseCallback, not the delegating wrapper
    best_cb = (_eval_best_callback(BaseCallback, make_env(topology, dt_seconds, episode_steps, reward_config, forecast),
                                   save_path, list(eval_seeds), max(1, eval_every // n_envs))
               if save_path else None)
    cbs = [c for c in [callback, best_cb] if c is not None]
    final_cb = CallbackList(cbs) if len(cbs) > 1 else (cbs[0] if cbs else None)

    model.learn(total_timesteps=int(timesteps), callback=final_cb)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info("Saved network RL model to %s.zip", save_path)
    return model


def _eval_best_callback(BaseCallback, env, save_path: str, seeds, every_calls: int):
    """Save <model_dir>/best_model.zip when the deterministic policy improves on fixed days.

    The training-episode mean used before compared different, random days (and a
    stochastic policy), so it picked lucky checkpoints rather than better ones.
    """
    best_path = str(Path(save_path).parent / "best_model")

    class _EvalBest(BaseCallback):
        def __init__(self):
            super().__init__()
            self._best = -float("inf")

        def _on_step(self) -> bool:
            if self.n_calls % every_calls:
                return True
            total = 0.0
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = env.step(action)
                    total += r
                    done = terminated or truncated
            mean = total / len(seeds)
            self.logger.record("eval/mean_reward", mean)
            if mean > self._best:
                self._best = mean
                self.model.save(best_path)
                logger.info("New best model (eval mean reward %.2f) → %s.zip", mean, best_path)
            return True

    return _EvalBest()
