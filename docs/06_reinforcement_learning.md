# Reinforcement Learning — HyTwin

HyTwin exposes three [Gymnasium](https://gymnasium.farama.org/)-compatible
environments, from single-site to full network scale. The **network
environment is the primary, recommended path** — see `05_network_layer.md`
for the network-specific reward and controller design. This document covers
all three, plus the shared Stable-Baselines3 training infrastructure.

---

## 1. Environments at a glance

| Environment | Scope | Obs | Actions | Notes |
|-------------|-------|-----|---------|-------|
| `H2GridEnv` | single site | 14-D | 3-D | Baseline env (electrolyzer · fuel cell · demand response) |
| `AdvancedH2GridEnv` | single site | 19-D | 4-D | Adds grid import fraction, price signal, outage flag |
| `NetworkRLEnv` (`NetworkH2GridEnv`) | multi-site network | per-node factored, `Box(-2, 2, shape=(obs_dim,))` | per-node factored, `Box(0, 1, shape=(act_dim,))` | `obs_dim`/`act_dim` scale with site/link count — see `05_network_layer.md` §6.2 |

`hytwin/rl/environment.py` → `H2GridEnv`; `hytwin/rl/advanced_environment.py`
→ `AdvancedH2GridEnv`; `hytwin/rl/network_environment.py` →
`NetworkH2GridEnv` + `NetworkRewardConfig`.

---

## 2. Single-site: `H2GridEnv`

### Constructor

```python
from hytwin.rl.environment import H2GridEnv

env = H2GridEnv(
    config_path="config/default_grid.yaml",
    episode_length=144,     # steps per episode (144 = 24h at 10-min steps)
    reward_config=None,     # optional RewardConfig
    render_mode=None,
)
obs, info = env.reset(seed=42)
```

### Observation space (14-D, normalised to [0,1])

| Idx | Variable | Physical range | Notes |
|-----|----------|-----------------|-------|
| 0 | `wind_power_norm` | 0–1000 kW | sum of wind turbines |
| 1 | `pv_power_norm` | 0–300 kW | PV array |
| 2 | `electrolyzer_power_norm` | 0–300 kW | |
| 3 | `fuel_cell_power_norm` | 0–150 kW | |
| 4 | `load_norm` | 0–400 kW | |
| 5 | `h2_soc` | 0–1 | already normalised |
| 6 | `h2_pressure_norm` | 0–700 bar | |
| 7 | `grid_exchange_norm` | −1000…+1000 kW | mapped to [0,1] |
| 8 | `renewable_fraction` | 0–1 | already normalised |
| 9 | `grid_self_sufficiency` | 0–1 | already normalised |
| 10 | `time_of_day_sin` | −1…+1 | mapped to [0,1] |
| 11 | `time_of_day_cos` | −1…+1 | mapped to [0,1] |
| 12 | `wind_speed_norm` | 0–25 m/s | from WeatherModel |
| 13 | `irradiance_norm` | 0–1000 W/m² | GHI |

### Action space (3-D, continuous in [-1, 1])

| Idx | Action | Physical mapping |
|-----|--------|--------------------|
| 0 | `el_setpoint` | $(a+1)/2 \to [0,1]$ electrolyzer setpoint |
| 1 | `fc_setpoint` | $(a+1)/2 \to [0,1]$ fuel-cell setpoint |
| 2 | `dr_fraction` | $\max(0,(a+1)/2 \times \text{max\_dr}) \to [0, 0.2]$ demand-response fraction |

### Reward (single-site `RewardConfig`)

A 7-component weighted sum: renewable fraction, self-sufficiency, H₂
production/consumption balance, component efficiency, H₂ SOC safety
(penalises going outside a safe band), demand comfort (unmet-load penalty),
and a grid-exchange penalty. See `hytwin/rl/rewards.py` for exact weights.

### Episode lifecycle

```
env.reset(seed=42)
   ├── Rebuilds SimulationEngine from YAML
   ├── Resets WeatherModel, StateManager, TimeSeriesRecorder
   ├── Samples initial conditions (tank SOC ~ U[0.3, 0.7])
   └── Returns obs[0], info{}

loop (step in range(episode_length)):
   env.step(action)
       ├── Converts action → control_actions
       ├── Calls engine.step_once() → GridTwin.step() → GridState
       ├── Computes reward(GridState)
       ├── Computes obs[t+1]
       ├── terminated = (step >= episode_length - 1) or (h2_soc < 0.01)
       └── Returns obs, reward, terminated, truncated, info
```

`info["reward_components"]` exposes the individual weighted terms every
step, for debugging and for the dashboard's live objective-term breakdown.

---

## 3. `AdvancedH2GridEnv`

Same structure as `H2GridEnv`, extended to 19-D observations (adds grid
import fraction, the local price signal, and an outage flag) and a 4-D
action space (adds explicit grid-import control on top of electrolyzer/
fuel-cell/DR). Used by `demos/demo_advanced.py --mode train_rl`.

---

## 4. Network: `NetworkH2GridEnv` / `NetworkRLEnv`

See `05_network_layer.md` §6.2–§7 for the full design rationale (per-node
factored obs/action, and the `NetworkRewardConfig` reward). In short:

```python
from hytwin.simulation.scenario import Scenario
from hytwin.rl.network_environment import NetworkH2GridEnv, NetworkRewardConfig

topo = Scenario.from_yaml("config/italy_network_large.yaml").topology()
env = NetworkH2GridEnv(topo, dt_seconds=600.0, episode_steps=144,
                        reward_config=NetworkRewardConfig())
```

`observation_space` is `Box(-2.0, 2.0, shape=(obs_dim,))` and
`action_space` is `Box(0.0, 1.0, shape=(act_dim,))`, where `obs_dim`/
`act_dim` scale with the number of sites and links in `topology` — a
7-node/12-link topology like `italy_network_large.yaml` has a
correspondingly larger space than the 3-node `italy_network_pilot.yaml`.

### Training from the CLI

```bash
# recommended: dedicated entry point with all options
python -m train --timesteps 500000 --n-envs 4 --seed 0
python -m train --help          # full option list
```

`python -m train` wraps `train_network_agent()` with argparse — same
defaults, no boilerplate. Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--timesteps` | 200 000 | total environment steps; a quick run. On the large network about 2M are needed to beat the classical controller over 30 days (0.5M are still worse) |
| `--n-envs` | 4 | parallel `SubprocVecEnv` workers (one per CPU core) |
| `--n-steps` | 576 | PPO rollout length per env |
| `--seed` | 0 | RNG seed |
| `--config` | `italy_network_large.yaml` | network scenario |
| `--save` | `output/rl_models/net_ppo_large` | output path (no `.zip`) |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

**Parallel environments** (`n_envs > 1`) use `SubprocVecEnv` to run
independent simulations in separate processes, each on its own CPU core.
The policy update runs on GPU when CUDA is available (`device="auto"`).
Measured speedup on an RTX 2080: **~3× with 4 envs** over a single env,
since the bottleneck is environment stepping, not the MLP update.

`train_network_agent(topology, …)` is the underlying Python API; it accepts
a `callback` argument used by the dashboard's AI Training screen to report
live progress and abort early (`_on_step` returning `False` stops training).

For a real, long training run, prefer the CLI over the dashboard's Training
screen, since a CLI run is not tied to the browser session.

---

## 5. Algorithms

- **PPO** (Stable-Baselines3) — the primary algorithm for both single-site
  and network environments, and the only one wired into the dashboard's live
  training and `rl` controller mode.
- **SAC / TD3 / DDPG** — also available for the single-site environments via
  `RLTrainer` (`hytwin/rl/trainer.py`), for offline experimentation.

---

## 6. Classical baseline

`ClassicalController` (single-site) / `NetworkClassicalController`
(network) provide a cost-aware, rule-based dispatch baseline for direct
comparison against trained agents — see `05_network_layer.md` §6.1 for the
network-aware electric-transport-first / green-H₂-first policy, and
`09_usage_guide.md` for how to run the comparison end to end.

---

## 7. Interpreting training results

There is **no fixed "optimal" reward value** to target — see
`05_network_layer.md` §7 for why the network reward is an unbounded,
episode-condition-dependent weighted sum. The two things that *are*
meaningful:

1. **Relative performance vs. `none`/`classical`**, computed under the
   identical reward formula on the same seed (`compare_controllers`).
2. The **learning curve's smoothed trend** (not its instantaneous value) —
   the dashboard's AI Training screen plots a smoothed episode-reward curve
   sourced from SB3's `Monitor` wrapper's per-episode `info["episode"]`
   records; look at whether the trend is rising and flattening, not at any
   single point on it.

For the single-site `H2GridEnv`, indicative reference numbers from a
short PPO run against a random-action baseline (default single-site config,
20k timesteps) are: mean reward ~506 (random) → ~520 (PPO), mean renewable
fraction ~0.42 → ~0.51, mean self-sufficiency ~0.65 → ~0.72, H₂ safety
violations ~12% → ~3%. These are illustrative of the single-site
environment only — see `05_network_layer.md` §6.2 for the corresponding
network-scale result (PPO vs. rule-based on the 7-node pilot).
