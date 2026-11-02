# How to Use HyTwin — Usage Guide

This is the practical, task-oriented companion to the rest of `docs/`. It
covers, with copy-pasteable commands and UI steps, both the **dashboard**
and the **CLI** paths for the four things you'll actually want to do:
run a network simulation, train an AI agent, compare traditional vs. AI
control, and interpret the results — including how to vary time horizon,
seed, and network configuration along the way.

All CLI snippets below are run from the repository root, with the project's
virtual environment active.

---

## 1. Running a network simulation

### Dashboard

```bash
python -m dashboard --speed-factor 30
# then open http://localhost:8060
```

The simulation is **built but paused** on launch. Press **Start** in the
sidebar to start the clock. The **Overview** screen shows live KPI tiles;
**Network & Map** shows the animated topology; **Analytics & KPIs** shows
time-series charts as the run progresses.

### CLI

There's no dedicated "just simulate and watch" CLI entry point for the
network layer — the CLI path for running a network scenario is through
`compare_controllers` (§3 below) or by driving `NetworkTwin` directly in a
script, e.g.:

```python
from datetime import datetime
from hytwin.simulation.scenario import Scenario
from hytwin.network.network_twin import NetworkTwin

topo = Scenario.from_yaml("config/italy_network_large.yaml").topology()
twin = NetworkTwin(topo, dt_seconds=600.0)
state = twin.reset(start_time=datetime(2024, 6, 15), seed=42)
for i in range(144):   # 24 hours at dt=600s
    state = twin.step()
    print(state.timestamp, state.total_load_kw, state.total_renewable_kw)
```

For a quick sanity check without writing a script, use the comparison
harness with a single strategy (§3) — it prints KPIs after running the
network for the requested number of steps.

---

## 2. Training an AI (RL) agent

### Dashboard — AI Training screen

1. Go to **AI Training** in the sidebar.
2. Set `timesteps`, `n_steps` (rollout size), `n_envs` (parallel workers),
   and `seed`; optionally name the run.
3. Press **▶ Start training** — the job runs in a background thread on the
   server. Watch live progress, ETA, and the smoothed learning-curve chart.
4. When it completes, open the model list and activate it as the live `rl`
   controller (no dashboard restart needed) — the model/topology
   compatibility check runs automatically and blocks activation with a
   clear error if the model was trained on a different network.

**`n_envs`** (default 4): each worker runs a full simulation in a separate
process, multiplying sample throughput proportionally. Set it to the number
of free CPU cores available on the server — 4 gives ~3× speedup over 1.
GPU (if available) is used automatically by SB3 for the policy update.

For a long, unattended training run the CLI is more convenient since it is
not tied to the browser session:

### CLI — recommended for real training runs

```bash
# default: 2M steps (~1-2 h on 4 cores), 4 parallel envs, saves to output/rl_models/net_ppo_large.zip
python -m train

# full options
python -m train --timesteps 500000 --n-envs 4 --n-steps 576 --seed 0
python -m train --help
```

To train a **named variant** (kept alongside the default model, selectable
from the dashboard's model list):

```bash
python -m train --timesteps 500000 --save output/rl_models/net_ppo_v2 --seed 1
```

To use the 3-node pilot network for a quick experiment:

```bash
python -m train --config config/italy_network_pilot.yaml --timesteps 100000
```

---

## 3. Running a Traditional-vs-AI comparison

### Dashboard — Scenario Comparison screen

- **Batch comparison**: pick 2–3 controllers (`none`/`classical`/`rl`),
  press run — a one-shot comparison over a configurable horizon, with
  side-by-side KPIs.
- **Live lockstep comparison**: the same set of controllers stepped forward
  together, in real time, on the same seed — useful for watching *when* and
  *why* one strategy diverges from another rather than just the final KPIs.

### CLI

```bash
python -c "
from hytwin.simulation.scenario import Scenario
from hytwin.network.compare import compare_controllers
from hytwin.control.network_rl_controller import NetworkRLController
topo = Scenario.from_yaml('config/italy_network_large.yaml').topology()
out = compare_controllers(topo, steps=1008, seed=42, strategies={
    'none': 'none', 'classical': 'classical',
    'rl': NetworkRLController.factory('output/rl_models/net_ppo_large')})
for name, kpis in out.items():
    print(name, kpis)
"
```

`compare_controllers` runs every strategy under **identical conditions**
(same seed ⇒ same weather/price draw for every site) and returns a
`{strategy_name: kpis_dict}` mapping — this is the only fair way to compare
controllers, since the network's behaviour depends heavily on the weather
and price draw of the episode.

---

## 4. Varying conditions

### 4.1 Time horizon

The `steps` argument (CLI) — or the horizon selector in the dashboard's
Scenario Comparison screen — controls how long a run/comparison lasts, at
`dt_seconds=600` (10-minute steps) by default:

| Horizon | `steps` | Use for |
|---------|---------|---------|
| A few hours | `steps=24` (4 hours) or similar | Quick sanity check of a config or a controller change |
| One day | `steps=144` | Standard demo / sanity run |
| One week | `steps=1008` | Statistically meaningful controller comparison — enough variety in weather/price to trust the KPI differences |
| One month | `steps≈4320` | Deeper validation, e.g. before trusting a training result |

Rule of thumb: use a short horizon while iterating on a config or reward
tweak (fast turnaround), and a week-or-longer horizon for any comparison
whose numbers you intend to actually trust — a single day can easily be
dominated by one lucky/unlucky weather draw.

```python
# short check
out = compare_controllers(topo, steps=24, seed=42, strategies={...})
# week-long, trustworthy comparison
out = compare_controllers(topo, steps=1008, seed=42, strategies={...})
```

### 4.2 Seed

The `seed` argument controls the random weather/price draw (and any
stochastic sensor faults). Two uses:

- **Reproducibility**: re-running with the same `seed` and the same config
  reproduces the exact same weather, prices, and outcome — use this when
  debugging a specific run or when you need a stable baseline to compare
  against across code changes.
- **Robustness / exploration**: sweep several seeds to see whether a
  controller's advantage holds up across different weather/price draws, not
  just one lucky episode:

```python
for seed in (0, 1, 2, 3, 4):
    out = compare_controllers(topo, steps=1008, seed=seed,
                               strategies={"classical": "classical", "rl": rl_ctrl})
    print(seed, out["classical"], out["rl"])
```

A trained agent that only beats the classical controller on one seed is not
yet trustworthy — check a handful of seeds before believing a training
result.

### 4.3 Network configuration

Three ways to point at a different (or edited) network, from least to most
interactive:

1. **Dashboard Configuration screen** (`07_dashboard.md` §11): pick a
   scenario from the dropdown, edit the YAML in the textarea, **Validate &
   Save**, then **Activate**. Takes effect on the next `/sim/reset`,
   training run, or comparison run — never on one already in progress.
2. **`--config` CLI flag** on the dashboard launcher:
   ```bash
   python -m dashboard --config config/italy_network_pilot.yaml --speed-factor 30
   ```
3. **`Scenario.from_yaml(...)`** pointed at a different file in a script:
   ```python
   topo = Scenario.from_yaml("config/italy_network_pilot.yaml").topology()
   ```

To create a genuinely new network (not just re-tune an existing one), copy
one of the existing scenario files under `config/`, then follow
`08_configuration_reference.md` §3.3 ("How to add a new node") — duplicate
a `- id: ...` site block, give it a new id, adjust its equipment and links.
Per-sensor fault frequency (how often a given sensor misbehaves) is tuned
via each sensor's `fault_probability` field — see
`08_configuration_reference.md` §7.

---

## 5. Interpreting the results

### KPIs

| KPI | Meaning |
|-----|---------|
| Self-sufficiency | Fraction of demand met without national-grid import |
| Reliability | Fraction of demand actually met (1 − unmet-demand fraction) |
| Renewable fraction | Share of energy served by wind/PV/H₂-from-renewables vs. grid import |
| Average H₂ SOC | Mean state of charge across all sites' tanks — a healthy controller keeps this near its target band, not draining it to zero |
| Cost | Total network energy cost (€) over the run |
| CO₂ | Total grid-import carbon emissions (kg) over the run |
| Grid import | Total energy imported from the national grid across all sites |
| Inter-node exchange | Total energy/H₂ moved between sites over the links — a sign the network topology itself is being used, not just each site's own grid connection |

### Reward / learning curve

There is **no fixed "optimal" reward value** — `NetworkRewardConfig`
produces an unbounded weighted sum whose achievable range depends on that
episode's specific weather and price draw (see `05_network_layer.md` §7).
Do not compare a raw reward number against a hardcoded target. Instead:

1. Compare **relative performance** vs. `none`/`classical` on the **same
   seed**, via `compare_controllers` — this is the only apples-to-apples
   comparison.
2. When training, watch the **learning curve's smoothed trend** (rising and
   flattening = converging), not any single episode's reward — the
   dashboard's AI Training screen and SB3's `Monitor`-derived
   `info["episode"]` records are both built around this trend, not a target
   value.

### Events & Alarms log

Each entry names the specific component at fault (site + device id + device
kind) and the fault kind in plain English — "outlier spike", "stuck reading
(not updating)", "sensor offline (no data)", "reading drifted from model
estimate" — with measured vs. model-expected value where available.
Threshold-crossing conditions (unmet demand, H₂ SOC, price spikes) use a
dead-band between raise and clear, so one real episode produces one raise
event and one matching "cleared/recovered" event, not a flood of
step-by-step re-triggers — a healthy run should show raise/clear pairs, not
isolated raises that never clear. See `07_dashboard.md` §9 and
`04_virtual_sensors.md` §12 for the full mapping from internal sensor status
to displayed message.

---

## 6. Test suite

```bash
pytest tests/ -v
```

168 tests, all passing — covering the digital twin, link models, component
models, network controller, network dashboard, network RL, network
topology, network twin, RL environment, sensors, and weather/cost realism.
