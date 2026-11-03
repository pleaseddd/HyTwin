# Network Layer — Multi-Site H2 Network — HyTwin

## 1. Overview

The network layer is what promotes HyTwin from a **single-plant digital
twin** to a **digital twin of a geographically distributed H₂ network**: a
graph of sites, each with its own local weather, equipment, and market
price, connected by explicit electric lines and H₂ pipelines, with a
network-wide dispatch balance and network-aware control (both rule-based and
RL). This is now the primary way to use HyTwin — the flagship scenario is
the 7-node Italian network in `config/italy_network_large.yaml`.

### Guiding design principle: reuse, not rewrite

`GridTwin` was already, in effect, the twin of a single site. The network
layer does not replace it: it promotes it to a **site twin** and adds a
network orchestrator on top. No existing physical model was modified to
support this.

```
┌───────────────────────────────────────────────────────────┐
│  NetworkTwin           (multi-site orchestrator)            │
│   • topology: sites + links (electric lines / H2 pipelines) │
│   • inter-site dispatch + explicit power/H2 flow             │
│   • NetworkState: per-node + system-level                    │
├───────────────────────────────────────────────────────────┤
│  GridTwin  (= "site twin", one per node)                     │
│   • local weather, local components, local balance           │
├───────────────────────────────────────────────────────────┤
│  TwinNode · Physical models · Sensors   (unchanged)           │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Topology — `SiteSpec`, `LinkSpec`, `NetworkTopology`

Defined in `hytwin/network/topology.py`.

```python
@dataclass
class SiteSpec:
    id: str
    location: Location            # name, lat, lon, alt_m
    grid_config: dict             # identical schema to the single-site `grid:` block
    weather_params: dict          # WeatherModel kwargs for this site's climate
    energy_cost: dict             # local market-zone price model config
    sensor_config: list           # optional per-site virtual sensors

@dataclass
class LinkSpec:
    id: str
    link_type: LinkType           # ELECTRIC_LINE | H2_PIPELINE
    from_site: str
    to_site: str
    params: dict
    length_km: Optional[float]    # auto-computed from site coordinates if omitted

class NetworkTopology:
    sites: Dict[str, SiteSpec]
    links: Dict[str, LinkSpec]

    @classmethod
    def from_config(cls, cfg: dict) -> "NetworkTopology": ...
```

`SiteSpec.from_config` merges the site's `location` (lat/lon/alt) into its
`weather_params` automatically, so `WeatherModel` always sees the right
geodata even if the `weather:` block omits it. When a link's `params` omits
`length_km`, `NetworkTopology` fills it in via the **great-circle distance**
between the two sites' coordinates.

`Scenario.from_yaml` is backward-compatible: a file with a `network:` block
builds the multi-site layer; an existing file with only a top-level `grid:`
block is wrapped as a single-site, zero-link `NetworkTopology`.

---

## 3. Configuration schema (YAML)

```yaml
network:
  sites:
    - id: trapani
      location: { name: "Trapani", lat: 37.98, lon: 12.51, alt_m: 20 }
      weather: { weibull_k: 2.2, weibull_c: 7.2, autocorr_wind: 0.91,
                 cloud_cover_mean: 0.20, temp_mean_c: 19.5, temp_amplitude_c: 11 }
      energy_cost: { f1_price: 0.23, f2_price: 0.15, f3_price: 0.08,
                     price_volatility: 0.10, seasonal_amplitude: 0.20,
                     spike_prob_per_day: 0.05, spike_multiplier: 3.0,
                     spike_duration_steps: 4, sell_ratio: 0.30 }
      grid:                          # identical schema to the single-site `grid:` block
        wind_turbines:   [ { id: trapani_wt1, params: { rated_power_kw: 500, ... } } ]
        pv_arrays:       [ ... ]
        electrolyzers:   [ ... ]
        fuel_cells:      [ ... ]
        hydrogen_tanks:  [ ... ]
        loads:           [ ... ]
        grid_connections:[ { id: trapani_grid, params: { max_import_kw: 400, ... } } ]

    - id: napoli
      location: { name: "Napoli", lat: 40.85, lon: 14.27, alt_m: 17 }
      ...

  links:
    - id: trapani_napoli_h2
      type: h2_pipeline
      from: trapani
      to: napoli
      params: { diameter_m: 0.4, max_flow_kg_h: 300,
                compressor_spec_kwh_per_kg: 2.2, line_pack_capacity_kg: 1000 }

    - id: trapani_napoli_elec
      type: electric_line
      from: trapani
      to: napoli
      params: { max_power_mw: 40, loss_per_1000km: 0.06, bidirectional: true }
```

Full field reference and more real examples: `08_configuration_reference.md`.

---

## 4. Explicit dispatch — the core of the layer

The single-site `GridTwin` closes its balance with an **implicit residual**
(`grid_exchange = load + el − supply`). With multiple sites and real
transmission constraints this doesn't scale: energy must flow **explicitly**
across links.

### `NetworkTwin.step()` sequence

```
1. WeatherField.step(ts)                        → weather per site
2. for each site:  GridTwin.step(dt, weather[site], actions[site])
      → local production/consumption, local H2 SOC, local surplus/deficit
3. dispatch_electric_lines(...) / dispatch_h2_pipelines(...)  (hytwin/network/dispatch.py)
      • each site's net_balance = generation − demand, from the models' actual
        values (GridState.physical); the sensor-fused view, one step late, is
        only what the controllers observe (NodeState renewable/FC/EL/load)
      • surplus/deficit routed across links, respecting capacity,
        resistive/compression losses, and availability
      • unresolved residual → that site's own GridConnectionModel
        (import/export), at its own local energy cost and carbon factor
4. aggregate NetworkState (per-node NodeState + per-link LinkFlow + system KPIs)
5. publish on EventBus, record on TimeSeriesRecorder
```

### Dispatch level

The current implementation is **L1: greedy merit-order** dispatch
(`hytwin/network/dispatch.py`, functions `dispatch_h2_pipelines` and
`dispatch_electric_lines`): deficits are covered first from neighbouring
sites with surplus via the cheapest available link, then from each site's
own national-grid connection. This is deterministic, fast, and fully
testable — a fuller LP/QP-based optimal dispatch (minimising cost + losses
under capacity/nodal-balance constraints) is future work, and would itself
become a reference "optimal controller" to benchmark the RL agent against.

---

## 5. Network state — `NodeState`, `LinkFlow`, `NetworkState`

Defined in `hytwin/network/network_state.py`:

```python
@dataclass
class NodeState:                      # per-site state after dispatch
    site_id: str
    timestamp: datetime
    renewable_kw: float
    fuel_cell_kw: float
    load_kw: float
    electrolyzer_kw: float
    compressor_load_kw: float         # H2-pipeline compression at this site
    generation_kw: float              # renewable + fuel cell
    demand_kw: float                  # load + electrolyzer + compressor
    link_import_kw: float             # power delivered in from neighbours
    link_export_kw: float             # power injected out to neighbours
    grid_import_kw: float             # national-grid slack
    grid_export_kw: float
    # ... plus H2 SOC/storage and cost/CO2 fields for the site

@dataclass
class LinkFlow:                       # per-link state this step
    link_id: str
    link_type: str
    from_site: str
    to_site: str
    flow: float                       # source-side (kW electric / kg-h H2)
    delivered: float                  # destination-side after loss/delay
    loss: float
    utilization: float
    available: bool

@dataclass
class NetworkState:                   # aggregated network snapshot
    timestamp: datetime
    nodes: Dict[str, NodeState]
    links: Dict[str, LinkFlow]
    total_load_kw: float
    total_renewable_kw: float
    total_generation_kw: float
    total_grid_import_kw: float
    total_grid_export_kw: float
    total_curtailed_kw: float
    unmet_demand_kw: float
    total_cost_eur_step: float
    cumulative_cost_eur: float
    total_co2_kg_step: float
    inter_node_power_kw: float        # sum of electric-link deliveries
    inter_node_h2_kg_h: float         # sum of pipeline deliveries
    # ... plus system-level self-sufficiency, renewable fraction, reliability
```

`GridState` (per-site, see `03_digital_twin.md`) remains the underlying
per-node building block — the dashboard's per-node drill-down and the
recorder continue to work per site without modification; `NetworkState`
wraps and aggregates it.

---

## 6. Control — traditional and AI

### 6.1 `NetworkClassicalController`

A network-aware overlay on top of the existing single-site
`ClassicalController` — one controller instance per site (built via
`NetworkClassicalController.from_network(network_twin)`), sharing each
site's own price model, plus two network-level dispatch policies:

- **Electric-transport-first**: renewable surplus that could serve a
  neighbouring site's deficit over an electric line is preferentially
  reserved for that transport rather than pushed into local H₂ production,
  since electric transmission is far more efficient than an
  electrolysis → pipeline → fuel-cell round trip.
- **Green-H₂-first**: fuel cells draw down stored H₂ ahead of importing from
  the national grid, so the network favours consuming its own green
  hydrogen reserve over paying for grid electricity when H₂ SOC allows it.

On the 7-node pilot network over a 7-day comparison against a naive
(uncontrolled) baseline, the rule-based network controller achieved
**−12.4% cost, −10.4% CO₂**, while building H₂ reserve rather than draining
it.

### 6.2 `NetworkRLController` / `NetworkRLEnv`

The flat, fixed-dimension observation/action vector of the single-site
`H2GridEnv` does not scale to N sites. HyTwin uses a **per-node factored**
design (a single agent, structured observation/action space):

- **Observation**: concatenation of fixed-size per-site blocks (local
  generation, load, H₂ SOC, price, sensor-fused state, …) plus per-link
  features (utilization, availability).
- **Action**: per-site blocks (electrolyzer/fuel-cell setpoints, demand
  response, local grid posture) plus link-level exchange decisions.

This keeps the interface regular (`observe()` / `decode_action()` as a list
of per-site blocks) while remaining a single PPO policy trainable directly
with Stable-Baselines3 — no custom multi-agent (MARL) or graph-network (GNN)
infrastructure is required for this release, though the factored design was
chosen specifically so it can migrate to MARL/GNN later without changing the
environment's external contract.

On the 7-node pilot network, a PPO agent trained for 40k steps outperformed
the rule-based network controller on cost (232 € vs 6643 € over the
comparison horizon), cut CO₂ by ~52%, reached ~0.91 self-sufficiency, **and**
still built H₂ reserve instead of draining it — see §7 on the reward design
that makes this possible.

`NetworkRLController.factory(model_path)` wraps a trained SB3 model into the
same controller interface used by `compare_controllers` and the dashboard's
`rl` controller mode.

---

## 7. RL reward — `NetworkRewardConfig`

Defined in `hytwin/rl/network_environment.py`. A multi-term weighted reward,
computed once per step from the resulting `NetworkState`:

```python
@dataclass
class NetworkRewardConfig:
    w_cost: float = 1.0                 # €/step penalty (scaled by ref_cost_eur_step)
    w_co2: float = 0.3                  # kg/step penalty (scaled by ref_co2_kg_step)
    w_unmet_linear: float = 9.0         # strong linear penalty on unmet demand
    w_unmet_quadratic: float = 25.0     # quadratic term — deters even small shortfalls
    w_unmet_event: float = 0.5          # flat penalty whenever unmet > unmet_tolerance
    unmet_tolerance: float = 0.002
    w_self_sufficiency: float = 0.4
    w_renewable: float = 0.3
    w_soc_health: float = 0.3           # reward SOC staying near soc_target
    w_curtailment: float = 0.1          # penalise wasted renewable
    w_storage_value: float = 1.0        # credit/debit H2 tank-mass changes
    h2_value_eur_per_kg: float = 4.0    # replacement value of stored H2
    soc_target: float = 0.5
    ref_cost_eur_step: float = 5.0
    ref_co2_kg_step: float = 10.0
    ref_curtail_kw: float = 500.0
```

### Why the H₂ storage-value term exists

Without `w_storage_value`, an agent can make its cost term look artificially
low by draining the H₂ tanks to zero — stored H₂ acts as "free" fuel from
the reward's point of view. `w_storage_value` credits the reward when tank
mass increases and debits it when tank mass decreases, valued at
`h2_value_eur_per_kg` (≈4 €/kg — the same units as the grid-cost term). This
internalises the reserve's replacement value directly into what the agent
optimises, mirroring the `storage_adjusted_cost` KPI used for reporting, and
is what prevents the reward-hacking behaviour described in §6.2.

### Unmet-demand penalty: linear + quadratic + event

A pure quadratic penalty barely discourages *tiny* shortfalls (its
derivative near zero is near zero), so early agents learned to risk small,
frequent unmet-demand events. The reward therefore combines a strong
**linear** term, a **quadratic** term, and a flat **event** penalty
triggered whenever unmet demand exceeds `unmet_tolerance` — together these
make reliability effectively non-negotiable for a well-trained policy
(self-sufficiency and reliability KPIs approach 1.0).

### There is no fixed "optimal" reward value

The reward is an **unbounded weighted sum** whose achievable range depends
on that episode's weather and price draw (a windy, sunny week and a calm,
overcast week are not comparable). There is therefore no fixed benchmark
reward to target. The correct way to evaluate a trained policy is:

1. **Relative performance** vs. the `none` (naive) and `classical`
   (rule-based) controllers, computed under the identical reward formula on
   the **same seed** (same weather/price draw) — this is exactly what
   `compare_controllers` does.
2. The **learning curve's smoothed trend** during training (not its raw
   absolute value) — see `06_reinforcement_learning.md` and the dashboard's
   AI Training screen.

---

## 8. Reproducible comparison — `hytwin.network.compare`

```python
from hytwin.network.compare import compare_controllers

out = compare_controllers(topology, steps=1008, seed=42, strategies={
    "none": "none", "classical": "classical",
    "rl": NetworkRLController.factory("output/rl_models/net_ppo_large"),
})
for name, kpis in out.items():
    print(name, kpis)
```

`compare_controllers` runs each strategy under **identical** conditions
(same seed → same weather/price draw across all sites) and returns a
`{strategy_name: kpis_dict}` mapping, so results are directly comparable.
This is the batch-comparison engine behind both the CLI comparison snippet
and the dashboard's Scenario Comparison screen (`07_dashboard.md`).

---

## 9. Dashboard integration

The network layer surfaces in full in the **Network Control Room**
dashboard (`dashboard/network_app.py`, `07_dashboard.md`): the animated
Italy map plots the real topology's sites/links from live lat/lon data, the
Control & AI screen switches the live controller between `none` /
`classical` / `rl`, Scenario Comparison runs `compare_controllers`-style
batch and live lockstep comparisons from the UI, and the Configuration
screen edits the `network:` YAML directly — adding a node is a matter of
duplicating a `- id: ...` site block and giving it a new id, with no code
changes required.
