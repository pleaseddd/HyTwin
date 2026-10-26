"""
Multi-Site Network RL Environment
=================================
A Gymnasium environment that wraps the :class:`NetworkTwin` so an RL agent can
learn a **network-wide** dispatch policy for the Italian H2 network.

Design (per-node factored, see docs/08_network_layer_plan.md §8)
---------------------------------------------------------------
* **Observation** — a fixed-size block per site concatenated in a stable order,
  plus two global time features.  Per-site block (8 features):
    [h2_soc, load/ref, renewable/ref, price/0.5, grid_available,
     fuel_cell/cap, electrolyzer/cap, net_balance/ref]
  → obs_dim = 8·N + 2.
* **Action** — a per-site block ``[el_setpoint, fc_setpoint, demand_response]``
  in [-1,1] (rescaled to [0,1] before decoding) → act_dim = 3·N.  The *inter-node flows and national-grid slack are
  resolved by the F3 network dispatch*, exactly as for the classical controller,
  so the agent learns the same lever set (local assets) and must discover the
  network-aware coordination the rule controller had to be hand-tuned for.

The reward is a network-level weighted objective (cost, CO₂, unmet demand,
self-sufficiency, renewable use, H₂-SOC health, demand-response cost).  Each
episode's weather / price / grid randomness lives on an **isolated**
``numpy.random.Generator`` subtree, seeded per-episode from Gymnasium's own
``self.np_random`` (see :meth:`NetworkH2GridEnv.reset`) — never the shared
global RNG — so this env stays fully reproducible given ``reset(seed=…)`` and
safe to run concurrently with any other simulation (e.g. a live dashboard) in
the same process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..network.network_twin import NetworkTwin
from ..network.network_state import NetworkState
from ..network.dispatch import site_soc
from ..models import (
    ElectrolyzerModel, FuelCellModel, EnergyLoadModel,
)
from .forecast_utils import _solar_elevation_deg, _ghi_clear_sky

logger = logging.getLogger(__name__)

N_SITE_OBS = 8
N_SITE_ACT = 3
N_GLOBAL_OBS = 2
FORECAST_HORIZONS_H = (1.0, 3.0)          # look-ahead horizons [hours]
N_FORECAST = 3 * len(FORECAST_HORIZONS_H)  # solar, price-band, load per horizon


def obs_dim_for_n_sites(n_sites: int) -> int:
    """Observation-space size for a network with *n_sites* nodes."""
    return N_SITE_OBS * n_sites + N_GLOBAL_OBS + N_FORECAST


def act_dim_for_n_sites(n_sites: int) -> int:
    """Action-space size for a network with *n_sites* nodes."""
    return N_SITE_ACT * n_sites


def infer_n_sites_from_obs_dim(obs_dim: int) -> Optional[int]:
    """
    Recover the site count a saved model was trained for from its observation
    dimension alone — used to validate that an RL model picked from the
    library is compatible with the currently active network topology before
    it is wired in as the live controller (a shape mismatch would otherwise
    surface as an opaque crash inside ``model.predict()`` mid-simulation).
    """
    rem = obs_dim - N_GLOBAL_OBS - N_FORECAST
    if rem <= 0 or rem % N_SITE_OBS != 0:
        return None
    return rem // N_SITE_OBS


def _pun_band_factor(ts: datetime) -> float:
    """Deterministic Italian PUN tariff band at *ts*: F1→1.0, F2→0.6, F3→0.2."""
    h, wd = ts.hour, ts.weekday()
    if wd == 6:
        return 0.2
    if wd < 5:
        if 8 <= h < 19:
            return 1.0
        if 7 <= h < 8 or 19 <= h < 23:
            return 0.6
        return 0.2
    return 0.6 if 7 <= h < 23 else 0.2


def _load_diurnal_factor(ts: datetime) -> float:
    """Deterministic diurnal demand proxy in [~0.6, 1.0] (daytime hump)."""
    h = ts.hour + ts.minute / 60.0
    hump = max(0.0, np.sin(np.pi * (h - 6.0) / 14.0)) if 6.0 <= h <= 20.0 else 0.0
    return float(0.62 + 0.38 * hump)


@dataclass
class NetworkRewardConfig:
    """Weights for the network-level reward (per step)."""
    w_cost: float = 1.0            # €/step penalty (scaled by ref_cost)
    w_co2: float = 0.3             # kg/step penalty (scaled by ref_co2)
    # Unmet-demand penalty: a small quadratic barely deters tiny shortfalls, so
    # the agent used to risk them.  A strong LINEAR term + a flat EVENT penalty
    # on any unmet make reliability essentially non-negotiable (→ 1.0).
    w_unmet_linear: float = 9.0
    w_unmet_quadratic: float = 25.0
    w_unmet_event: float = 0.5     # flat penalty whenever unmet > tolerance
    unmet_tolerance: float = 0.002
    w_self_sufficiency: float = 0.4
    w_renewable: float = 0.3
    w_soc_health: float = 0.3      # reward SOC near target band
    w_curtailment: float = 0.1     # penalise wasted renewable
    # Value of stored H₂: credit charging / debit draining the tanks at
    # ``h2_value_eur_per_kg`` (same units as grid cost).  Without this the agent
    # treats stored H₂ as free fuel and drains it to zero to fake a low cost —
    # this term internalises the reserve's replacement value so the policy is
    # sustainable, mirroring the ``storage_adjusted_cost`` KPI.
    w_storage_value: float = 1.0
    h2_value_eur_per_kg: float = 4.0
    # Demand response is not free: shed load is paid back to consumers at
    # ``dr_cost_eur_per_kwh``.  Without it the agent sheds load at every step.
    w_demand_response: float = 1.0
    dr_cost_eur_per_kwh: float = 0.20
    soc_target: float = 0.5
    # Typical |cost| and CO2 per step on the pilot network over random days
    # (spento/classical, 40 days: ~12-13 EUR, ~22-26 kg); 5 / 10 were tuned
    # on a single June Saturday.  Re-measure for other topologies.
    ref_cost_eur_step: float = 13.0
    ref_co2_kg_step: float = 25.0
    ref_curtail_kw: float = 500.0


class NetworkH2GridEnv(gym.Env):
    """Gymnasium env for network-wide H2 dispatch control."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology,
        dt_seconds: float = 600.0,
        episode_steps: int = 144,
        start_time: Optional[datetime] = None,
        reward_config: Optional[NetworkRewardConfig] = None,
    ) -> None:
        super().__init__()
        self._topo = topology
        self._dt = float(dt_seconds)
        self._ep_len = int(episode_steps)
        self._start = start_time or datetime(2024, 6, 15, 0, 0)
        self._rc = reward_config or NetworkRewardConfig()

        self._site_ids: List[str] = list(topology.site_ids)
        self._n = len(self._site_ids)

        # Per-site rated references (for normalisation + action scaling).
        self._el_ids: Dict[str, List[Tuple[str, float]]] = {}
        self._fc_ids: Dict[str, List[Tuple[str, float]]] = {}
        self._load_ids: Dict[str, List[str]] = {}
        self._load_ref: Dict[str, float] = {}
        self._renew_ref: Dict[str, float] = {}
        self._el_cap: Dict[str, float] = {}
        self._fc_cap: Dict[str, float] = {}
        for sid in self._site_ids:
            gc = topology.sites[sid].grid_config
            self._el_ids[sid] = [(e["id"], float(e["params"]["rated_power_kw"]))
                                 for e in gc.get("electrolyzers", [])]
            self._fc_ids[sid] = [(f["id"], float(f["params"]["rated_power_kw"]))
                                 for f in gc.get("fuel_cells", [])]
            self._load_ids[sid] = [str(l["id"]) for l in gc.get("loads", [])]
            self._load_ref[sid] = max(
                (float(l["params"].get("base_load_kw", 100.0)) for l in gc.get("loads", [])),
                default=100.0,
            )
            self._renew_ref[sid] = (
                sum(float(w["params"]["rated_power_kw"]) for w in gc.get("wind_turbines", []))
                + sum(float(p["params"]["rated_power_kw"]) for p in gc.get("pv_arrays", []))
            ) or 100.0
            self._el_cap[sid] = sum(r for _, r in self._el_ids[sid])
            self._fc_cap[sid] = sum(r for _, r in self._fc_ids[sid])

        # Network centroid (for the shared physics-based solar forecast).
        locs = [topology.sites[s].location for s in self._site_ids]
        self._lat = float(np.mean([l.lat for l in locs])) if locs else 41.5
        self._lon = float(np.mean([l.lon for l in locs])) if locs else 12.5
        self._alt = float(np.mean([l.alt_m for l in locs])) if locs else 50.0

        obs_dim = N_SITE_OBS * self._n + N_GLOBAL_OBS + N_FORECAST
        act_dim = N_SITE_ACT * self._n
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(obs_dim,), dtype=np.float32)
        # Symmetric [-1, 1] (SB3 advice): with [0, 1] PPO's initial Gaussian
        # (mean 0, std 1) had ~2/3 of its samples clipped to 0 or 1.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

        self._twin: Optional[NetworkTwin] = None
        self._ts = self._start
        self._step_count = 0
        self._last_state: Optional[NetworkState] = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        # Gymnasium's own `self.np_random` is (re)seeded above only when an
        # explicit `seed` is passed; on later resets (e.g. each new episode
        # during training) it just continues its own independent stream. We
        # draw one integer from it here to seed this episode's NetworkTwin —
        # giving each episode a fresh, fully isolated RNG subtree (weather /
        # sensors / outages), reproducible end-to-end from a single
        # `env.reset(seed=...)`, and never touching the shared global
        # `numpy.random` state (safe for concurrent training + live sim).
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._twin = NetworkTwin(self._topo, soc_target=self._rc.soc_target, seed=episode_seed)
        # Random start day (all seasons, weekdays and tariff bands), as in
        # advanced_environment; options={"random_start": False} keeps start_time.
        random_start = True if options is None else bool(options.get("random_start", True))
        day_offset = int(self.np_random.integers(0, 365)) if random_start else 0
        self._ts = self._start + timedelta(days=day_offset)
        self._step_count = 0
        self._last_state = None
        self._prev_h2_total = self._total_h2()
        return self._observe_cold(), {}

    def step(self, action: np.ndarray):
        action = (np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) + 1.0) / 2.0
        actions_by_site = self._decode(action)
        ns = self._twin.step(self._dt, self._ts, actions_by_site)
        self._last_state = ns
        self._ts = self._ts + timedelta(seconds=self._dt)
        self._step_count += 1

        reward, terms = self._reward(ns)
        obs = self._observe(ns)
        terminated = False
        truncated = self._step_count >= self._ep_len
        info = {"reward_terms": terms, "network_state": ns}
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode(self, action: np.ndarray) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for i, sid in enumerate(self._site_ids):
            base = i * N_SITE_ACT
            el_f, fc_f, dr_f = action[base], action[base + 1], action[base + 2]
            site_act: Dict[str, Dict[str, Any]] = {}
            for eid, rated in self._el_ids[sid]:
                site_act[eid] = {"power_setpoint_kw": float(el_f) * rated}
            for fid, rated in self._fc_ids[sid]:
                site_act[fid] = {"power_setpoint_kw": float(fc_f) * rated}
            for lid in self._load_ids[sid]:
                site_act[lid] = {"demand_response": float(dr_f) * 0.2}
            out[sid] = site_act
        return out

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe_cold(self) -> np.ndarray:
        """Initial obs before the first step (from tank initial SOC)."""
        feats: List[float] = []
        for sid in self._site_ids:
            soc = site_soc(self._twin._tanks_by_site[sid])
            feats += [soc, 0.0, 0.0, 0.30 / 0.5, 1.0, 0.0, 0.0, 0.0]
        feats += self._time_feats(self._ts)
        feats += self._forecast_feats(self._ts)
        return np.clip(np.array(feats, dtype=np.float32), -2.0, 2.0)

    def _forecast_feats(self, ts: datetime) -> List[float]:
        """
        Deterministic look-ahead the agent can rely on: clear-sky solar factor,
        PUN tariff band and diurnal load factor at each horizon.  These are the
        *predictable structure* of the problem (day/night, price peaks, demand
        cycle) — enough for the policy to pre-charge H₂ before price spikes or
        renewable lulls without needing a stochastic weather forecast.
        """
        out: List[float] = []
        for hh in FORECAST_HORIZONS_H:
            fts = ts + timedelta(hours=hh)
            elev = _solar_elevation_deg(fts, self._lat, self._lon)
            ghi = _ghi_clear_sky(elev, self._alt)
            out.append(float(np.clip(ghi / 900.0, 0.0, 1.0)))
            out.append(_pun_band_factor(fts))
            out.append(_load_diurnal_factor(fts))
        return out

    def _observe(self, ns: NetworkState) -> np.ndarray:
        feats: List[float] = []
        for sid in self._site_ids:
            n = ns.nodes[sid]
            lref, rref = self._load_ref[sid], self._renew_ref[sid]
            net = (n.renewable_kw - n.load_kw) / (rref + 1e-9)
            feats += [
                n.h2_soc,
                n.load_kw / (lref + 1e-9),
                n.renewable_kw / (rref + 1e-9),
                n.price_eur_kwh / 0.5,
                1.0 if n.grid_available else 0.0,
                n.fuel_cell_kw / (self._fc_cap[sid] + 1e-9),
                n.electrolyzer_kw / (self._el_cap[sid] + 1e-9),
                net,
            ]
        feats += self._time_feats(ns.timestamp)
        feats += self._forecast_feats(ns.timestamp)
        return np.clip(np.array(feats, dtype=np.float32), -2.0, 2.0)

    @staticmethod
    def _time_feats(ts: datetime) -> List[float]:
        ang = 2 * np.pi * ts.hour / 24.0
        return [float(np.sin(ang)), float(np.cos(ang))]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward(self, ns: NetworkState) -> Tuple[float, Dict[str, float]]:
        rc = self._rc
        cost_term = -rc.w_cost * (ns.total_cost_eur_step / rc.ref_cost_eur_step)
        co2_term = -rc.w_co2 * (ns.total_co2_kg_step / rc.ref_co2_kg_step)
        unmet_frac = ns.unmet_demand_kw / (ns.total_load_kw + 1e-9)
        unmet_term = -(rc.w_unmet_linear * unmet_frac
                       + rc.w_unmet_quadratic * unmet_frac ** 2)
        if unmet_frac > rc.unmet_tolerance:
            unmet_term -= rc.w_unmet_event
        ss_term = rc.w_self_sufficiency * ns.network_self_sufficiency
        renew_term = rc.w_renewable * ns.network_renewable_fraction
        soc_dev = abs(ns.avg_h2_soc - rc.soc_target)
        soc_term = rc.w_soc_health * (1.0 - 2.0 * soc_dev)
        curt_term = -rc.w_curtailment * min(1.0, ns.total_curtailed_kw / rc.ref_curtail_kw)

        # Value of the change in stored H₂ this step (charge = credit, drain =
        # debit), in the same normalised € units as the cost term.
        h2_now = self._total_h2()
        delta_h2 = h2_now - self._prev_h2_total
        self._prev_h2_total = h2_now
        storage_term = rc.w_storage_value * (
            delta_h2 * rc.h2_value_eur_per_kg / rc.ref_cost_eur_step
        )

        dr_term = -rc.w_demand_response * (
            self._dr_shed_kw() * self._dt / 3600.0 * rc.dr_cost_eur_per_kwh / rc.ref_cost_eur_step
        )

        total = (cost_term + co2_term + unmet_term + ss_term + renew_term
                 + soc_term + curt_term + storage_term + dr_term)
        terms = {
            "cost": cost_term, "co2": co2_term, "unmet": unmet_term,
            "self_sufficiency": ss_term, "renewable": renew_term,
            "soc": soc_term, "curtailment": curt_term, "storage": storage_term,
            "demand_response": dr_term,
        }
        return total, terms

    def _dr_shed_kw(self) -> float:
        return float(sum(
            self._twin.site(sid).nodes[lid].model.state.values.get("dr_shed_kw", 0.0)
            for sid in self._site_ids for lid in self._load_ids[sid]
        ))

    def _total_h2(self) -> float:
        # Tanks plus gas in transit: an H2 transfer is not a loss.
        return float(sum(
            sum(t.mass_kg for t in tanks)
            for tanks in self._twin._tanks_by_site.values()
        ) + sum(p.linepack_kg for p in self._twin._pipe_models.values()))

    # ------------------------------------------------------------------
    @property
    def site_ids(self) -> List[str]:
        return list(self._site_ids)
