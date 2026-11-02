"""
Perfect-foresight LP optimum (benchmark)
========================================
Linear program for one network episode with weather, load, prices and grid
outages known in advance (read from a shadow copy of the exogenous models,
which do not depend on the actions).  It decides every lever *and* the
inter-site flows, so its cost is a lower bound for any controller on the same
episode — the reference the classical controller and the RL agent are
measured against.

Per site and step: electrolyzer / fuel-cell power and H2, demand response,
grid import / export, curtailment, unmet demand, usable tank mass; per link
and step: pipeline injection (delivered ``delay`` steps later) and electric
flow each way (with its loss).  Objective = grid cost + DR cost + VoLL·unmet
(+ optional CO2 price) − value of the H2 left in tanks and pipes at the end.

Linearisation (measured on the simulator's own models, see Task 3 notes):
the electrolyzer H2(P) curve is replaced by its upper concave envelope over
the points where the model's Newton solve converged, the fuel-cell H2(P)
curve by its lower convex envelope (both at the units' operating temperature).  Minimum-load and on/off are relaxed, ramp limits
never bind at 10-minute steps, so the LP is slightly optimistic.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from typing import Dict, List

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..models import ElectrolyzerModel, FuelCellModel, EnergyLoadModel
from .topology import LinkType


def _hull(p: np.ndarray, h: np.ndarray, upper: bool):
    """Slopes/intercepts of the upper concave (or lower convex) envelope of points (p, h)."""
    pts: List[tuple] = []
    for x, y in zip(p, h):
        while len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if (cross >= 0) if upper else (cross <= 0):
                pts.pop()
            else:
                break
        pts.append((x, y))
    a = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in zip(pts, pts[1:])]
    return [(s, y1 - s * x1) for s, (x1, y1) in zip(a, pts)]


def _unit_curve(model, dt: float, n: int = 120):
    """(P [kW], H2 [kg/step]) for one unit, only points where the solve is consistent."""
    m = copy.deepcopy(model)
    P = np.linspace(0.0, m._rated_kw, n)
    if isinstance(m, ElectrolyzerModel):
        h = np.array([m._power_to_h2(x, dt) for x in P])
        # drop points where Newton stopped short (output falls with more power or jumps above the trend)
        ok = np.ones(n, bool)
        kept = 0
        for i in range(1, n):
            if h[i] < h[kept] or (i + 1 < n and h[i] > h[i + 1] and h[i + 1] >= h[kept]):
                ok[i] = False
            else:
                kept = i
        return P[ok], h[ok]
    return P, np.array([m._power_to_h2_demand(x, dt)[0] for x in P])


class _Vars:
    def __init__(self):
        self.n, self.lb, self.ub, self.c = 0, [], [], []

    def add(self, shape, lb=0.0, ub=np.inf, cost=0.0):
        k = int(np.prod(shape))
        idx = np.arange(self.n, self.n + k).reshape(shape)
        self.n += k
        for lst, v in ((self.lb, lb), (self.ub, ub), (self.c, cost)):
            lst.append(np.broadcast_to(np.asarray(v, float), shape).reshape(-1))
        return idx


class _Rows:
    def __init__(self):
        self.r, self.c, self.v, self.b, self.n = [], [], [], [], 0

    def add(self, terms, rhs):
        """terms: list of (var index array, coefficient array); one row per element of rhs."""
        rhs = np.asarray(rhs, float).reshape(-1)
        rows = np.arange(self.n, self.n + len(rhs))
        for idx, coef in terms:
            idx = np.asarray(idx).reshape(-1)
            coef = np.broadcast_to(np.asarray(coef, float), idx.shape)
            # a single row may sum several variables (e.g. gas in a pipe over a window)
            self.r.append(rows if len(rows) == len(idx) else np.repeat(rows, len(idx)))
            self.c.append(idx); self.v.append(coef)
        self.b.append(rhs)
        self.n += len(rhs)

    def matrix(self, nvar):
        if not self.n:
            return None, None
        return (sparse.csr_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                                  shape=(self.n, nvar)), np.concatenate(self.b))


def solve_episode(env, steps: int, h2_value: float = 1.5, dr_cost: float = 0.20,
                  voll: float = 10.0, co2_price: float = 0.0, lines_mode: str = "free",
                  mip_gap: float = 1e-3) -> Dict[str, float]:
    """
    Optimum for the episode an env has just been reset to (``env.reset(seed=...)``).

    ``lines_mode="free"``: any flow on the electric lines (multi-hop routing).
    ``lines_mode="neighbour"``: like the simulator's dispatch_electric_lines, a line only
    carries power from a site in surplus to a neighbour in deficit, at most the sender's
    surplus and the receiver's deficit (no transit); needs one binary per site and step (MILP).

    Returns totals comparable with the evaluation rows: ``cost`` (grid + DR, €),
    ``co2`` (kg), ``unmet`` (kWh), ``curt`` (kWh), ``dh2`` (kg, tanks + pipes),
    plus ``objective`` and solver ``status``.
    """
    from ..rl.network_environment import _ExogShadow

    twin, dt = env._twin, env._dt
    dt_h = dt / 3600.0
    sites = list(env.site_ids)
    S, T = len(sites), steps
    sh = _ExogShadow(twin)
    ex = np.array([[sh.step(dt, env._ts + timedelta(seconds=dt * t))[s] for s in sites] for t in range(T)])
    ren, load, buy, sell, _, avail = (ex[:, :, k] for k in range(6))   # (T, S)

    units = {s: [n.model for n in twin._sites[s]._nodes.values()] for s in sites}
    els = {s: [m for m in units[s] if isinstance(m, ElectrolyzerModel)] for s in sites}
    fcs = {s: [m for m in units[s] if isinstance(m, FuelCellModel)] for s in sites}
    tanks = {s: twin._tanks_by_site[s] for s in sites}
    dr_frac = np.array([max((m._dr_factor for m in units[s] if isinstance(m, EnergyLoadModel)), default=0.0)
                        for s in sites])
    gi = [twin._grid_info[s] for s in sites]
    cf = np.array([g["carbon_factor"] for g in gi])

    V, A_eq, A_ub = _Vars(), _Rows(), _Rows()
    el_cap = np.array([sum(m._rated_kw for m in els[s]) for s in sites])
    fc_cap = np.array([sum(m._rated_kw for m in fcs[s]) for s in sites])
    cap = np.array([sum(t.max_capacity_kg for t in tanks[s]) for s in sites])
    m0 = np.array([sum(t.soc * t.max_capacity_kg for t in tanks[s]) for s in sites])
    pel = V.add((T, S), ub=el_cap)
    hel = V.add((T, S), ub=np.where(el_cap > 0, np.inf, 0.0))     # no electrolyzer -> no H2 made
    pfc = V.add((T, S), ub=fc_cap)
    hfc = V.add((T, S), ub=np.where(fc_cap > 0, np.inf, 0.0))
    dr = V.add((T, S), ub=dr_frac * load, cost=dr_cost * dt_h)
    imp = V.add((T, S), ub=np.array([g["max_import_kw"] for g in gi]) * avail,
                cost=(buy + co2_price * cf) * dt_h)
    exp_ = V.add((T, S), ub=np.array([g["max_export_kw"] for g in gi]), cost=-sell * dt_h)
    curt = V.add((T, S))
    unmet = V.add((T, S), cost=voll * dt_h)
    mass_cost = np.zeros((T, S)); mass_cost[-1] = -h2_value     # H2 left at the end is worth h2_value
    mass = V.add((T, S), ub=cap, cost=mass_cost)

    pipes = [l for l in env._topo.links.values() if l.link_type == LinkType.H2_PIPELINE]
    lines = [l for l in env._topo.links.values() if l.link_type == LinkType.ELECTRIC_LINE]
    q, fa, fb = {}, {}, {}
    for l in pipes:
        pm = twin._pipe_models[l.id]
        d = max(0, int(round(pm._delay_s / dt)))
        qc = np.zeros(T); qc[max(0, T - d):] = -h2_value          # still in the pipe at the end
        q[l.id] = (V.add((T,), ub=pm._max_flow_kg_h * dt_h, cost=qc), d, pm)
    for l in lines:
        lm = twin._line_models[l.id]
        fa[l.id] = V.add((T,), ub=lm._max_kw)
        fb[l.id] = V.add((T,), ub=lm._max_kw if lm._bidirectional else 0.0)

    # electrolyzer: H2 <= n·f(P/n) (upper envelope); fuel cell: H2 >= n·g(P/n)
    for j, s in enumerate(sites):
        if els[s]:
            n = len(els[s])
            for a, b in _hull(*_unit_curve(els[s][0], dt), upper=True):
                A_ub.add([(hel[:, j], 1.0), (pel[:, j], -a)], np.full(T, n * b))
        if fcs[s]:
            n = len(fcs[s])
            for a, b in _hull(*_unit_curve(fcs[s][0], dt), upper=False):
                A_ub.add([(pfc[:, j], a), (hfc[:, j], -1.0)], np.full(T, -n * b))

    # electric balance per site and step
    for j, s in enumerate(sites):
        terms = [(pfc[:, j], 1), (imp[:, j], 1), (unmet[:, j], 1), (dr[:, j], 1),
                 (pel[:, j], -1), (exp_[:, j], -1), (curt[:, j], -1)]
        for l in lines:
            loss = twin._line_models[l.id]._loss_fraction
            if l.from_site == s:
                terms += [(fa[l.id], -1), (fb[l.id], 1 - loss)]
            if l.to_site == s:
                terms += [(fa[l.id], 1 - loss), (fb[l.id], -1)]
        for l in pipes:
            if l.from_site == s:
                terms.append((q[l.id][0], -q[l.id][2]._compressor_spec / dt_h))
        A_eq.add(terms, load[:, j] - ren[:, j])

    integrality = None
    if lines_mode == "neighbour":
        comp_cap = np.array([sum(q[l.id][2]._compressor_spec * q[l.id][2]._max_flow_kg_h for l in pipes
                                 if l.from_site == s) for s in sites])
        big = ren + load + el_cap + fc_cap + comp_cap + 1.0                 # (T, S) bound on |net|
        sp, dm = V.add((T, S), ub=big), V.add((T, S), ub=big)
        z = V.add((T, S), ub=1.0)                                           # 1 = site in surplus
        for j, s in enumerate(sites):
            # sp - dm = net before the lines = ren - load + dr + fc - el - compressor
            terms = [(sp[:, j], 1), (dm[:, j], -1), (dr[:, j], -1), (pfc[:, j], -1), (pel[:, j], 1)]
            terms += [(q[l.id][0], q[l.id][2]._compressor_spec / dt_h) for l in pipes if l.from_site == s]
            A_eq.add(terms, ren[:, j] - load[:, j])
            out, inn = [], []
            for l in lines:
                loss = twin._line_models[l.id]._loss_fraction
                if l.from_site == s:
                    out.append((fa[l.id], 1.0)); inn.append((fb[l.id], 1 - loss))
                if l.to_site == s:
                    out.append((fb[l.id], 1.0)); inn.append((fa[l.id], 1 - loss))
            if out:
                A_ub.add(out + [(sp[:, j], -1.0)], np.zeros(T))           # sends only its surplus
                A_ub.add(inn + [(dm[:, j], -1.0)], np.zeros(T))           # receives only its deficit
            A_ub.add([(sp[:, j], 1.0), (z[:, j], -big[:, j])], np.zeros(T))
            A_ub.add([(dm[:, j], 1.0), (z[:, j], big[:, j])], big[:, j])
        integrality = np.zeros(V.n); integrality[z.reshape(-1)] = 1

    # tanks: mass[t] = mass[t-1] + hel - hfc - out + in(delayed); charge / discharge rate limits
    for j, s in enumerate(sites):
        _tank_rows(A_eq, A_ub, mass[:, j], hel[:, j], hfc[:, j], m0[j], T,
                   [(q[l.id][0], q[l.id][1]) for l in pipes if l.to_site == s],
                   [q[l.id][0] for l in pipes if l.from_site == s],
                   len(tanks[s]) * min(t._max_in for t in tanks[s]) * dt if tanks[s] else 0.0,
                   len(tanks[s]) * min(t._max_out for t in tanks[s]) * dt if tanks[s] else 0.0)

    # pipe line-pack: gas in transit <= capacity
    for l in pipes:
        qi, d, pm = q[l.id]
        if d > 0:
            for t in range(T):
                win = qi[max(0, t - d + 1):t + 1]
                A_ub.add([(win, 1.0)], [pm._linepack_cap])

    Aeq, beq = A_eq.matrix(V.n)
    Aub, bub = A_ub.matrix(V.n)
    c = np.concatenate(V.c)
    bounds = np.column_stack([np.concatenate(V.lb), np.concatenate(V.ub)])
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs",
                  integrality=integrality, options={"mip_rel_gap": mip_gap} if integrality is not None else None)
    if res.status != 0:
        return {"status": res.status, "message": res.message}
    x = res.x
    val = lambda idx: x[np.asarray(idx)]
    in_pipes = sum(val(q[l.id][0])[max(0, T - q[l.id][1]):].sum() for l in pipes)
    # the agent's levers as env actions (fraction of rated power; DR as the env's 0-1 slot = shed share / 0.2)
    actions = np.zeros((T, len(env._act_slots)), np.float32)
    for k, (s, kind) in enumerate(env._act_slots):
        j = sites.index(s)
        if kind == "el":
            actions[:, k] = val(pel)[:, j] / el_cap[j]
        elif kind == "fc":
            actions[:, k] = val(pfc)[:, j] / fc_cap[j]
        else:
            actions[:, k] = np.divide(val(dr)[:, j], load[:, j] * 0.2, out=np.zeros(T), where=load[:, j] > 0)
    return {
        "status": 0,
        "objective": float(res.fun),
        "cost": float(((val(imp) * buy - val(exp_) * sell) * dt_h).sum() + (val(dr) * dt_h).sum() * dr_cost),
        "co2": float((val(imp) * cf * dt_h).sum()),
        "unmet": float((val(unmet) * dt_h).sum()),
        "curt": float((val(curt) * dt_h).sum()),
        "dh2": float(val(mass)[-1].sum() - m0.sum() + in_pipes),
        "el_kwh": float((val(pel) * dt_h).sum()),
        "fc_kwh": float((val(pfc) * dt_h).sum()),
        "dr_kwh": float((val(dr) * dt_h).sum()),
        "n_vars": V.n,
        "by_site": {k: val(v).sum(0) * dt_h for k, v in dict(el=pel, fc=pfc, dr=dr, imp=imp, exp=exp_,
                                                             curt=curt, unmet=unmet).items()},   # kWh per site
        "actions": np.clip(actions, 0.0, 1.0),
    }


def replay(env, seed: int, actions: np.ndarray, dr_cost: float = 0.20) -> Dict[str, float]:
    """Run the LP's electrolyzer / fuel-cell / DR set-points in the real simulator (its own flows)."""
    env.reset(seed=seed)
    h0 = env._total_h2()
    tot = np.zeros(4)
    for a in actions:
        ns = env.step(a)[4]["network_state"]
        tot += (ns.total_cost_eur_step + env._dr_shed_kw() * env._dt / 3600 * dr_cost, ns.total_co2_kg_step,
                ns.unmet_demand_kw * env._dt / 3600, ns.total_curtailed_kw * env._dt / 3600)
    return {"cost": float(tot[0]), "co2": float(tot[1]), "unmet": float(tot[2]), "curt": float(tot[3]),
            "dh2": float(env._total_h2() - h0)}


def _tank_rows(A_eq, A_ub, mass, hel, hfc, m0, T, inflows, outflows, max_in, max_out):
    """mass[t] - mass[t-1] - hel[t] + hfc[t] + sum out[t] - sum in[t-d] = 0 (mass[-1] = m0)."""
    for t in range(T):
        terms = [(mass[t:t + 1], 1.0), (hel[t:t + 1], -1.0), (hfc[t:t + 1], 1.0)]
        if t:
            terms.append((mass[t - 1:t], -1.0))
        terms += [(o[t:t + 1], 1.0) for o in outflows]
        ins = [qi[t - d:t - d + 1] for qi, d in inflows if t - d >= 0]
        terms += [(i, -1.0) for i in ins]
        A_eq.add(terms, [m0 if t == 0 else 0.0])
        A_ub.add([(hel[t:t + 1], 1.0)] + [(i, 1.0) for i in ins], [max_in])
        A_ub.add([(hfc[t:t + 1], 1.0)] + [(o[t:t + 1], 1.0) for o in outflows], [max_out])
