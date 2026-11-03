"""
test_network_dashboard.py
========================
F6 tests for the network dashboard worker (FastAPI-independent): it builds the
topology payload, produces well-formed NetworkState frames, tracks per-node
history, and honours the runtime controller switch.
Run: pytest tests/test_network_dashboard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.network_app import NetworkSimulationWorker, DEFAULT_CONFIG


@pytest.fixture
def worker():
    w = NetworkSimulationWorker(DEFAULT_CONFIG, dt_seconds=600.0, seed=42)
    w._build()
    return w


def _first_site(worker):
    return worker.topology_payload()["sites"][0]["id"]


def test_topology_payload(worker):
    t = worker.topology_payload()
    assert len(t["sites"]) >= 3
    for s in t["sites"]:
        assert "lat" in s and "lon" in s
    assert len(t["links"]) >= 2
    assert all(l["length_km"] > 0 for l in t["links"])


def test_frame_is_well_formed(worker):
    frame = worker.step_once()
    assert "ts" in frame and "step" in frame and "controller" in frame
    sites = set(worker.topology_payload()["sites"][i]["id"] for i in range(len(worker.topology_payload()["sites"])))
    assert set(frame["nodes"].keys()) == sites
    assert len(frame["links"]) == len(worker.topology_payload()["links"])
    n = frame["nodes"][_first_site(worker)]
    for key in ("load_kw", "h2_soc", "grid_import_kw", "price_eur_kwh",
                "renewable_kw", "link_import_kw"):
        assert key in n
    assert 0.0 <= n["h2_soc"] <= 1.0


def test_node_history_accumulates(worker):
    sid = _first_site(worker)
    for _ in range(5):
        worker.step_once()
    hist = worker.node_history(sid)
    assert len(hist) == 5
    assert "soc" in hist[0] and "grid_import_kw" in hist[0]


def test_component_detail(worker):
    sid = _first_site(worker)
    for _ in range(4):
        worker.step_once()
    comps = worker.node_components(sid)
    assert comps
    cid = comps[0]["id"]
    d = worker.component_detail(sid, cid)
    assert d["id"] == cid and d["kind"] == comps[0]["kind"]
    assert len(d["history"]) == 4  # one row per step


def test_controller_switch_changes_actions(worker):
    """Switching to 'none' vs 'classical' must change the dispatch outcome."""
    worker.set_controller("classical")
    for _ in range(30):
        worker.step_once()
    soc_classical = worker._prev_state.avg_h2_soc

    w2 = NetworkSimulationWorker(DEFAULT_CONFIG, dt_seconds=600.0, seed=42)
    w2._build()
    w2.set_controller("none")
    for _ in range(30):
        w2.step_once()
    soc_none = w2._prev_state.avg_h2_soc

    assert soc_classical != soc_none


def test_frame_balance_closes(worker):
    """The frames the dashboard streams reflect a physically closed balance."""
    for _ in range(20):
        worker.step_once()
    ns = worker._prev_state
    for node in ns.nodes.values():
        supply = node.generation_kw + node.link_import_kw + node.grid_import_kw
        sink = (node.demand_kw + node.link_export_kw
                + node.grid_export_kw + node.curtailed_kw - node.unmet_kw)
        assert abs(supply - sink) < 1e-6


# ---------------------------------------------------------------------------
# FastAPI app: REST + WebSocket (regression guard for the WS route)
# ---------------------------------------------------------------------------

_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_app_serves_rest_and_websocket():
    import json
    from starlette.testclient import TestClient
    from dashboard.network_app import create_app

    app, _ = create_app()
    with TestClient(app) as client:
        # The sim is built (topology ready) but NOT auto-running — the operator
        # must explicitly press "Avvia" (see NetworkSimulationWorker.prepare()).
        status = client.get("/status").json()
        assert status["built"] is True
        assert status["running"] is False
        # REST
        topo = client.get("/network/topology").json()
        site_ids = {s["id"] for s in topo["sites"]}
        assert len(site_ids) >= 3
        first = topo["sites"][0]["id"]
        assert client.get("/").status_code == 200
        assert client.post("/control", json={"type": "none"}).json()["controller"] == "none"
        # Explicitly start the sim, then the WebSocket delivers a well-formed
        # frame (guards both /sim/start and the /ws route binding).
        assert client.post("/sim/start").json()["status"] == "started"
        with client.websocket_connect("/ws") as ws:
            frame = json.loads(ws.receive_text())
        assert set(frame["nodes"].keys()) == site_ids
        assert len(frame["links"]) == len(topo["links"])
        assert "total_load_kw" in frame and "reliability_index" in frame
        # Enriched presentation payloads
        for key in ("components", "weather", "actions", "objective", "new_events"):
            assert key in frame
        assert [c["kind"] for c in frame["components"][first]]  # device list present


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_new_api_endpoints():
    import time
    from starlette.testclient import TestClient
    from dashboard.network_app import create_app

    app, _ = create_app()
    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        assert "none" in cfg["controllers"] and "classical" in cfg["controllers"]
        assert "rl_available" in cfg
        client.post("/sim/start")  # sim is paused by default — must start explicitly
        time.sleep(1.0)  # let the sim thread accumulate history
        assert len(client.get("/api/history").json()["kpi"]) > 0
        assert "events" in client.get("/api/events").json()
        first = cfg["sites"][0]["id"]
        nd = client.get("/node/" + first).json()
        assert nd["components"] and "history" in nd
        # per-element operation endpoint
        cid = nd["components"][0]["id"]
        comp = client.get(f"/api/component/{first}/{cid}").json()
        assert comp["id"] == cid and "history" in comp and "current" in comp
        # Scenario comparison (small horizon)
        cmp = client.post("/api/compare", json={"steps": 24, "seed": 42}).json()
        assert "none" in cmp["result"] and "classical" in cmp["result"]
        for kpis in cmp["result"].values():
            assert "storage_adjusted_cost_eur" in kpis and "reliability" in kpis


_HAS_SB3 = __import__("importlib").util.find_spec("stable_baselines3") is not None


@pytest.mark.skipif(not _HAS_FASTAPI or not _HAS_SB3, reason="fastapi/sb3 not installed")
def test_model_registry_and_training_mechanism(tmp_path):
    """
    Mechanism test for the Training-mode feature: launches a very short PPO job
    (not a real training run — just enough steps to exercise the endpoints,
    progress callback, early-stop path and model-registry sidecar) and checks
    it can be selected as the active RL controller without restarting the app.
    """
    import time
    from starlette.testclient import TestClient
    from dashboard.network_app import create_app

    app, _ = create_app(rl_model_path=str(tmp_path / "unused"))
    with TestClient(app) as client:
        models_before = client.get("/api/models").json()["models"]

        started = client.post(
            "/api/train/start",
            json={"timesteps": 200, "n_steps": 64, "name": "pytest_mechanism_smoke", "seed": 0},
        ).json()
        assert started["state"] == "running"

        # A second concurrent start must be rejected (409), not silently ignored.
        conflict = client.post("/api/train/start", json={"timesteps": 200})
        assert conflict.status_code == 409

        for _ in range(60):
            st = client.get("/api/train/status").json()
            if st["state"] in ("completed", "error", "stopped"):
                break
            time.sleep(1)
        assert st["state"] == "completed", st
        assert st["save_path"].endswith("pytest_mechanism_smoke.zip")

        models_after = client.get("/api/models").json()["models"]
        assert len(models_after) == len(models_before) + 1
        new_model = next(m for m in models_after if m["name"] == "pytest_mechanism_smoke")
        assert new_model["has_metadata"] is True
        assert new_model["timesteps_completed"] >= 200
        assert new_model["n_sites"] == 7  # DEFAULT_CONFIG is the 7-node network
        assert new_model["compatible"] is True  # matches the app's own topology

        # Switch the live RL controller to the freshly trained model.
        sel = client.post("/api/models/select", json={"path": new_model["path"]}).json()
        assert sel["active"] == new_model["path"]
        assert client.get("/api/models").json()["active"] == new_model["path"]

    # A DIFFERENT app (3-node pilot network) sharing the same model directory
    # must flag the 7-node model as incompatible and refuse to activate it —
    # this is the guard against the shape-mismatch crash the model-selection
    # feature would otherwise risk (see hytwin.rl.network_environment
    # obs_dim_for_n_sites / infer_n_sites_from_obs_dim).
    from hytwin.simulation.scenario import Scenario
    pilot_config = str(ROOT / "config" / "italy_network_pilot.yaml")
    app2, _ = create_app(config_path=pilot_config, rl_model_path=str(tmp_path / "unused"))
    with TestClient(app2) as client2:
        reg = client2.get("/api/models").json()
        expected = reg["expected_obs_dim"]
        pilot_n_sites = len(Scenario.from_yaml(pilot_config).topology().site_ids)
        assert expected is not None and pilot_n_sites == 3
        mismatch = next(m for m in reg["models"] if m["name"] == "pytest_mechanism_smoke")
        assert mismatch["compatible"] is False

        resp = client2.post("/api/models/select", json={"path": mismatch["path"]})
        assert resp.status_code == 409
        assert "incompatible" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Live comparison (Traditional vs AI, side by side, identical conditions)
# ---------------------------------------------------------------------------

def test_live_compare_worker_diverges_after_cold_start():
    """Unit-level: FastAPI-independent check that strategies diverge from the
    controller's second decision onward (step 1 is identical by design — the
    classical controller's cold-start with no prev_state falls back to the
    same naive defaults as 'none')."""
    from dashboard.network_app import LiveCompareWorker, DEFAULT_CONFIG, DEFAULT_RL_MODEL

    w = LiveCompareWorker(DEFAULT_CONFIG, DEFAULT_RL_MODEL)
    w._build(["none", "classical"], seed=5)
    f1 = w.step_once()
    assert f1["kpis"]["none"] == f1["kpis"]["classical"]  # cold start: identical
    for _ in range(5):
        f = w.step_once()
    assert f["kpis"]["none"]["cum_cost"] != f["kpis"]["classical"]["cum_cost"]


@pytest.mark.skipif(not _HAS_FASTAPI or not _HAS_SB3, reason="fastapi/sb3 not installed")
def test_live_compare_endpoints_and_websocket(tmp_path):
    import time
    from starlette.testclient import TestClient
    from dashboard.network_app import create_app

    # A fresh checkout ships no pre-trained models (output/ is generated, not
    # versioned — see docs/09_usage_guide.md for real training runs), so the
    # 'rl' strategy needs a model trained during the test itself: a very
    # short smoke run (same mechanism as test_model_registry_and_training_
    # mechanism above), activated via the normal /api/models/select path.
    app, _ = create_app(rl_model_path=str(tmp_path / "unused"))
    with TestClient(app) as client:
        assert client.get("/api/live_compare/status").json()["configured"] is False

        # Fewer than 2 strategies must be rejected.
        bad = client.post("/api/live_compare/start", json={"strategies": ["classical"], "seed": 1})
        assert bad.status_code == 400

        trained = client.post(
            "/api/train/start",
            json={"timesteps": 200, "n_steps": 64, "name": "pytest_live_compare_smoke", "seed": 0},
        ).json()
        assert trained["state"] == "running"
        st = trained
        for _ in range(60):
            st = client.get("/api/train/status").json()
            if st["state"] in ("completed", "error", "stopped"):
                break
            time.sleep(1)
        assert st["state"] == "completed", st
        rl_path = st["save_path"][:-len(".zip")]
        select = client.post("/api/models/select", json={"path": rl_path})
        assert select.status_code == 200, select.json()

        started = client.post(
            "/api/live_compare/start",
            json={"strategies": ["none", "classical", "rl"], "seed": 7},
        )
        assert started.status_code == 200
        assert started.json()["running"] is True

        time.sleep(1.0)
        import json as _json
        with client.websocket_connect("/ws/compare") as ws:
            frame = _json.loads(ws.receive_text())
        assert set(frame["kpis"].keys()) == {"none", "classical", "rl"}
        for kpis in frame["kpis"].values():
            assert "cum_cost" in kpis and "cum_co2" in kpis and "self_sufficiency" in kpis

        # Pause / resume cycle: state is retained (step count keeps advancing
        # after resume, not reset).
        client.post("/api/live_compare/stop")
        paused = client.get("/api/live_compare/status").json()
        assert paused["running"] is False and paused["configured"] is True
        step_at_pause = paused["step"]

        client.post("/api/live_compare/resume")
        time.sleep(0.5)
        resumed = client.get("/api/live_compare/status").json()
        assert resumed["running"] is True
        assert resumed["step"] >= step_at_pause


# ---------------------------------------------------------------------------
# Learning curve (per-episode reward history during training)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_FASTAPI or not _HAS_SB3, reason="fastapi/sb3 not installed")
def test_training_captures_reward_history(tmp_path):
    """
    Mechanism test: a short training job (enough steps for a few episodes to
    complete, episode_steps=144 on the default network) must report each
    completed episode's return as a {step, reward} point on
    /api/train/status, while the lightweight main /status must NOT carry
    this (potentially long) history — only the dedicated training endpoint
    should, to keep the 1s global status poll cheap on long training runs.
    """
    import time
    from starlette.testclient import TestClient
    from dashboard.network_app import create_app

    app, _ = create_app(rl_model_path=str(tmp_path / "unused"))
    with TestClient(app) as client:
        started = client.post(
            "/api/train/start",
            json={"timesteps": 450, "n_steps": 128, "name": "pytest_reward_smoke", "seed": 0},
        ).json()
        assert started["state"] == "running"

        for _ in range(60):
            st = client.get("/api/train/status").json()
            if st["state"] in ("completed", "error", "stopped"):
                break
            time.sleep(1)
        assert st["state"] == "completed", st

        history = st["reward_history"]
        assert len(history) >= 2  # 450 steps / 144-step episodes -> >= 3 completed
        assert all("step" in p and "reward" in p for p in history)
        # steps must be strictly increasing (one point per completed episode)
        assert all(history[i]["step"] < history[i + 1]["step"] for i in range(len(history) - 1))

        # The lightweight global status must not embed the reward history.
        main_status = client.get("/status").json()
        assert "reward_history" not in main_status["training"]


def test_objective_terms_match_env_reward():
    """The dashboard's objective panel uses the env reward's weights and per-network scales."""
    import numpy as np
    from dashboard.network_app import _objective_terms
    from hytwin.rl.network_environment import NetworkH2GridEnv
    from hytwin.simulation.scenario import Scenario
    topo = Scenario.from_yaml(DEFAULT_CONFIG).topology()
    env = NetworkH2GridEnv(topo)
    env.reset(seed=1)
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
    terms = _objective_terms(info["network_state"], topo)
    for k in ("cost", "co2", "unmet", "curtailment", "soc"):
        assert terms[k] == pytest.approx(info["reward_terms"][k], abs=1e-3), k
