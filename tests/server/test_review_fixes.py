"""Review must-fix findings (game-agnostic subset): seed is not an oracle, ?seat= is honoured."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from server.sessions import SessionStore
from tests.support.toy_adapter import NAME, ToyAdapter, toy_app

Json = dict[str, Any]


def _create(c: TestClient, seats: list[str], seed: int | None = None) -> Json:
    body: Json = {"game": NAME, "params": {"num_players": len(seats)}, "seats": seats}
    if seed is not None:
        body["seed"] = seed
    r = c.post("/api/games", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _view(c: TestClient, gid: str, seat: int) -> Json:
    r = c.get(f"/api/games/{gid}", params={"seat": seat})
    assert r.status_code == 200, r.text
    return r.json()


def _act(c: TestClient, gid: str, action: int) -> Json:
    r = c.post(f"/api/games/{gid}/actions", json={"action": action})
    assert r.status_code == 200, r.text
    return r.json()


def _keys(x: Any) -> set[str]:
    if isinstance(x, dict):
        return set(x) | {k for v in x.values() for k in _keys(v)}
    if isinstance(x, list):
        return {k for v in x for k in _keys(v)}
    return set()


# ------------------------------------------------------------------ A: seed is not an oracle


def test_seed_and_salt_never_in_client_responses() -> None:
    c = TestClient(toy_app())
    g = _create(c, ["human", "random"], seed=424242)
    gid = g["id"]
    bodies = [json.dumps(g)]
    for seat in (0, 1):
        bodies.append(json.dumps(_view(c, gid, seat)))
    for _ in range(3):
        if g["is_terminal"]:
            break
        g = _act(c, gid, g["legal_actions"][0]["id"])
    exp = c.get(f"/api/games/{gid}/export")
    assert exp.status_code == 200, exp.text
    bodies.append(exp.text)
    for seat in (0, 1):
        bodies.append(json.dumps(_view(c, gid, seat)))
    for b in bodies:
        assert "seed" not in _keys(json.loads(b)) and "salt" not in _keys(json.loads(b))
        assert "424242" not in b


def test_export_import_replays_without_seed_and_old_exports_import() -> None:
    c = TestClient(toy_app())
    g = _create(c, ["human", "random"])
    gid = g["id"]
    exp = c.get(f"/api/games/{gid}/export").json()
    assert "seed" not in exp
    r = c.post("/api/games/import", json=exp)
    assert r.status_code == 200, r.text
    assert r.json()["history"] == _view(c, gid, 0)["history"]
    r = c.post("/api/games/import", json={**exp, "seed": 7})  # old export format
    assert r.status_code == 200, r.text


def test_salt_changes_bot_and_chance_streams() -> None:
    store = SessionStore({NAME: ToyAdapter()})
    # Many seeds so a coincidental match of two short random histories cannot pass the test.
    differs = 0
    for seed in range(5, 15):
        a = store.create(NAME, {"num_players": 2}, ["random", "random"], seed, salt="aaa")
        b = store.create(NAME, {"num_players": 2}, ["random", "random"], seed, salt="bbb")
        d = store.create(NAME, {"num_players": 2}, ["random", "random"], seed, salt="aaa")
        assert a.history == d.history
        differs += a.history != b.history
    assert differs > 0  # same seed, other salt: not reproducible from the seed


def test_default_games_are_not_seed_reproducible() -> None:
    c = TestClient(toy_app())
    histories = {
        tuple(e["label"] for e in _create(c, ["random", "random"])["history"]) for _ in range(8)
    }
    assert len(histories) > 1


# ------------------------------------------------------------------ C: mutations honour ?seat=


def test_mutation_responses_render_the_requested_seat() -> None:
    c = TestClient(toy_app(dev_mode=True))
    g = _create(c, ["human", "human"], seed=3)
    gid = g["id"]
    a = g["legal_actions"][0]["id"]
    r = c.post(f"/api/games/{gid}/actions?seat=0", json={"action": a})
    assert r.status_code == 200 and r.json()["seat"] == 0
    r = c.post(f"/api/games/{gid}/back?seat=1")
    assert r.status_code == 200 and r.json()["seat"] == 1
    r = c.post(f"/api/games/{gid}/rewind?seat=1", json={"player_actions": 0})
    assert r.status_code == 200 and r.json()["seat"] == 1
