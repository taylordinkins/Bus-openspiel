"""Phase 2 unit 10: named scenarios, scenario start (named / pasted), rejection, round trips."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.app import create_app

Json = dict[str, Any]
SCEN = Path(__file__).resolve().parent.parent / "scenarios"


def _client() -> TestClient:
    return TestClient(create_app(inline_bots=True))


def _list(c: TestClient) -> list[Json]:
    r = c.get("/api/scenarios")
    assert r.status_code == 200, r.text
    return r.json()


def test_list_named_scenarios() -> None:
    items = _list(_client())
    assert items, "the manifest offers at least one scenario"
    ids = [s["id"] for s in items]
    assert len(ids) == len(set(ids))
    for s in items:
        assert set(s) == {
            "id",
            "game",
            "title",
            "description",
            "players",
            "returns",
            "round",
            "phase",
        }
        assert s["game"] == "bus" and s["title"]
        assert 3 <= s["players"] <= 5
    assert "fresh_4p" in ids


def test_list_unknown_game_404() -> None:
    assert _client().get("/api/scenarios", params={"game": "nope"}).status_code == 404


def test_get_named_scenario_and_unknown_404() -> None:
    c = _client()
    r = c.get("/api/scenarios/fresh_4p")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scenario"]["format"] == 1
    assert "expect" not in body["scenario"]
    assert c.get("/api/scenarios/no_such").status_code == 404


@pytest.mark.parametrize("sid", [s["id"] for s in _list(_client())])
def test_every_named_scenario_starts_a_game(sid: str) -> None:
    c = _client()
    sc = c.get(f"/api/scenarios/{sid}").json()
    n = sc["players"]
    r = c.post(
        "/api/games", json={"game": "bus", "seats": ["human"] * n, "scenario": sc["scenario"]}
    )
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["view"]["view"]["state"]["phase"] == sc["phase"]
    assert g["view"]["view"]["state"]["parameters"]["returns"] == sc["returns"]


def test_scenario_returns_taken_when_unset_and_mismatch_rejected() -> None:
    c = _client()
    doc = json.loads((SCEN / "end" / "returns_margin_3p.json").read_text())
    doc.pop("expect", None)
    r = c.post("/api/games", json={"game": "bus", "seats": ["human"] * 3, "scenario": doc})
    assert r.status_code == 200, r.text
    assert r.json()["view"]["view"]["state"]["parameters"]["returns"] == "margin"
    r = c.post(
        "/api/games",
        json={"seats": ["human"] * 3, "params": {"returns": "win_loss"}, "scenario": doc},
    )
    assert r.status_code == 422
    assert "returns" in r.json()["detail"]


def test_pasted_scenario_starts_game() -> None:
    c = _client()
    doc = json.loads((SCEN / "sel_round1.json").read_text())
    r = c.post(
        "/api/games",
        json={"game": "bus", "seats": ["human", "random", "human", "random"], "scenario": doc},
    )
    assert r.status_code == 200, r.text
    assert r.json()["view"]["view"]["state"]["phase"] == "SEL"


@pytest.mark.parametrize(
    "doc",
    [
        {"format": 2},
        {"format": 1, "bogus": 1},
        {"format": 1, "parameters": {"players": 7}},
        {"no": "format"},
    ],
)
def test_invalid_scenario_rejected_with_message(doc: Json) -> None:
    r = _client().post("/api/games", json={"seats": ["human"] * 4, "scenario": doc})
    assert 400 <= r.status_code < 500, r.text
    assert "invalid scenario" in r.json()["detail"]


def test_seat_count_mismatch_rejected() -> None:
    r = _client().post("/api/games", json={"seats": ["human"] * 3, "scenario": {"format": 1}})
    assert 400 <= r.status_code < 500
    assert r.json()["detail"]


def _play(c: TestClient, g: Json, steps: int) -> Json:
    rng = random.Random(3)
    for _ in range(steps):
        if g["is_terminal"]:
            break
        a = rng.choice(g["legal_actions"])["id"]
        r = c.post(f"/api/games/{g['id']}/actions", json={"action": a})
        assert r.status_code == 200, r.text
        g = r.json()
    return g


@pytest.mark.parametrize("sid", ["fresh_4p", "resolve_ln", None])
def test_export_import_round_trip(sid: str | None) -> None:
    c = _client()
    if sid is None:
        body: Json = {"params": {"num_players": 3}, "seats": ["human"] * 3, "seed": 5}
    else:
        sc = c.get(f"/api/scenarios/{sid}").json()
        body = {"seats": ["human"] * sc["players"], "scenario": sc["scenario"]}
    g = c.post("/api/games", json=body).json()
    g = _play(c, g, 15)
    exp = c.get(f"/api/games/{g['id']}/export").json()
    assert ("scenario" in exp) == (sid is not None)
    # The lobby posts the export (with the seat types chosen there) to /api/games/import.
    r = c.post("/api/games/import", json=exp)
    assert r.status_code == 200, r.text
    g2 = r.json()
    assert g2["view"]["view"]["state"] == g["view"]["view"]["state"]
    assert g2["history_length"] == g["history_length"]
    # Exporting again gives the same file.
    assert c.get(f"/api/games/{g2['id']}/export").json() == exp


def test_view_state_round_trips_as_scenario() -> None:
    """The position (view.state) pasted back as a scenario gives an equal state (UI_DESIGN 6.1)."""
    c = _client()
    g = _play(c, c.post("/api/games", json={"seats": ["human"] * 4, "seed": 2}).json(), 20)
    pos = g["view"]["view"]["state"]
    r = c.post("/api/games", json={"seats": ["human"] * 4, "scenario": pos})
    assert r.status_code == 200, r.text
    assert r.json()["view"]["view"]["state"] == pos
