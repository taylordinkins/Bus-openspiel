"""Phase 2 unit 2: Bus through the HTTP API with the default adapter registry."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.app import create_app

Json = dict[str, Any]


def _client() -> TestClient:
    return TestClient(create_app(inline_bots=True))


def _new(c: TestClient, n: int = 4, seed: int = 1) -> Json:
    r = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": n}, "seats": ["human"] * n, "seed": seed},
    )
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("n", [3, 4, 5])
def test_create_bus_game(n: int) -> None:
    g = _new(_client(), n)
    assert g["game"] == "bus"
    assert g["seats"] == ["human"] * n
    assert len(g["view"]["view"]["state"]["seats"]) == n


def test_create_rejects_two_and_six_players() -> None:
    c = _client()
    for n in (2, 6):
        r = c.post(
            "/api/games",
            json={"game": "bus", "params": {"num_players": n}, "seats": ["human"] * n},
        )
        assert 400 <= r.status_code < 500


def test_get_view_shape() -> None:
    c = _client()
    g = _new(c)
    got = c.get(f"/api/games/{g['id']}").json()
    gv = got["view"]
    assert gv["current_player"] == 0
    assert gv["is_terminal"] is False
    assert set(gv["view"]) == {"state", "derived", "prompt", "targets"}
    assert gv["view"]["prompt"]["phase"] == "SB"
    assert gv["view"]["prompt"]["actor"] == 0


def test_legal_actions_entries() -> None:
    c = _client()
    gv = _new(c)["view"]
    assert gv["legal_actions"]
    for la in gv["legal_actions"]:
        assert isinstance(la["id"], int)
        assert la["label"].startswith("P0: build ")
        assert la["kind"] == "build"
        assert isinstance(la["params"], dict)  # ACTIONS 5 keys; the shell's schema says list[int]
        assert {"site", "node", "tier", "building_type"} <= set(la["params"]) | set(la)


def test_targets_match_legal_actions_over_http() -> None:
    c = _client()
    gv = _new(c)["view"]
    sites = gv["view"]["targets"]["sites"]
    flat = [i for ids in sites.values() for i in ids]
    assert sorted(flat) == sorted(la["id"] for la in gv["legal_actions"])


def test_post_action_history_label_matches_legal_label() -> None:
    c = _client()
    g = _new(c)
    gid = g["id"]
    la = g["view"]["legal_actions"][5]
    r = c.post(f"/api/games/{gid}/actions", json={"action": la["id"]})
    assert r.status_code == 200, r.text
    g2 = r.json()
    hist = g2["view"]["history"] if "history" in g2["view"] else g2["history"]
    assert hist[-1]["label"] == la["label"]
    assert hist[-1]["id"] == la["id"]
    assert hist[-1]["actor"] == 0
    assert g2["view"]["view"]["prompt"]["context"] == {"done": 1}
    assert (
        g2["view"]["view"]["state"]["move_offset"] == g["view"]["view"]["state"]["move_offset"] + 1
    )


def test_play_through_setup_and_first_selection_over_http() -> None:
    """Drive a 4-player game through setup (16 nodes) and into SEL, checking history labels."""
    c = _client()
    g = _new(c, 4)
    labels = []
    for _ in range(16):
        la = g["view"]["legal_actions"][0]
        labels.append(la["label"])
        g = c.post(f"/api/games/{g['id']}/actions", json={"action": la["id"]}).json()
    assert g["view"]["view"]["prompt"]["phase"] == "SEL"
    hist = g["history"] if "history" in g else g["view"]["history"]
    assert [h["label"] for h in hist] == labels
    assert not any("Player 1" in lab for lab in labels)
