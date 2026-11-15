"""Phase 2 unit 9: read-only playback frames (`GET /api/games/{id}?at=N`, UI_DESIGN 6.8)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from server.app import create_app

Json = dict[str, Any]


def _game(c: TestClient) -> Json:
    r = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": 3}, "seats": ["human"] * 3, "seed": 2},
    )
    assert r.status_code == 200, r.text
    g = r.json()
    for _ in range(6):
        g = c.post(
            f"/api/games/{g['id']}/actions", json={"action": g["legal_actions"][0]["id"]}
        ).json()
    return g


def test_frame_matches_the_view_at_that_point() -> None:
    c = TestClient(create_app(inline_bots=True))
    seen: list[Json] = []
    r = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": 3}, "seats": ["human"] * 3, "seed": 2},
    )
    g = r.json()
    seen.append(g)
    for _ in range(6):
        g = c.post(
            f"/api/games/{g['id']}/actions", json={"action": g["legal_actions"][0]["id"]}
        ).json()
        seen.append(g)
    for k, past in enumerate(seen):
        f = c.get(f"/api/games/{g['id']}?at={k}&seat={past['seat']}").json()
        assert f["history_length"] == k
        assert f["history"] == past["history"]
        assert f["view"] == past["view"]
        assert f["legal_actions"] == past["legal_actions"]


def test_frame_at_current_length_is_the_live_view() -> None:
    c = TestClient(create_app(inline_bots=True))
    g = _game(c)
    live = c.get(f"/api/games/{g['id']}").json()
    assert c.get(f"/api/games/{g['id']}?at={g['history_length']}").json() == live


def test_frame_out_of_range_is_409() -> None:
    c = TestClient(create_app(inline_bots=True))
    g = _game(c)
    assert c.get(f"/api/games/{g['id']}?at={g['history_length'] + 1}").status_code == 409
    assert c.get(f"/api/games/{g['id']}?at=-1").status_code == 409


def test_frame_does_not_change_the_game() -> None:
    c = TestClient(create_app(inline_bots=True))
    g = _game(c)
    c.get(f"/api/games/{g['id']}?at=1")
    assert c.get(f"/api/games/{g['id']}").json()["history_length"] == g["history_length"]


def test_frame_marks_bot_turns_as_thinking() -> None:
    c = TestClient(create_app(inline_bots=True))
    r = c.post(
        "/api/games",
        json={
            "game": "bus",
            "params": {"num_players": 3},
            "seats": ["human", "random", "random"],
            "seed": 4,
        },
    )
    g = r.json()
    # human moves until the inline bots have played some turns
    while not any(e["actor"] != 0 for e in g["history"]):
        assert g["current_player"] == 0
        a = g["legal_actions"][0]["id"]
        g = c.post(f"/api/games/{g['id']}/actions", json={"action": a}).json()
    for k in range(g["history_length"]):
        f = c.get(f"/api/games/{g['id']}?at={k}").json()
        cp = f["current_player"]
        assert f["bot_thinking"] == (cp is not None and f["seats"][cp] != "human")
        assert f["bot_error"] is None
