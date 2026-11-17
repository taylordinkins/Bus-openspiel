"""Phase 3 unit 6: heuristic/MCTS seats and the per-game think-time cap."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import server.bots as bots_mod
from server.app import create_app

Json = dict[str, Any]


def _create(c: TestClient, seats: list[str], **extra: Any) -> Json:
    r = c.post(
        "/api/games",
        json={
            "game": "bus",
            "params": {"num_players": len(seats)},
            "seats": seats,
            "seed": 4,
            **extra,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("seat", ["random", "heuristic", "mcts:5", "mcts:5-heuristic"])
def test_each_bot_type_plays_a_full_game(seat: str) -> None:
    c = TestClient(create_app(inline_bots=True))
    seats = [seat, "random", "random"]
    g = _create(c, seats, bot_think_time_s=0.5)
    assert g["is_terminal"] and g["bot_error"] is None
    assert g["seat_types"] == seats
    assert g["bot_think_time_s"] == 0.5


@pytest.mark.parametrize("v", [0.05, 61, "x"])
def test_think_time_validation(v: Any) -> None:
    c = TestClient(create_app(inline_bots=True))
    r = c.post(
        "/api/games",
        json={"game": "bus", "seats": ["random"] * 3, "bot_think_time_s": v},
    )
    assert r.status_code == 422


def test_think_time_default_and_export_round_trip() -> None:
    c = TestClient(create_app(inline_bots=True))
    g = _create(c, ["human", "random", "random"])
    assert g["bot_think_time_s"] == 5.0
    assert "bot_think_time_s" not in c.get(f"/api/games/{g['id']}/export").json()
    g = _create(c, ["mcts:5", "random", "random"], bot_think_time_s=2.5)
    exp = c.get(f"/api/games/{g['id']}/export").json()
    assert exp["bot_think_time_s"] == 2.5
    g2 = c.post("/api/games/import", json=exp).json()
    assert g2["bot_think_time_s"] == 2.5 and g2["history"] == g["history"]


class RecordingSearch:
    limits: list[float | None] = []

    def __init__(self, limit: float | None = None) -> None:
        self.limit = limit

    def with_time_limit(self, s: float) -> RecordingSearch:
        return RecordingSearch(s)

    def choose(self, state: Any, rng: Any) -> int:
        RecordingSearch.limits.append(self.limit)
        return rng.choice(state.legal_actions())


@pytest.mark.parametrize("inline", [True, False])
def test_think_time_reaches_the_bot(monkeypatch: pytest.MonkeyPatch, inline: bool) -> None:
    RecordingSearch.limits = []
    monkeypatch.setitem(bots_mod.BOTS, "mcts", RecordingSearch())
    c = TestClient(create_app(inline_bots=inline))
    g = _create(c, ["mcts", "random", "random"], bot_think_time_s=0.2)
    deadline = time.monotonic() + 20
    while not g["is_terminal"] and time.monotonic() < deadline:
        time.sleep(0.05)
        g = c.get(f"/api/games/{g['id']}").json()
    assert g["is_terminal"]
    assert RecordingSearch.limits and set(RecordingSearch.limits) == {0.2}


class BrokenSearch:
    def with_time_limit(self, s: float) -> BrokenSearch:
        return self

    def choose(self, state: Any, rng: Any) -> int:
        raise RuntimeError("no result")


class IllegalSearch(BrokenSearch):
    def choose(self, state: Any, rng: Any) -> int:
        return -12345


@pytest.mark.parametrize("stub", [BrokenSearch, IllegalSearch])
def test_search_failure_falls_back_to_heuristic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, stub: type
) -> None:
    monkeypatch.setitem(bots_mod.BOTS, "mcts", stub())
    c = TestClient(create_app(inline_bots=True))
    with caplog.at_level(logging.WARNING):
        g = _create(c, ["mcts", "random", "random"])
    assert g["is_terminal"] and g["bot_error"] is None
    assert "heuristic" in caplog.text


def test_non_search_bot_error_still_sets_bot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def choose(self, state: Any, rng: Any) -> int:
            raise RuntimeError("boom")

    monkeypatch.setitem(bots_mod.BOTS, "heuristic", Boom())
    c = TestClient(create_app(inline_bots=True))
    g = _create(c, ["heuristic", "random", "random"])
    assert g["bot_error"] and "boom" in g["bot_error"]


def test_real_mcts_respects_tiny_time_limit() -> None:
    c = TestClient(create_app(inline_bots=True))
    t = time.monotonic()
    g = _create(c, ["mcts:20000", "random", "random"], bot_think_time_s=0.1)
    assert g["bot_error"] is None
    assert time.monotonic() - t < 60  # unbounded 20000-sim moves would take far longer


def test_net_seat_listed_and_plays_a_full_game() -> None:
    c = TestClient(create_app(inline_bots=True))
    models = c.get("/api/bot-models").json()
    assert [m["seat_type"] for m in models] == ["net:ppo-long", "net:ppo-smoke"]
    assert models[0]["label"] == "Trained net" and 4 in models[0]["players"]
    seats = ["net:ppo-long", "random", "heuristic", "random"]
    g = _create(c, seats)
    assert g["is_terminal"] and g["bot_error"] is None and g["seat_types"] == seats


def test_net_seat_bot_moves_after_human_move() -> None:
    c = TestClient(create_app(inline_bots=True))
    g = _create(c, ["human", "net:ppo-long", "random"])
    # The human plays until a net/random seat has acted; the bots then move without bot_error.
    for _ in range(6):
        if g["current_player"] != 0:
            break
        first = g["legal_actions"][0]
        a = first["id"] if isinstance(first, dict) else first
        r = c.post(f"/api/games/{g['id']}/actions", json={"action": a})
        assert r.status_code == 200, r.text
        g = r.json()
    assert g["bot_error"] is None
    assert {h["actor"] for h in g["history"]} >= {0, 1}


@pytest.mark.parametrize(
    "seat", ["net:missing", "net:../ppo-long", "net:ppo-long.npz", "net", "net:/etc/passwd"]
)
def test_bad_net_seats_rejected(seat: str) -> None:
    c = TestClient(create_app(inline_bots=True))
    r = c.post("/api/games", json={"game": "bus", "seats": [seat, "random", "random"]})
    assert r.status_code == 422, r.text


def test_net_seat_wrong_player_count_rejected() -> None:
    c = TestClient(create_app(inline_bots=True))
    r = c.post("/api/games", json={"game": "bus", "seats": ["net:ppo-long", "random"]})
    assert 400 <= r.status_code < 500, r.text


def test_net_mcts_seat_listed_and_plays_a_full_game() -> None:
    c = TestClient(create_app(inline_bots=True))
    model = c.get("/api/bot-models").json()[0]
    assert [o["seat_type"] for o in model["search"]] == [
        "mcts_net:50-ppo-long",
        "mcts_net:200-ppo-long",
        "mcts_net:1000-ppo-long",
    ]
    seats = ["mcts_net:3-ppo-long", "random", "net:ppo-long", "heuristic"]
    g = _create(c, seats, bot_think_time_s=2)
    assert g["is_terminal"] and g["bot_error"] is None and g["seat_types"] == seats


@pytest.mark.parametrize("seat", ["mcts_net:5-missing", "mcts_net:5-../x", "mcts_net:ppo-long"])
def test_net_mcts_bad_seat_rejected(seat: str) -> None:
    c = TestClient(create_app(inline_bots=True))
    r = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": 3}, "seats": [seat, "random", "random"]},
    )
    assert r.status_code in (400, 422)
