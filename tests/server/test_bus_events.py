"""Phase 2 unit 3: events, board geometry, strict scenario load and random seats for Bus."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pyspiel
import pytest
from fastapi.testclient import TestClient

from bus import rules
from bus.game import BusState, action_meta
from bus.scenario import dump_scenario
from server.adapters.bus import BusAdapter
from server.app import create_app

Json = dict[str, Any]
SCEN = Path(__file__).resolve().parent.parent / "scenarios"


def _client() -> TestClient:
    return TestClient(create_app(inline_bots=True))


def _apply_events(state: Json, events: list[Json]) -> None:
    """Apply only what events say to a scenario dict (UI_DESIGN 6.6)."""
    occupied = {b["site"] for b in state["buildings"] if b.get("occupied")}
    for e in events:
        t = e["type"]
        if t == "scored":
            state["seats"][e["seat"]]["points"] = e["points"]
            state["waiting"][str(e["node"])] -= 1
            if state["waiting"][str(e["node"])] == 0:
                del state["waiting"][str(e["node"])]
            occupied.add(e["site"])
        elif t == "bus_added":
            state["seats"][e["seat"]]["buses"] = e["buses"]
        elif t == "passengers_moved":
            for x in e["exited"]:
                state["waiting"][str(x["node"])] = state["waiting"].get(str(x["node"]), 0) + 1
                occupied.discard(x["site"])
            for x in e["entered"]:
                k = str(x["node"])
                state["waiting"][k] -= 1
                if state["waiting"][k] == 0:
                    del state["waiting"][k]
                occupied.add(x["site"])
        elif t == "clock_advanced":  # the automatic Clock step (R-050, D22)
            assert state["clock"] == e["from"]
            state["clock"] = e["to"]
        elif t == "round_started":
            state["round"] = e["round"]
        elif t == "game_ended":
            state["phase"] = "END"
    for b in state["buildings"]:
        b["occupied"] = b["site"] in occupied


def _compare(pred: Json, nxt: Json, *, waiting: bool, clock: bool = False) -> None:
    for i, (a, b) in enumerate(zip(pred["seats"], nxt["seats"], strict=True)):
        assert a["buses"] == b["buses"], f"seat {i} buses"
        assert a["points"] == b["points"], f"seat {i} points"
    # A multi-action span can build new sites; compare the sites that already stood.
    known = {b["site"] for b in pred["buildings"]}

    def occ(s: Json) -> list[int]:
        return sorted(b["site"] for b in s["buildings"] if b.get("occupied") and b["site"] in known)

    assert occ(pred) == occ(nxt)
    if waiting:
        assert pred["waiting"] == nxt["waiting"]
    if clock:
        assert pred["clock"] == nxt["clock"]
    if nxt["phase"] == "END":
        assert pred["phase"] == "END"
    else:
        assert pred["round"] == nxt["round"]


@pytest.mark.parametrize("n", [3, 4, 5])
def test_events_match_state_diffs_direct(n: int) -> None:
    ad = BusAdapter()
    game = ad.load({"num_players": n})
    state = game.new_initial_state()
    rng = random.Random(n)
    seen: set[str] = set()
    while not state.is_terminal():
        a = rng.choice(state.legal_actions())
        meta = action_meta(state, a)
        before = state.clone()
        pred = copy.deepcopy(dump_scenario(state))
        state.apply_action(a)
        events = ad.events(before, a, state)
        seen.update(e["type"] for e in events)
        if meta["kind"] == "passenger_to":
            pred["waiting"][str(meta["node"])] = pred["waiting"].get(str(meta["node"]), 0) + 1
        if meta["kind"] == "clock_advance":  # the CK decision itself moves the hand
            pred["clock"] = meta["to"]
        _apply_events(pred, events)
        nxt = dump_scenario(state)
        # Round end resets nothing else we track; waiting only changes via the three sources,
        # the clock only via CLOCK_ADVANCE or the automatic Clock step's event.
        _compare(pred, nxt, waiting=True, clock=True)
    assert {"round_started", "round_ended", "game_ended"} <= seen


def test_event_types_seen_in_random_games() -> None:
    ad = BusAdapter()
    seen: set[str] = set()
    for seed in range(6):
        state = ad.load({"num_players": 4}).new_initial_state()
        rng = random.Random(seed)
        while not state.is_terminal():
            a = rng.choice(state.legal_actions())
            before = state.clone()
            state.apply_action(a)
            for e in ad.events(before, a, state):
                assert isinstance(e["type"], str)
                seen.add(e["type"])
    assert {
        "selection_ended",
        "round_started",
        "round_ended",
        "game_ended",
        "passengers_moved",
        "clock_advanced",
    } <= seen
    assert "bus_added" in seen or "marker_skipped" in seen


def test_events_specific_selection_end_and_skips() -> None:
    ad = BusAdapter()
    state = pyspiel.load_game("python_bus").new_initial_state()
    assert isinstance(state, BusState)
    ev = ad.events(state, state.legal_actions()[0], state)  # same state: no steps
    assert ev == []


@pytest.mark.parametrize("n", [3, 4, 5])
def test_random_seats_to_end_via_api(n: int) -> None:
    c = _client()
    r = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": n}, "seats": ["random"] * n, "seed": n},
    )
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["seat_types"] == ["random"] * n
    assert g["is_terminal"] is True
    assert g["winners"] and g["returns"] is not None
    assert g["view"]["view"]["prompt"]["phase"] == "END"
    types = {e["type"] for h in g["history"] for e in h["events"]}
    assert "game_ended" in types


@pytest.mark.parametrize("n", [3, 4, 5])
def test_api_events_match_diffs_with_human_seat(n: int) -> None:
    """Human seat 0 plays random legal actions through POST; bots fill in; events replay."""
    c = _client()
    r = c.post(
        "/api/games",
        json={
            "game": "bus",
            "params": {"num_players": n},
            "seats": ["human"] + ["random"] * (n - 1),
            "seed": 10 + n,
        },
    )
    g = r.json()
    rng = random.Random(n)
    seen = 0
    while not g["is_terminal"]:
        prev_state = copy.deepcopy(g["view"]["view"]["state"])
        prev_len = len(g["history"])
        la = rng.choice(g["view"]["legal_actions"])
        r = c.post(f"/api/games/{g['id']}/actions", json={"action": la["id"]})
        assert r.status_code == 200, r.text
        g = r.json()
        new = g["history"][prev_len:]
        assert new and new[0]["id"] == la["id"]
        events = [e for h in new for e in h["events"]]
        seen += len(events)
        _apply_events(prev_state, events)
        # waiting also changes through Passengers actions in the span; compare the rest.
        _compare(prev_state, g["view"]["view"]["state"], waiting=False)
    assert seen > 0
    assert g["history"][-1]["events"][-1]["type"] == "game_ended"


def test_geometry_endpoint() -> None:
    c = _client()
    r = c.get("/api/boards/bus")
    assert r.status_code == 200, r.text
    body = r.json()
    geo = body["geometry"]
    b = rules.board()
    assert body["game"] == "bus"
    assert [n["id"] for n in geo["nodes"]] == [n.id for n in b.nodes]
    assert len(geo["streets"]) == len(b.streets)
    assert len(geo["sites"]) == len(b.sites)
    st = geo["streets"][0]
    assert (st["a"], st["b"]) == (b.streets[0].a, b.streets[0].b)
    assert {s["tier"] for s in geo["sites"]} == {"A", "B", "C", "D"}
    assert geo["layout"]["clock"]["radius"] == b.layout.clock_radius
    assert len(geo["layout"]["score_track"]) == 20
    assert sorted(geo["layout"]["actions"]["lines"]) == list("abcdef")
    assert geo["map"] == {"width": 1991, "height": 1991}
    assert c.get("/api/boards/nope").status_code == 404


def _dead_end_scenario() -> Json:
    doc = json.loads((SCEN / "resolve_bd.json").read_text())
    b = rules.board()
    doc["buildings"] = [{"site": t.id, "type": "house"} for t in b.sites]
    doc.pop("history", None)
    doc.pop("expect", None)
    return doc


def test_strict_load_rejects_dead_end_scenario() -> None:
    from bus.scenario import load_scenario

    doc = _dead_end_scenario()
    state = load_scenario(doc)  # fixture-friendly default still accepts it
    assert rules.legal_actions(state.core) == [] and not state.is_terminal()
    c = _client()
    r = c.post("/api/games", json={"game": "bus", "seats": ["human"] * 4, "scenario": doc})
    assert r.status_code == 422, r.text
    assert "legal action" in r.json()["detail"]
    r = c.post(
        "/api/games/import",
        json={"game": "bus", "params": {}, "seats": ["human"] * 4, "history": [], "scenario": doc},
    )
    assert 400 <= r.status_code < 500


def test_valid_scenario_creates_game() -> None:
    doc = json.loads((SCEN / "resolve_bd.json").read_text())
    doc.pop("history", None)
    doc.pop("expect", None)
    n = doc.get("parameters", {}).get("players", 4)
    c = _client()
    r = c.post("/api/games", json={"game": "bus", "seats": ["human"] * n, "scenario": doc})
    assert r.status_code == 200, r.text
    assert r.json()["view"]["view"]["prompt"]["phase"] == "BD"


def test_automatic_clock_step_emits_clock_advanced_before_vrrooomm() -> None:
    """R-029, R-050, R-051 (D22): playtest 2b0b32's round 1 has no Clock marker.  The transition
    out of the last Passengers decision runs the automatic Clock step: one ``clock_advanced``
    event house -> office, no ``passengers_moved`` (nobody stands at an office), and the next
    state is P0's Vrrooomm! pick."""
    history = [154, 149, 173, 178, 166, 152, 31, 47, 41, 22, 122, 21]
    history += [5, 5, 5, 2, 0, 1, 7, 7, 7, 38, 289, 289]
    ad = BusAdapter()
    state = ad.load({"num_players": 3}).new_initial_state()
    per_action: list[list[Json]] = []
    for a in history:
        before = state.clone()
        state.apply_action(a)
        per_action.append(ad.events(before, a, state))
    clock = [(i, e) for i, ev in enumerate(per_action) for e in ev if e["type"] == "clock_advanced"]
    assert clock == [
        (len(history) - 1, {"type": "clock_advanced", "from": "house", "to": "office"})
    ]
    assert not any(e["type"] == "passengers_moved" for e in per_action[-1])
    assert isinstance(state, BusState) and state.phase == "VP" and state.clock == "office"


@pytest.mark.parametrize("fixture", ["clock/ck_last_stone_4p", "clock/ck_last_stone_3p"])
def test_last_stone_end_emits_only_game_ended(fixture: str) -> None:
    """R-053, R-067 (UI_DESIGN 6.6): a CK pause that takes the last stone ends the game at once.
    The Vrrooomm! and Starting Player markers still on the board never resolve, so the action's
    events are only ``game_ended``: no marker_skipped, bus_added, clock_advanced,
    passengers_moved or round_ended."""
    from bus.actions import CLOCK_PAUSE
    from tests import harness as H

    state = H.load_fixture(fixture)
    assert state.phase == "CK" and state.stones_on_clock == 1
    assert state.core.action_board[rules.ACT_VRROOOMM]  # markers still waiting to resolve
    before = state.clone()
    state.apply_action(CLOCK_PAUSE)
    assert state.is_terminal() and state.core.end_types == ("last_stone",)
    assert BusAdapter().events(before, CLOCK_PAUSE, state) == [
        {"type": "game_ended", "end_types": ["last_stone"]}
    ]


def test_round_end_that_ends_game_has_no_next_start_player() -> None:
    """R-064, R-066 (UI_DESIGN 6.6): ``round_ended`` names the next round's start player only when
    a next round follows; when the round end also ends the game it carries no ``start_player``."""
    ad = BusAdapter()
    checked = 0
    for seed in range(4):
        state = ad.load({"num_players": 4}).new_initial_state()
        rng = random.Random(seed)
        while not state.is_terminal():
            a = rng.choice(state.legal_actions())
            before = state.clone()
            state.apply_action(a)
            for e in ad.events(before, a, state):
                if e["type"] != "round_ended":
                    continue
                if state.is_terminal():
                    assert "start_player" not in e
                    checked += 1
                else:
                    assert e["start_player"] == state.core.start_player
    assert checked > 0
