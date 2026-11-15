"""Phase 2 unit 8: adapter fields of docs/UI_DESIGN.md 6.7 (last_stone, street_ends, drop_out)."""

from __future__ import annotations

import pytest

import bus.game as bus_game
from server.adapters.bus import BusAdapter
from tests import harness as H
from tests.support import play as P

adapter = BusAdapter()


def _states() -> list:
    out = [
        H.load_fixture(f)
        for f in ("resolve_ln", "resolve_ck", "clock/ck_last_stone_3p", "setup_l1_mid",
                  "setup_l2_mid", "sel_round1", "sel_pass2_dropped", "lines/full_loop")
    ]  # fmt: skip
    for n, seed in [(3, 1), (4, 2), (5, 3)]:
        out += P.random_game(n, seed, every=5)
    return [s for s in out if not s.is_terminal()]


STATES = _states()


@pytest.mark.parametrize("fixture", ["resolve_ck", "clock/ck_last_stone_3p"])
def test_ck_last_stone_equals_pause_ends_game(fixture: str) -> None:
    s = H.load_fixture(fixture)
    pause = next(
        a for a in s.legal_actions() if bus_game.action_meta(s, a)["kind"] == "clock_pause"
    )
    ctx = adapter.view(s, 0)["prompt"]["context"]
    assert ctx["last_stone"] is bus_game.action_meta(s, pause)["ends_game"]


def test_ck_last_stone_both_values_seen() -> None:
    seen = {
        adapter.view(s, 0)["prompt"]["context"]["last_stone"]
        for s in STATES
        if s.core.phase == "CK"
    }
    assert seen == {False, True}


def test_street_ends_parallel_to_streets() -> None:
    n = 0
    for s in STATES:
        t = adapter.view(s, 0)["targets"]
        assert t["street_ends"].keys() == t["streets"].keys()
        for sid, ids in t["streets"].items():
            ends = t["street_ends"][sid]
            assert [e["id"] for e in ends] == ids
            for e in ends:
                m = bus_game.action_meta(s, e["id"])
                assert (e["end"], e["node"]) == (m["end"], m["from_node"])
                n += 1
    assert n > 20


def test_street_ends_loop_street_has_two_entries() -> None:
    s = H.load_fixture("lines/full_loop")
    t = adapter.view(s, 0)["targets"]
    assert any(len(v) == 2 for v in t["street_ends"].values())


def test_street_ends_empty_outside_line_phases() -> None:
    t = adapter.view(H.load_fixture("resolve_ck"), 0)["targets"]
    assert t["street_ends"] == {}


def test_drop_out_button_iff_legal_and_mandatory_is_its_negation() -> None:
    sel = [s for s in STATES if s.core.phase == "SEL"]
    assert sel
    for s in sel:
        v = adapter.view(s, 0)
        legal = any(bus_game.action_meta(s, a)["kind"] == "drop_out" for a in s.legal_actions())
        assert bool(v["targets"]["buttons"]) is legal
        assert v["prompt"]["context"]["mandatory"] is (not legal)
