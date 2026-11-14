"""action_meta, forced_action, game_result, max_game_length (ACTIONS 1.5, 3.12, 5, 6).

Phase 1 unit 13, written ahead of the code.

Locations are not fixed by the docs beyond ``bus.rules.forced_action``; ``tests.support.play.find``
looks for ``action_meta`` / ``game_result`` / ``forced_action`` in bus.rules, bus.actions,
bus.observer and bus.game.  ``game_result(state)`` is read through ``field`` so it may return a dict
or an object with ``end_types`` and ``winners``.

Tests are named test_ACT5_* (action_meta), test_ACT15_* (forced_action), test_ACT312_*
(game_result), test_ACT6_* (max_game_length).
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from bus import actions as A
from bus.board import load_board
from tests import harness as H
from tests.support import play as P

B = load_board()
TYPES = ["house", "office", "pub"]
NEXT = {"house": "office", "office": "pub", "pub": "house"}
ACTION_KEYS = ["lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player"]
KINDS = {
    "select",
    "drop_out",
    "extend",
    "build",
    "passenger_to",
    "clock_advance",
    "clock_pause",
    "pick_passenger",
    "deliver_to",
}


def meta(s: Any, a: int) -> dict[str, Any]:
    return P.find("action_meta")(s, a)


def line_end_nodes(line: list[int]) -> tuple[int, int]:
    """(end 0 node, end 1 node) of a line per ACTIONS 1.3."""
    st = [B.streets[x] for x in line]
    if len(st) == 1:
        return st[0].a, st[0].b
    first = {st[0].a, st[0].b} - {st[1].a, st[1].b}
    # end 0 is the endpoint of the first street not shared with the second
    n0 = first.pop() if first else st[0].a
    node = st[0].a if st[0].b == n0 else st[0].b
    for x in st[1:]:
        node = x.b if x.a == node else x.a
    return n0, node


def rep_states() -> list[Any]:
    """Decision states from random games of all sizes, with kinds spread over the game."""
    out = []
    for n, seed in [(3, 31), (4, 32), (5, 33), (4, 34), (3, 35), (5, 36)]:
        out += [s for s in P.random_game(n, seed, every=2) if not s.is_terminal()]
    return out


# ---- action_meta: common fields ------------------------------------------------------------------


def test_ACT5_common_fields_for_every_legal_action() -> None:
    """ACTIONS 5: every legal action has id, kind, player, label; JSON-serialisable; the kind
    matches decode_action; label equals action_to_string; player is the acting seat."""
    seen_kinds: set[str] = set()
    for s in rep_states():
        for a in s.legal_actions():
            m = meta(s, a)
            json.dumps(m)
            assert m["id"] == a
            kind, _ = A.decode_action(a)
            assert m["kind"] == kind
            assert m["player"] == s.current_player()
            assert m["label"] == s.action_to_string(a)
            seen_kinds.add(m["kind"])
    assert seen_kinds == KINDS


def test_ACT5_static_meta_for_any_action_id() -> None:
    """ACTIONS 5: via decode_action a static dict is available for any id (kind and id)."""
    s = P.game(4).new_initial_state()
    for a in range(A.NUM_DISTINCT_ACTIONS):
        m = meta(s, a)
        json.dumps(m)
        assert m["id"] == a
        assert m["kind"] == A.decode_action(a)[0]


def test_ACT5_meta_does_not_mutate_state() -> None:
    """action_meta is a pure read: the state's string and legal actions are unchanged."""
    for s in rep_states()[::25]:
        before, legal = str(s), list(s.legal_actions())
        for a in legal:
            meta(s, a)
        assert str(s) == before and list(s.legal_actions()) == legal


# ---- per kind ----------------------------


def _legal_of(kind: str) -> list[tuple[Any, int]]:
    out = []
    for s in rep_states():
        for a in s.legal_actions():
            if A.decode_action(a)[0] == kind:
                out.append((s, a))
    return out


def test_ACT5_select_fields() -> None:
    """ACTIONS 5 select: action 0..6, its name, and space = the earliest free letter (R-025)."""
    pairs = _legal_of("select")
    assert pairs
    for s, a in pairs[::7]:
        m = meta(s, a)
        k = A.decode_action(a)[1][0]
        assert m["action"] == k
        norm = str(m["action_name"]).lower().replace("!", "").replace(" ", "_")
        assert norm == ACTION_KEYS[k]
        board = P.dump(s)["action_board"][ACTION_KEYS[k]]
        assert m["space"] == board.index(None)


def test_ACT5_drop_out_has_no_extra_targets() -> None:
    """ACTIONS 5 drop_out: only the common keys carry information."""
    pairs = _legal_of("drop_out")
    assert pairs
    s, a = pairs[0]
    m = meta(s, a)
    assert m["kind"] == "drop_out" and m["player"] == s.current_player()


def test_ACT5_extend_fields() -> None:
    """ACTIONS 5 extend: end, street, from_node (the end's node), to_node (new end), setup,
    shared_end (bool).  Checked against the seat's line and the board."""
    pairs = _legal_of("extend")
    assert pairs
    for s, a in pairs[::5]:
        m = meta(s, a)
        e, st = A.decode_action(a)[1]
        assert m["end"] == e and m["street"] == st
        assert isinstance(m["shared_end"], bool)
        ph = s.phase
        assert m["setup"] == {"L1": "first", "L2": "second"}.get(ph)
        street = B.streets[st]
        if ph == "L1":
            assert m["from_node"] is None
            assert m["shared_end"] is False
            continue
        line = list(s.seat(s.current_player()).line)
        frm = line_end_nodes(line)[e]
        assert m["from_node"] == frm
        assert frm in (street.a, street.b)
        assert m["to_node"] == (street.b if street.a == frm else street.a)


def test_ACT5_extend_first_setup_marker_has_no_from_node() -> None:
    """ACTIONS 5 / 2: the first setup marker (L1) has from_node null, setup == "first", end 0."""
    s = H.load_fixture("setup_l1_mid")
    assert s.phase == "L1"
    a = s.legal_actions()[0]
    m = meta(s, a)
    assert m["from_node"] is None and m["setup"] == "first" and m["end"] == 0
    assert m["street"] == A.decode_action(a)[1][1]


def test_ACT5_extend_second_setup_marker() -> None:
    """ACTIONS 5 extend in L2: setup == "second" and from_node is an end of the one-street line."""
    s = H.load_fixture("setup_l2_mid")
    assert s.phase == "L2"
    line = list(s.seat(s.current_player()).line)
    assert len(line) == 1
    for a in s.legal_actions():
        m = meta(s, a)
        assert m["setup"] == "second"
        assert m["from_node"] in (B.streets[line[0]].a, B.streets[line[0]].b)
        assert m["to_node"] != m["from_node"]


def test_ACT5_extend_in_lines_phase_has_no_setup_tag() -> None:
    """ACTIONS 5 extend: setup is null outside the setup phases (LN)."""
    s = H.load_fixture("lines/sequential")
    assert s.phase == "LN"
    for a in s.legal_actions():
        assert meta(s, a)["setup"] is None


def test_ACT5_build_fields() -> None:
    """ACTIONS 5 build: site, node, tier, building_type, setup (true only in SB)."""
    pairs = _legal_of("build")
    assert pairs
    for s, a in pairs[::9]:
        m = meta(s, a)
        t, y = A.decode_action(a)[1]
        assert m["site"] == t
        assert m["node"] == B.sites[t].node
        assert m["tier"] == B.sites[t].tier
        assert m["building_type"] == TYPES[y]
        assert m["setup"] is (s.phase == "SB")


def test_ACT5_passenger_to_fields() -> None:
    """ACTIONS 5 passenger_to: station index and its node (ascending node id, ACTIONS 1.2)."""
    pairs = _legal_of("passenger_to")
    assert pairs
    for s, a in pairs[::5]:
        m = meta(s, a)
        k = A.decode_action(a)[1][0]
        assert m["station"] == k
        assert m["node"] == B.stations[k]


def test_ACT5_clock_advance_from_to() -> None:
    """ACTIONS 5 clock_advance: from the clock's type to the next in the cycle (R-050)."""
    for name in ("clock/ck_movement", "clock/ck_occupied_pub"):
        s = H.load_fixture(name)
        m = meta(s, A.CLOCK_ADVANCE)
        assert m["from"] == s.clock
        assert m["to"] == NEXT[s.clock]


def test_ACT5_clock_pause_stones() -> None:
    """ACTIONS 5 clock_pause: stones_left_after = stones on clock - 1, ends_game only for the
    last stone (R-053)."""
    s = H.load_fixture("clock/ck_movement")
    m = meta(s, A.CLOCK_PAUSE)
    assert m["stones_left_after"] == s.stones_on_clock - 1 and m["ends_game"] is False
    last = H.load_fixture("clock/ck_last_stone_4p")
    m = meta(last, A.CLOCK_PAUSE)
    assert m["stones_left_after"] == 0 and m["ends_game"] is True
    m3 = meta(H.load_fixture("clock/ck_last_stone_3p"), A.CLOCK_PAUSE)
    assert m3["ends_game"] is True


def test_ACT5_pick_passenger_fields() -> None:
    """ACTIONS 5 pick_passenger: node and waiting count there."""
    pairs = _legal_of("pick_passenger")
    assert pairs
    for s, a in pairs[::5]:
        m = meta(s, a)
        n = A.decode_action(a)[1][0]
        assert m["node"] == n
        assert m["waiting"] == s.waiting_at(n) >= 1


def test_ACT5_deliver_to_fields() -> None:
    """ACTIONS 5 deliver_to: site, node, tier, building_type (the clock's type), picked_node,
    same_node (R-061)."""
    pairs = _legal_of("deliver_to")
    assert pairs
    same = set()
    for s, a in pairs:
        m = meta(s, a)
        t = A.decode_action(a)[1][0]
        picked = P.dump(s)["pending"]["picked_node"]
        assert m["site"] == t
        assert m["node"] == B.sites[t].node
        assert m["tier"] == B.sites[t].tier
        assert m["building_type"] == s.clock
        assert m["picked_node"] == picked
        assert m["same_node"] is (picked == B.sites[t].node)
        same.add(m["same_node"])
    assert same == {True, False} or same == {False}


# ---- forced_action ----------------------------


def test_ACT15_forced_action_matches_single_legal_action() -> None:
    """ACTIONS 1.5: forced_action returns the id exactly when one action is legal, else None."""
    f = P.find("forced_action")
    forced = other = 0
    for s in rep_states():
        legal = s.legal_actions()
        got = f(s)
        if len(legal) == 1:
            assert got == legal[0]
            forced += 1
        else:
            assert got is None
            other += 1
    assert forced > 0 and other > 0


def test_ACT15_forced_action_none_at_terminal() -> None:
    """ACTIONS 1.5: no legal action at END means no forced action."""
    f = P.find("forced_action")
    s = H.load_fixture("end_both")
    assert f(s) is None


def test_ACT15_forced_action_is_importable_from_rules() -> None:
    """ACTIONS 1.5 names the location: bus.rules.forced_action."""
    from bus import rules

    assert callable(rules.forced_action)


def test_ACT15_known_forced_and_free_positions() -> None:
    """ACTIONS 1.5: a lines position with a single legal extension is forced; the fresh game
    (many sites) is not."""
    from bus import rules

    one = H.load_fixture("lines/dead_one_end")
    assert len(one.legal_actions()) == 1
    assert rules.forced_action(one) == one.legal_actions()[0]
    assert rules.forced_action(P.game(4).new_initial_state()) is None


# ---- game_result ----------------------------


def test_ACT312_game_result_end_types_and_winners() -> None:
    """ACTIONS 3.12: game_result exposes the end types (R-066) and winners (R-069)."""
    gr = P.find("game_result")
    r = gr(H.load_fixture("end/tiebreak_stones"))
    assert sorted(P.field(r, "end_types")) == ["sites_full"]
    assert sorted(P.field(r, "winners")) == [0]
    r = gr(H.load_fixture("end/tiebreak_shared"))
    assert sorted(P.field(r, "winners")) == [0, 1]


def test_ACT312_both_end_kinds() -> None:
    """ACTIONS 3.12 / R-066: both ending kinds (round-end markers/sites_full, last stone) and the
    simultaneous pair are reported."""
    gr = P.find("game_result")
    assert sorted(P.field(gr(H.load_fixture("end_both")), "end_types")) == [
        "markers",
        "sites_full",
    ]
    assert list(P.field(gr(H.load_fixture("end_last_stone")), "end_types")) == ["last_stone"]


def test_ACT312_winners_agree_with_state_and_returns() -> None:
    """ACTIONS 3.12 vs R-071: winners are the seats with the top win_loss return, and end_types
    equals state.end_types, over played terminal games."""
    gr = P.find("game_result")
    for n in (3, 4, 5):
        for seed in range(3):
            s = P.random_game(n, 50 + seed, every=10**6)[-1]
            r = gr(s)
            assert list(P.field(r, "end_types")) == list(s.end_types)
            winners = sorted(P.field(r, "winners"))
            top = max(s.returns())
            assert winners == [p for p, v in enumerate(s.returns()) if v == top]


def test_ACT312_game_result_before_terminal_is_none_or_raises() -> None:
    """ACTIONS 3.12: END only; before the game ends there is no result (None or an exception;
    see report, the docs do not say which)."""
    gr = P.find("game_result")
    for s in (P.game(4).new_initial_state(), H.load_fixture("sel_round1")):
        try:
            r = gr(s)
        except Exception:  # noqa: BLE001
            continue
        assert r is None


def test_ACT312_game_result_is_json_serialisable() -> None:
    """The result feeds the server/UI later: end types and winners must be plain data."""
    gr = P.find("game_result")
    r = gr(H.load_fixture("end_both"))
    json.dumps({"end_types": list(P.field(r, "end_types")), "winners": list(P.field(r, "winners"))})


# ---- max_game_length ----------------------------


@pytest.mark.parametrize("n", [3, 4, 5])
def test_ACT6_max_game_length_declared(n: int) -> None:
    """ACTIONS 6: max_game_length = 234 * n."""
    assert P.game(n).max_game_length() == 234 * n


def test_ACT6_bound_holds_for_a_few_random_games() -> None:
    """ACTIONS 6: quick check on a handful of games (the 300-game run is the slow test)."""
    for n in (3, 4, 5):
        for seed in range(3):
            s = P.random_game(n, 700 + seed, every=10**6)[-1]
            assert len(s.history()) <= 234 * n


@pytest.mark.slow
@pytest.mark.parametrize("n", [3, 4, 5])
def test_ACT6_max_game_length_bound_over_300_random_games(n: int) -> None:
    """ACTIONS 6: no random game exceeds 234 * n decisions; records the observed maximum."""
    longest = 0
    for seed in range(300):
        rng = random.Random(10_000 * n + seed)
        s = P.game(n).new_initial_state()
        while not s.is_terminal():
            s.apply_action(rng.choice(s.legal_actions()))
        longest = max(longest, len(s.history()))
        assert len(s.history()) <= 234 * n, (n, seed)
    print(f"observed max game length n={n}: {longest} (bound {234 * n})")
