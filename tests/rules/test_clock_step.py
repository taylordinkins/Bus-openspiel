"""The Clock step every round (D22): R-015, R-029, R-049 example, R-050..R-060, S-CLOCKSTEP.

Written ahead of the engine change.  Everything starts from
``tests/scenarios/clockstep/report_setup.json`` (3 players, round 1 selection, clock house, 4 stones
on the clock; helpers in ``tests/support/clockstep.py``).  Position in that fixture (ids from the
board data as used in playtest report 2b0b32):
  buildings  s0 office n10, s1 office n14, s2 house n13, s6 house n18, s8 office n20, s10 house n21
  lines      P0 n5-n10-n13, P1 n17-n20-n25, P2 n5-n14-n18          (stations n5, n25)
  waiting    one roundabout passenger each at n13 n17 n18 n21; supply 11; all seats 1 bus.
A round "with no Clock marker" is six Lines markers (``ALL_LINES``); only the Lines marker on
letter a has power 1 (M = 1), so the single Lines decision is resolved by its first legal action.

The three old R-061 same-node tests in tests/rules/test_vrrooomm.py were replaced when R-061 was
retired (D22).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from bus.actions import CLOCK_ADVANCE, CLOCK_PAUSE
from tests import harness as H
from tests.support import clockstep as C
from tests.support.clockstep import (
    ALL_LINES,
    BUILDINGS,
    BUSES,
    CLOCK,
    LINES,
    STARTING,
    VRROOOMM,
    deliver,
    pick,
)

NEXT = {"house": "office", "office": "pub", "pub": "house"}
ROUNDABOUTS = (13, 17, 18, 21)


def seats_stones(s: Any) -> list[int]:
    return [s.seat(p).stones for p in range(3)]


def one_round(s: Any) -> None:
    C.play_round(s, ALL_LINES)
    assert s.phase == "SEL"


def cleared_edit(**kw: Any) -> dict[str, Any]:
    d = C.setup_doc()
    for k, v in kw.items():
        d[k] = v
    return d


# ---- R-050 / R-051: automatic advance without a Clock marker -------------------------------


def test_R050_round_1_without_clock_marker_advances_house_to_office() -> None:
    """R-050 (round 1 included), R-051: nobody selects Clock in round 1; the clock goes house ->
    office at the Clock step, with no decision node and no stone taken."""
    s = C.setup_state()
    assert s.clock == "house"
    C.play_round(s, ALL_LINES)
    assert s.phase == "SEL" and C.next_round(s) == 2
    assert s.clock == "office"
    assert s.stones_on_clock == 4
    assert seats_stones(s) == [0, 0, 0]


def test_R050_automatic_advance_cycles_one_space_per_round() -> None:
    """R-050/R-051: with no Clock marker the hand moves exactly one space each round:
    house -> office -> pub -> house -> office; stones never change."""
    s = C.setup_state()
    seen: list[str] = []
    for _ in range(4):
        one_round(s)
        seen.append(s.clock)
    assert seen == ["office", "pub", "house", "office"]
    assert s.stones_on_clock == 4 and seats_stones(s) == [0, 0, 0]


@pytest.mark.parametrize("start", ["house", "office", "pub"])
def test_R051_automatic_advance_from_each_type(start: str) -> None:
    """R-051: advance from house/office/pub goes to the next type in the cycle."""
    d = C.setup_doc()
    d["clock"] = start
    s = C.load(d)
    one_round(s)
    assert s.clock == NEXT[start]


def test_R050_no_ck_decision_node_without_marker() -> None:
    """R-050/ACTIONS 3.9: without a Clock marker no CK node appears and the game never ends at
    the step; play_round reaches the next selection with the game still running."""
    s = C.setup_state()
    C.select(s, ALL_LINES)
    C.resolve_lines(s)
    assert s.phase != "CK" and not s.is_terminal()


# ---- R-050 / R-051 / R-052: a Clock marker --------------------------------------------------


def ck_state(clock: str = "house", **doc_edit: Any) -> Any:
    """P2 holds the Clock marker (plus Lines); returns the state at the CK node."""
    d = C.setup_doc()
    d["clock"] = clock
    for k, v in doc_edit.items():
        d[k] = v
    s = C.load(d)
    C.play_round(s, {0: [LINES, LINES], 1: [LINES, LINES], 2: [CLOCK, LINES]})
    return s


def test_R050_clock_marker_owner_chooses_advance_or_pause() -> None:
    """R-050: with a Clock marker the owner (P2) gets a decision node, advance or pause."""
    s = ck_state()
    assert s.phase == "CK" and s.current_player() == 2
    assert sorted(s.legal_actions()) == sorted([CLOCK_ADVANCE, CLOCK_PAUSE])


def test_R051_marker_advance_moves_one_space_takes_no_stone_and_moves_once() -> None:
    """R-050/R-051: P2's Clock marker chooses advance: house -> office once (not also the
    automatic advance, which would reach pub), no stone, still office when the round ends."""
    s = ck_state()
    s.apply_action(CLOCK_ADVANCE)
    assert s.clock == "office"
    assert s.stones_on_clock == 4 and seats_stones(s) == [0, 0, 0]
    assert s.phase == "SEL" and C.next_round(s) == 2
    assert s.clock == "office"


def test_R051_marker_advance_from_pub_wraps_to_house() -> None:
    """R-051: marker advance at pub gives house (the cycle wraps)."""
    s = ck_state("pub")
    s.apply_action(CLOCK_ADVANCE)
    assert s.clock == "house"


def test_R052_pause_keeps_clock_and_takes_a_stone() -> None:
    """R-052: pause leaves the hand at house and P2 takes 1 stone (4 -> 3 on the clock)."""
    s = ck_state()
    s.apply_action(CLOCK_PAUSE)
    assert s.clock == "house"
    assert s.stones_on_clock == 3 and seats_stones(s) == [0, 0, 1]
    assert not s.is_terminal()


def test_R052_pause_only_holds_for_the_marker_round_next_round_advances() -> None:
    """R-050/R-052: after a pause round, a round without a marker advances automatically."""
    s = ck_state()
    s.apply_action(CLOCK_PAUSE)
    one_round(s)
    assert s.clock == "office"
    assert s.stones_on_clock == 3


# ---- R-053: last stone --------------------------------------------------------------------


def last_stone_doc() -> dict[str, Any]:
    d = C.setup_doc()
    d["stones_on_clock"] = 1
    d["seats"][0]["stones"] = 3
    return d


def last_stone_state() -> Any:
    """P0 holds Clock, P1 two Vrrooomm! markers, P2 Lines; one stone left."""
    s = C.load(last_stone_doc())
    C.play_round(s, {0: [CLOCK, LINES], 1: [VRROOOMM, VRROOOMM], 2: [LINES, LINES]})
    return s


def test_R053_pause_with_last_stone_ends_game_without_passenger_movement() -> None:
    """R-053: P0 pauses taking the last stone: the game is over at once, nobody steps in (the
    roundabout passengers stay waiting although houses stand at n13/n18/n21), P1's Vrrooomm!
    markers do not resolve, and P0 keeps the stone."""
    s = last_stone_state()
    assert s.phase == "CK" and s.current_player() == 0
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal()
    assert s.stones_on_clock == 0 and s.seat(0).stones == 4
    assert C.occupied(s) == set()
    assert C.waiting(s) == {13: 1, 17: 1, 18: 1, 21: 1}
    assert [s.seat(p).points for p in range(3)] == [0, 0, 0]
    assert s.clock == "house"


def test_R053_advance_with_last_stone_continues_and_vrrooomm_resolves() -> None:
    """R-051/R-053: advancing does not take the stone; the game goes on, passengers move, and P1's
    Vrrooomm! (clock office) gets a node: P1 line n17-n20-n25 has the passenger at n17 and the
    empty office s8 at n20."""
    s = last_stone_state()
    s.apply_action(CLOCK_ADVANCE)
    assert not s.is_terminal()
    assert s.stones_on_clock == 1
    assert s.clock == "office"
    assert s.phase == "VP" and s.current_player() == 1
    assert pick(17) in s.legal_actions()


# ---- R-054 / R-055: step-out and step-in ----------------------------------------------------


def test_R054_everyone_steps_out_whatever_the_building_type() -> None:
    """R-054: a passenger inside a house (s2, n13) steps out at the automatic Clock step house ->
    office; no office at n13, so it waits there with the roundabout passenger (2 at n13)."""
    d = C.setup_doc()
    C.set_building(d, 2, "house", occupied=True)
    s = C.load(d)
    one_round(s)
    assert s.clock == "office"
    assert not s.is_occupied(2)
    assert s.waiting_at(13) == 2


def test_R055_step_in_at_the_automatic_step_only_into_the_new_type() -> None:
    """R-015 example/R-055: n13 has house s2 and an office s3; clock house, no Clock marker.
    The step moves the clock to office and the n13 passenger enters the office, not the house."""
    d = C.setup_doc()
    C.set_building(d, 3, "office")
    s = C.load(d)
    one_round(s)
    assert s.clock == "office"
    assert C.occupied(s) == {3}
    assert s.waiting_at(13) == 0


def test_R015_pause_in_round_1_steps_into_the_house() -> None:
    """R-015 example: the same position but P2's Clock marker pauses: clock stays house and the
    n13 passenger enters house s2 (not office s3); n18 enters s6 and n21 enters s10."""
    d = C.setup_doc()
    C.set_building(d, 3, "office")
    s = C.load(d)
    C.play_round(s, {0: [LINES, LINES], 1: [LINES, LINES], 2: [CLOCK, LINES]})
    s.apply_action(CLOCK_PAUSE)
    assert s.clock == "house"
    assert C.occupied(s) == {2, 6, 10}
    assert C.waiting(s) == {17: 1}


def test_R055_roundabout_passengers_step_in_on_the_third_automatic_round() -> None:
    """R-015/R-050/R-055: automatic rounds give office, pub (no pubs: nobody enters), house. At
    the third step the roundabout passengers at n13, n18, n21 enter houses s2, s6, s10; n17
    (no building) keeps waiting."""
    s = C.setup_state()
    one_round(s)
    one_round(s)
    assert C.occupied(s) == set()
    one_round(s)
    assert s.clock == "house"
    assert C.occupied(s) == {2, 6, 10}
    assert C.waiting(s) == {17: 1}


def test_R054_next_automatic_step_empties_every_building_then_steps_in_again() -> None:
    """R-054/R-055: a fourth automatic round (clock office): the houses empty, no office has a
    waiting passenger, so all four passengers wait again."""
    s = C.setup_state()
    for _ in range(4):
        one_round(s)
    assert s.clock == "office"
    assert C.occupied(s) == set()
    assert C.waiting(s) == {13: 1, 17: 1, 18: 1, 21: 1}


def test_R055_step_in_is_one_to_one_surplus_waits() -> None:
    """R-055: clock office; n21 holds 3 waiting passengers (roundabout 1 + 2) and two pubs
    (s9, s10).  The automatic step advances to pub: two enter, one waits."""
    d = C.setup_doc()
    d["clock"] = "office"
    C.set_building(d, 9, "pub")
    C.set_building(d, 10, "pub")
    C.set_waiting(d, {13: 1, 17: 1, 18: 1, 21: 3})
    s = C.load(d)
    one_round(s)
    assert s.clock == "pub"
    assert C.occupied(s) == {9, 10}
    assert s.waiting_at(21) == 1


def test_R055_more_buildings_than_passengers_leaves_buildings_empty() -> None:
    """R-055: min(waiting, buildings): one waiting passenger at n21 and two pubs: exactly one pub
    is filled (which one is not rule-relevant), the other stays empty, nobody waits at n21."""
    d = C.setup_doc()
    d["clock"] = "office"
    C.set_building(d, 9, "pub")
    C.set_building(d, 10, "pub")
    s = C.load(d)
    one_round(s)
    assert len(C.occupied(s) & {9, 10}) == 1
    assert s.waiting_at(21) == 0


def test_R055_stations_never_fill_at_the_automatic_step() -> None:
    """R-055/R-012: passengers waiting at stations n5/n25 stay waiting at every automatic step."""
    d = C.setup_doc()
    C.set_waiting(d, {13: 1, 17: 1, 18: 1, 21: 1, 5: 2, 25: 1})
    s = C.load(d)
    for _ in range(3):
        one_round(s)
        assert s.waiting_at(5) == 2 and s.waiting_at(25) == 1


def test_R055_post_step_invariant_holds_after_every_automatic_round() -> None:
    """R-055: after each Clock step no node has a waiting passenger and an empty building of the
    clock's type, and every occupied building is of the clock's type.  Checked over 6 rounds on
    a position with buildings of all types and extra waiting passengers."""
    d = C.setup_doc()
    C.set_building(d, 3, "office")
    C.set_building(d, 4, "pub")
    C.set_building(d, 5, "house")
    C.set_building(d, 9, "pub")
    C.set_waiting(d, {13: 3, 17: 2, 18: 1, 21: 2, 5: 1})
    s = C.load(d)
    sites_at = {13: (2, 3), 17: (4, 5), 18: (6, 7), 21: (9, 10), 14: (1,), 10: (0,), 20: (8,)}
    for _ in range(6):
        one_round(s)
        dd = C.dump(s)
        kind = {b["site"]: b["type"] for b in dd["buildings"]}
        for b in dd["buildings"]:
            if b.get("occupied"):
                assert b["type"] == s.clock, b
        for node, sites in sites_at.items():
            if s.waiting_at(node):
                for t in sites:
                    assert kind.get(t) != s.clock or s.is_occupied(t), (node, t)
        assert len(C.occupied(s)) + sum(C.waiting(s).values()) + s.passenger_supply == 15


# ---- R-029 / R-049: the step at its slot, including a sites-full round -------------------------


def test_R029_buildings_built_this_round_take_part_in_the_clock_step() -> None:
    """R-029/R-050: Buildings resolves before the Clock step.  P0 builds an office at a roundabout
    node; with no Clock marker the clock goes house -> office and the roundabout passenger
    there steps into the new office in the same round."""
    s = C.setup_state()
    C.play_round(s, {0: [BUILDINGS, LINES], 1: [LINES, LINES], 2: [LINES, LINES]})
    assert s.phase == "BD" and s.current_player() == 0
    target = None
    for a in s.legal_actions():
        m = re.search(r"build office at site (\d+) \(\w+, n(\d+)\)", s.action_to_string(0, a))
        if m and int(m.group(2)) in (13, 17, 18):
            target = (a, int(m.group(1)))
            break
    assert target is not None, "no legal office build at n13/n17/n18 in the lowest open tier"
    s.apply_action(target[0])
    while s.phase == "BD":
        s.apply_action(s.legal_actions()[0])
    assert s.phase == "SEL" and s.clock == "office"
    assert target[1] in C.occupied(s)


def test_R049_clock_step_runs_in_the_sites_full_round() -> None:
    """R-049 example/R-050: P0's Buildings fills the last site; no Clock marker, yet the step
    runs: house -> office, the n17 passenger enters office s4, then the game ends at round end
    with 'sites_full'."""
    d = C.setup_doc()
    for t in range(C.SITES):
        if t != 40:
            C.set_building(d, t, "office" if t == 4 else "house")
    s = C.load(d)
    C.play_round(s, {0: [BUILDINGS, LINES], 1: [LINES, LINES], 2: [LINES, LINES]})
    assert s.phase == "BD"
    s.apply_action(s.legal_actions()[0])
    assert s.is_terminal()
    assert "sites_full" in list(s.end_types)
    assert s.clock == "office"
    assert 4 in C.occupied(s)


# ---- R-056: Vrrooomm! sees the new clock type -------------------------------------------------


def vrr_state() -> Any:
    """Round 1, only P0 holds a Vrrooomm! marker (plus Starting Player); no Clock marker."""
    s = C.setup_state()
    C.play_round(s, {0: [VRROOOMM, STARTING], 1: [LINES, LINES], 2: [LINES, LINES]})
    return s


def test_R056_vrrooomm_in_round_1_uses_the_clock_after_the_step() -> None:
    """R-056/R-050: no Clock marker; at Vrrooomm! the clock reads office.  P0 (line n5-n10-n13)
    may pick only the n13 passenger and deliver only to office s0 at n10, not to house s2."""
    s = vrr_state()
    assert s.clock == "office"
    assert s.phase == "VP" and s.current_player() == 0
    assert set(s.legal_actions()) == {pick(13)}
    s.apply_action(pick(13))
    assert set(s.legal_actions()) == {deliver(0)}
    s.apply_action(deliver(0))
    assert s.seat(0).points == 1
    assert s.is_occupied(0) and not s.is_occupied(2)
    assert s.waiting_at(13) == 0


def test_R056_no_empty_building_of_the_new_type_no_node() -> None:
    """R-059/R-060: the second automatic round reads pub; there is no pub on the board, so P0's
    Vrrooomm! marker has count 0 and produces no node; the round simply ends."""
    s = C.setup_state()
    one_round(s)
    C.play_round(
        s, {p: [VRROOOMM if p == 0 else LINES, STARTING if p == 0 else LINES] for p in range(3)}
    )
    assert s.clock == "pub"
    assert s.phase == "SEL" and C.next_round(s) == 3
    assert s.seat(0).points == 0


# ---- R-058 / R-059 / R-060: eligibility, blocking, forced count -------------------------------


def blocked_doc() -> dict[str, Any]:
    """Position for pause tests: s0 (n10) is an empty house; waiting adds 2 at station n5 and 1
    at station n25 to the roundabout passengers."""
    d = C.setup_doc()
    C.set_building(d, 0, "house")
    C.set_waiting(d, {13: 1, 17: 1, 18: 1, 21: 1, 5: 2, 25: 1})
    return d


def paused_vp(buses_plan: bool = False) -> Any:
    """P2 pauses at house (n13, n18, n21 passengers enter s2, s6, s10); then P0's Vrrooomm!."""
    s = C.load(blocked_doc())
    p0 = [VRROOOMM, BUSES if buses_plan else STARTING]
    C.play_round(s, {0: p0, 1: [VRROOOMM, STARTING if buses_plan else BUSES], 2: [CLOCK, LINES]})
    assert s.phase == "CK"
    s.apply_action(CLOCK_PAUSE)
    return s


def test_R058_passenger_inside_a_building_of_the_clock_type_is_not_eligible() -> None:
    """R-058: after the pause the n13 passenger sits in house s2 and cannot be picked; the only
    pick for P0 (line n5-n10-n13) is the station n5 passenger.  R-059: the only empty house on
    the line is s0 (s2 is occupied) so s0 is the only delivery."""
    s = paused_vp()
    assert C.occupied(s) == {2, 6, 10}
    assert s.phase == "VP" and s.current_player() == 0
    assert set(s.legal_actions()) == {pick(5)}
    s.apply_action(pick(5))
    assert set(s.legal_actions()) == {deliver(0)}
    s.apply_action(deliver(0))
    assert s.seat(0).points == 1 and s.is_occupied(0) and s.waiting_at(5) == 1


def test_R059_blocked_marker_does_nothing() -> None:
    """R-059 blocking example/R-060: P1's marker (line n17-n20-n25): passengers wait at n17 and
    n25 but the line has no empty house (s8 is an office), so P1's marker has count 0, produces
    no node and scores nothing; the round ends after P0's single move."""
    s = paused_vp()
    s.apply_action(pick(5))
    s.apply_action(deliver(0))
    assert s.phase == "SEL" and C.next_round(s) == 2
    assert [s.seat(p).points for p in range(3)] == [1, 0, 0]


def test_R060_count_is_min_of_buses_eligible_and_empty_buildings() -> None:
    """R-060: P0 has 2 buses (Buses action), 2 eligible passengers at n5, but only one empty
    house on its line: moves_left is 1 and the marker ends after one delivery."""
    s = paused_vp(buses_plan=True)
    assert s.seat(0).buses == 2
    assert s.phase == "VP" and s.current_player() == 0
    assert C.dump(s)["pending"]["moves_left"] == 1
    s.apply_action(pick(5))
    s.apply_action(deliver(0))
    assert s.phase == "SEL"
    assert s.seat(0).points == 1


def test_R061_retired_delivery_is_never_to_the_picked_passengers_own_node() -> None:
    """R-059/R-061 (retired): in every delivery offered after a pick, the building's node differs
    from the node the passenger was picked at."""
    for s, picked in ((vrr_state(), 13), (paused_vp(), 5)):
        s.apply_action(pick(picked))
        for a in s.legal_actions():
            m = re.search(r"\(\w+, n(\d+)\)", s.action_to_string(0, a))
            assert m and int(m.group(1)) != picked


# ---- S-CLOCKSTEP loader checks ----


def vp_doc() -> dict[str, Any]:
    d = C.dump(paused_vp())
    assert d["phase"] == "VP"
    d.pop("supply", None)
    return d


def vd_doc() -> dict[str, Any]:
    s = paused_vp()
    s.apply_action(pick(5))
    d = C.dump(s)
    assert d["phase"] == "VD"
    d.pop("supply", None)
    return d


def code_of(doc: dict[str, Any]) -> str:
    with pytest.raises(H.scenario_module().ScenarioError) as e:
        H.load_doc_state(doc)
    return str(e.value.code)


def test_S_CLOCKSTEP_valid_vp_and_vd_states_load() -> None:
    """S-CLOCKSTEP: a state the Clock step could leave behind (clock house, houses s2/s6/s10
    occupied, waiting passengers only at nodes without an empty house) loads in VP and in VD
    (the picked passenger at n5 does not trip check (a))."""
    assert H.load_doc_state(vp_doc()).phase == "VP"
    assert H.load_doc_state(vd_doc()).phase == "VD"


def test_S_CLOCKSTEP_a_rejects_waiting_passenger_beside_empty_building_of_clock_type() -> None:
    """S-CLOCKSTEP (a): in VP clock house, empty house s0 at n10; a waiting passenger at n10 is
    impossible after the step -> rejected."""
    d = vp_doc()
    w = dict(d["waiting"])
    w["10"] = w.get("10", 0) + 1
    d["waiting"] = w
    assert code_of(d) == "S-CLOCKSTEP"


def test_S_CLOCKSTEP_a_also_in_vd() -> None:
    """S-CLOCKSTEP (a) applies in VD as well."""
    d = vd_doc()
    w = dict(d["waiting"])
    w["10"] = w.get("10", 0) + 1
    d["waiting"] = w
    assert code_of(d) == "S-CLOCKSTEP"


def test_S_CLOCKSTEP_b_rejects_occupied_building_of_another_type() -> None:
    """S-CLOCKSTEP (b): office s8 occupied while the clock reads house -> rejected."""
    d = vp_doc()
    for b in d["buildings"]:
        if b["site"] == 8:
            b["occupied"] = True
    assert code_of(d) == "S-CLOCKSTEP"


def test_S_CLOCKSTEP_b_also_in_vd() -> None:
    """S-CLOCKSTEP (b) applies in VD as well."""
    d = vd_doc()
    for b in d["buildings"]:
        if b["site"] == 8:
            b["occupied"] = True
    assert code_of(d) == "S-CLOCKSTEP"


def test_S_CLOCKSTEP_not_checked_outside_vp_vd() -> None:
    """SCENARIOS S-CLOCKSTEP scope: the same breaking edits are accepted in SEL (fixtures with
    waiting passengers beside empty buildings, and occupied buildings of any type, exist there)."""
    d = C.setup_doc()
    C.set_building(d, 8, "office", occupied=True)
    C.set_waiting(d, {13: 1, 17: 1, 18: 1, 10: 1})
    assert H.load_doc_state(d).phase == "SEL"
