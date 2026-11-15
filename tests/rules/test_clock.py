"""Clock action and passenger movement (R-050..R-055; ACTIONS 3.9, 4).

Written ahead of Phase 1 unit 10; tests whose rule had not landed carry a pending marker, which the
implementer removes as each rule lands.

Fixtures under ``tests/scenarios/clock/`` (4 players unless noted, seat 3's Clock marker, phase CK):
  ck_movement          clock house, no occupied building.  Buildings: n13 {s2 pub, s3 house},
                       n17 {s4 office, s5 pub}, n18 {s6 house, s7 office}, n21 {s9 pub, s10 pub},
                       n10 {s0 house}, n14 {s1 office}.  Waiting: n13 1, n17 1, n18 1, n21 3,
                       station n5 2, station n25 1.  5 stones on the clock.
  ck_occupied_pub      clock pub; s2, s5, s9 (pubs) and s6 (house) occupied; 1 waiting at n21.
  ck_last_stone_4p/3p  one stone left, a Vrrooomm! marker (seat 0) and Starting Player marker
                       (seat 1) still on the board, s5 occupied, waiting n18 1 and n21 2,
                       start_player 2, seat 0 has 2 points.
  resolve_ck           the shared CK fixture: roundabout passengers (n13, n17, n18, n21) at the
                       first Clock, buildings s0..s7 empty.

Accessors assumed beyond the harness: state.stones_on_clock, state.clock, state.seat(p).stones,
state.is_occupied(site), state.waiting_at(node), state.passenger_supply, state.end_types,
state.site_building(t).  Round / start player / points are read through ``dump_scenario``.

Passenger-movement expectations are worked out by hand from the building lists above.
"""

from __future__ import annotations

from typing import Any

import pytest

from bus.actions import CLOCK_ADVANCE, CLOCK_PAUSE
from bus.board import load_board
from tests import harness as H

B = load_board()
N = len(B.nodes)
T = len(B.sites)
TYPES = ["house", "office", "pub"]
NEXT = {"house": "office", "office": "pub", "pub": "house"}
PASSENGER_TOTAL = 15
OWNER = 3


def ck_state(name: str = "clock/ck_movement", **over: Any) -> Any:
    doc = H.load_doc(name)
    doc.update(over)
    return H.load_doc_state(doc)


def occupied(s: Any) -> set[int]:
    return {t for t in range(T) if s.is_occupied(t)}


def waiting(s: Any) -> dict[int, int]:
    return {n: s.waiting_at(n) for n in range(N) if s.waiting_at(n)}


def total_passengers(s: Any) -> int:
    return len(occupied(s)) + sum(s.waiting_at(n) for n in range(N)) + s.passenger_supply


def dump(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s)


# ---- R-050: legal actions ---------------------------------------------------------------------


@pytest.mark.parametrize("clock", TYPES)
def test_R050_legal_actions_are_exactly_advance_and_pause(clock: str) -> None:
    """R-050: the owner of the resolving Clock marker chooses advance or pause; nothing else is
    legal, whatever the clock type."""
    s = ck_state(clock=clock)
    assert s.phase == "CK" and s.current_player() == OWNER
    assert sorted(s.legal_actions()) == sorted([CLOCK_ADVANCE, CLOCK_PAUSE])


def test_R050_pause_is_legal_with_one_stone_left() -> None:
    """R-050/R-052: pause stays legal when only one stone remains (a stone is always on the
    clock at a CK node)."""
    s = ck_state("clock/ck_last_stone_4p")
    assert s.stones_on_clock == 1
    assert sorted(s.legal_actions()) == sorted([CLOCK_ADVANCE, CLOCK_PAUSE])


def test_R050_three_player_game_same_choices_owner_acts() -> None:
    """R-050, R-006: in a 3-player game the owner (seat 2) has the same two choices."""
    s = ck_state("clock/ck_last_stone_3p")
    assert s.current_player() == 2
    assert sorted(s.legal_actions()) == sorted([CLOCK_ADVANCE, CLOCK_PAUSE])


def test_R050_other_actions_not_legal() -> None:
    """R-050: ids that are not Clock choices (select 0, drop-out, station placement) are not
    legal at CK."""
    s = ck_state()
    legal = set(s.legal_actions())
    assert len(legal) == 2
    assert all(a not in legal for a in range(CLOCK_ADVANCE))


@pytest.mark.parametrize("choice", [CLOCK_ADVANCE, CLOCK_PAUSE])
def test_R050_choice_leaves_ck_phase(choice: int) -> None:
    """R-050: the Clock marker is a single decision; after it the phase is no longer CK for this
    marker (the fixture has no other Clock marker)."""
    s = ck_state("resolve_ck")
    s.apply_action(choice)
    assert s.phase != "CK"


# ---- R-051: advance ---------------------------------------------------------------------------


@pytest.mark.parametrize("clock", TYPES)
def test_R051_advance_moves_to_next_type(clock: str) -> None:
    """R-051: house -> office -> pub -> house."""
    s = ck_state(clock=clock)
    s.apply_action(CLOCK_ADVANCE)
    assert s.clock == NEXT[clock]


@pytest.mark.parametrize("clock", TYPES)
def test_R051_advance_takes_no_stone(clock: str) -> None:
    """R-051/R-052: advance leaves the stones where they are."""
    s = ck_state(clock=clock)
    s.apply_action(CLOCK_ADVANCE)
    assert s.stones_on_clock == 5
    assert all(s.seat(p).stones == 0 for p in range(4))
    assert not s.is_terminal()


def test_R051_advance_with_last_stone_does_not_end_the_game() -> None:
    """R-051/R-053: only a pause can take the last stone; advance with one stone left continues
    (the stone stays on the clock)."""
    s = ck_state("clock/ck_last_stone_4p")
    s.apply_action(CLOCK_ADVANCE)
    assert not s.is_terminal()
    assert s.stones_on_clock == 1
    assert "last_stone" not in list(s.end_types)
    assert s.clock == "office"


# ---- R-052: pause -----------------------------------------------------------------------------


@pytest.mark.parametrize("clock", TYPES)
def test_R052_pause_keeps_clock_type(clock: str) -> None:
    """R-052: pause leaves the clock hand unchanged."""
    s = ck_state(clock=clock)
    s.apply_action(CLOCK_PAUSE)
    assert s.clock == clock


def test_R052_pause_moves_one_stone_to_the_owner() -> None:
    """R-052: 5 stones on the clock; after a pause by seat 3 it holds 1 and 4 remain; other
    seats gain nothing."""
    s = ck_state()
    s.apply_action(CLOCK_PAUSE)
    assert s.stones_on_clock == 4
    assert [s.seat(p).stones for p in range(4)] == [0, 0, 0, 1]
    assert not s.is_terminal()


def test_R052_stones_accumulate_with_the_holder() -> None:
    """R-052: a stone once taken stays; a second pause by a seat that already holds one makes
    it two (2 left on the clock)."""
    s = ck_state(stones_on_clock=3, seats=_with_stones([0, 0, 0, 2]))
    s.apply_action(CLOCK_PAUSE)
    assert s.stones_on_clock == 2
    assert s.seat(3).stones == 3
    assert not s.is_terminal()


def _with_stones(held: list[int]) -> list[dict[str, Any]]:
    seats = H.load_doc("clock/ck_movement")["seats"]
    for i, h in enumerate(held):
        seats[i]["stones"] = h
    return seats


def test_R052_three_player_game_starts_with_four_stones() -> None:
    """R-052, R-006: in a 3-player game 4 stones, so after a pause 3 remain (spec example).
    Uses the 3-player fixture with its stones reset: seat 2 holds none, 4 on the clock."""
    doc = H.load_doc("clock/ck_last_stone_3p")
    doc["stones_on_clock"] = 4
    for sd in doc["seats"]:
        sd["stones"] = 0
    s = H.load_doc_state(doc)
    s.apply_action(CLOCK_PAUSE)
    assert s.stones_on_clock == 3
    assert [s.seat(p).stones for p in range(3)] == [0, 0, 1]


def test_R052_pause_does_not_change_points() -> None:
    """R-052/R-008: a stone is not a point; the score discs do not move on a pause."""
    doc = H.load_doc("clock/ck_last_stone_4p")
    doc["stones_on_clock"] = 2
    doc["seats"][1]["stones"] = 1
    s = H.load_doc_state(doc)
    before = [sd.get("points", 0) for sd in dump(s)["seats"]]
    s.apply_action(CLOCK_PAUSE)
    after = [sd.get("points", 0) for sd in dump(s)["seats"]]
    assert before == after


# ---- R-053: last stone ------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["clock/ck_last_stone_4p", "clock/ck_last_stone_3p"])
def test_R053_last_stone_pause_is_terminal(fixture: str) -> None:
    """R-053, R-006: taking the last stone (4 players: 5th stone; 3 players: 4th) ends the game
    immediately; end type is exactly last_stone; ACTIONS 3.9: terminal, no legal action."""
    s = ck_state(fixture)
    owner = s.current_player()
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal()
    assert s.phase == "END"
    assert list(s.end_types) == ["last_stone"]
    assert s.legal_actions() == []
    assert s.stones_on_clock == 0
    assert s.seat(owner).stones >= 1
    assert sum(s.seat(p).stones for p in range(s.num_players())) == (
        4 if fixture.endswith("3p") else 5
    )


def test_R053_pausing_player_keeps_the_stone_4p() -> None:
    """R-053: the pausing player (seat 3, who held none) keeps the last stone; others unchanged."""
    s = ck_state("clock/ck_last_stone_4p")
    s.apply_action(CLOCK_PAUSE)
    assert [s.seat(p).stones for p in range(4)] == [2, 2, 0, 1]


def test_R053_pausing_player_keeps_the_stone_3p() -> None:
    """R-053: 3 players, holders 1,1,1 + seat 2 pauses: seat 2 holds 2, total 4 (R-006)."""
    s = ck_state("clock/ck_last_stone_3p")
    s.apply_action(CLOCK_PAUSE)
    assert [s.seat(p).stones for p in range(3)] == [1, 1, 2]


def test_R053_no_passenger_movement_on_the_final_pause() -> None:
    """R-053: no passenger movement: the occupied building stays occupied, waiting passengers
    do not enter even though a building of the clock's type exists at their node."""
    s = ck_state("clock/ck_last_stone_4p", clock="pub")
    # s9/s10 are pubs at n21 where 2 passengers wait; s5 is a pub at n17, occupied.
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal()
    assert occupied(s) == {5}
    assert waiting(s) == {18: 1, 21: 2}
    assert s.passenger_supply == 11


def test_R053_pending_vrrooomm_does_not_resolve() -> None:
    """R-053: a Vrrooomm! marker later in the round does not resolve: terminal, not phase VP,
    no moves; the Starting Player marker does not change the start player; round unchanged."""
    s = ck_state("clock/ck_last_stone_4p")
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal() and s.phase == "END"
    d = dump(s)
    assert d["round"] == 1
    assert d["start_player"] == 2
    assert d["seats"][0].get("points", 0) == 2


@pytest.mark.parametrize("fixture", ["clock/ck_last_stone_4p", "clock/ck_last_stone_3p"])
def test_R053_last_stone_end_dump_load_round_trip(fixture: str) -> None:
    """R-053, SCENARIOS 6 S-BOARD / 9.9: the position after a last-stone pause (markers still on
    the board) dumps and reloads: terminal, end_types ["last_stone"], same action board, and
    dump(load(dump)) == dump."""
    s = ck_state(fixture)
    s.apply_action(CLOCK_PAUSE)
    d1 = dump(s)
    s2 = H.load_doc_state(d1)
    assert s2.is_terminal()
    assert list(s2.end_types) == ["last_stone"]
    d2 = dump(s2)
    assert d2["action_board"] == d1["action_board"]
    assert d2["action_board"] == H.load_doc(fixture)["action_board"]
    assert d2 == d1


def test_R053_stones_on_clock_zero_only_at_end() -> None:
    """R-053 (S-STONES): the clock holds zero stones only in END; before the pause it holds 1."""
    s = ck_state("clock/ck_last_stone_4p")
    assert s.stones_on_clock == 1 and not s.is_terminal()
    s.apply_action(CLOCK_PAUSE)
    assert s.stones_on_clock == 0


def test_R053_terminal_returns_are_defined() -> None:
    """R-053 with R-071 (unit 12): terminal returns exist, one per seat."""
    s = ck_state("clock/ck_last_stone_4p")
    s.apply_action(CLOCK_PAUSE)
    assert len(s.returns()) == 4


def test_R053_not_last_stone_continues() -> None:
    """R-053 (negative): a pause that leaves a stone on the clock does not end the game."""
    doc = H.load_doc("clock/ck_last_stone_4p")
    doc["stones_on_clock"] = 2
    doc["seats"][1]["stones"] = 1
    s = H.load_doc_state(doc)
    s.apply_action(CLOCK_PAUSE)
    assert not s.is_terminal()
    assert s.stones_on_clock == 1
    assert list(s.end_types) == []


# ---- R-054 / R-055: passenger movement --------------------------------------------------------


EXPECT_FROM_EMPTY = {
    # (clock before, choice) -> (clock type used for entering, occupied sites, waiting)
    ("house", CLOCK_ADVANCE): ({4, 7}, {13: 1, 21: 3, 5: 2, 25: 1}),
    ("house", CLOCK_PAUSE): ({3, 6}, {17: 1, 21: 3, 5: 2, 25: 1}),
    ("office", CLOCK_ADVANCE): ({2, 5, 9, 10}, {18: 1, 21: 1, 5: 2, 25: 1}),
    ("office", CLOCK_PAUSE): ({4, 7}, {13: 1, 21: 3, 5: 2, 25: 1}),
    ("pub", CLOCK_ADVANCE): ({3, 6}, {17: 1, 21: 3, 5: 2, 25: 1}),
    ("pub", CLOCK_PAUSE): ({2, 5, 9, 10}, {18: 1, 21: 1, 5: 2, 25: 1}),
}


@pytest.mark.parametrize(("clock", "choice"), list(EXPECT_FROM_EMPTY))
def test_R055_step_in_uses_the_clocks_type_after_the_choice(clock: str, choice: int) -> None:
    """R-055: after advance the NEW type, after pause the SAME type; at each node
    min(waiting, buildings of that type) passengers enter; the rest wait there."""
    s = ck_state(clock=clock)
    s.apply_action(choice)
    occ, wait = EXPECT_FROM_EMPTY[(clock, choice)]
    assert occupied(s) == occ
    assert waiting(s) == wait
    assert total_passengers(s) == PASSENGER_TOTAL


def test_R055_one_passenger_per_building_leftovers_wait() -> None:
    """R-055: pub clock after advance, n21 has 3 waiting and 2 pubs: both fill, one waits."""
    s = ck_state(clock="office")
    s.apply_action(CLOCK_ADVANCE)
    assert occupied(s) >= {9, 10}
    assert s.waiting_at(21) == 1


def test_R055_fewer_passengers_than_buildings_fills_ascending_site_id() -> None:
    """R-055: n21 has pubs s9 and s10, one passenger waits: the lower site id (s9) is filled."""
    s = ck_state(clock="office", waiting={"21": 1})
    s.apply_action(CLOCK_ADVANCE)
    assert occupied(s) == {9}
    assert s.waiting_at(21) == 0


def test_R055_no_building_of_the_type_at_the_node_nobody_enters() -> None:
    """R-055 (negative): n17 has no house; clock house with a passenger there: it stays waiting
    and no building elsewhere is filled by it."""
    s = ck_state(clock="house", waiting={"17": 1})
    s.apply_action(CLOCK_PAUSE)
    assert occupied(s) == set()
    assert waiting(s) == {17: 1}


def test_R055_passengers_do_not_cross_nodes() -> None:
    """R-055: entering is per node; 3 waiting at n13 and none elsewhere, clock office: n13 has
    no office, so nobody enters even though offices exist at n17, n18, n14."""
    s = ck_state(clock="house", waiting={"13": 3})
    s.apply_action(CLOCK_ADVANCE)
    assert occupied(s) == set()
    assert waiting(s) == {13: 3}


def test_R055_node_with_two_types_fills_only_the_clocks_type() -> None:
    """R-055: n13 has a pub (s2) and a house (s3), 2 passengers: clock house -> only s3 fills,
    one passenger waits."""
    s = ck_state(clock="house", waiting={"13": 2})
    s.apply_action(CLOCK_PAUSE)
    assert occupied(s) == {3}
    assert s.waiting_at(13) == 1


def test_R054_step_out_happens_for_every_building_type_then_step_in_new_type() -> None:
    """R-054, R-055: pub clock, advance to house.  s2 (pub, n13), s5 (pub, n17), s9 (pub, n21)
    and s6 (house, n18) are occupied.  All step out; then s3 (house, n13) and s6 (n18) fill.
    n17 (no house) keeps 1 waiting, n21 keeps 2."""
    s = ck_state("clock/ck_occupied_pub")
    assert occupied(s) == {2, 5, 6, 9}
    s.apply_action(CLOCK_ADVANCE)
    assert s.clock == "house"
    assert occupied(s) == {3, 6}
    assert waiting(s) == {17: 1, 21: 2}
    assert total_passengers(s) == PASSENGER_TOTAL


def test_R054_passenger_stepping_out_can_re_enter_a_building_of_the_new_type() -> None:
    """R-054, R-055: the passenger in house s6 steps out and steps back into s6 when the new
    clock type is house (same node, same building)."""
    s = ck_state("clock/ck_occupied_pub")
    s.apply_action(CLOCK_ADVANCE)
    assert s.is_occupied(6)
    assert s.waiting_at(18) == 0


def test_R055_pause_same_type_buildings_refill_and_freed_passengers_help_others() -> None:
    """R-055 (spec example): after a pause, passengers in pubs step out and re-enter pubs at
    the same node.  Pub clock, pause: s2, s5 re-fill; n21's occupant steps out and with the
    waiting one fills s9 and s10; the house passenger at n18 steps out and waits."""
    s = ck_state("clock/ck_occupied_pub")
    s.apply_action(CLOCK_PAUSE)
    assert s.clock == "pub"
    assert occupied(s) == {2, 5, 9, 10}
    assert waiting(s) == {18: 1}
    assert s.stones_on_clock == 4 and s.seat(OWNER).stones == 1
    assert total_passengers(s) == PASSENGER_TOTAL


def test_R054_step_out_does_not_touch_the_supply() -> None:
    """R-054/R-005: movement shuffles passengers between buildings and nodes; the supply is
    unchanged."""
    s = ck_state("clock/ck_occupied_pub")
    before = s.passenger_supply
    s.apply_action(CLOCK_ADVANCE)
    assert s.passenger_supply == before


def test_R054_occupied_building_of_other_type_empties_when_nobody_can_enter() -> None:
    """R-054 (boundary): the lone passenger in a house steps out; clock goes to office; n18 has
    an office (s7) so it enters s7, leaving s6 empty."""
    doc = H.load_doc("clock/ck_movement")
    doc["clock"] = "house"
    doc["waiting"] = {}
    for b in doc["buildings"]:
        if b["site"] == 6:
            b["occupied"] = True
    s = H.load_doc_state(doc)
    s.apply_action(CLOCK_ADVANCE)
    assert occupied(s) == {7}
    assert s.waiting_at(18) == 0


def test_R054_no_matching_building_leaves_passenger_waiting() -> None:
    """R-054, R-055: n10 has only a house (s0).  Passenger in s0, clock house -> office: it
    steps out and waits at n10 (no office there)."""
    doc = H.load_doc("clock/ck_movement")
    doc["waiting"] = {}
    for b in doc["buildings"]:
        if b["site"] == 0:
            b["occupied"] = True
    s = H.load_doc_state(doc)
    s.apply_action(CLOCK_ADVANCE)
    assert occupied(s) == set()
    assert waiting(s) == {10: 1}


# ---- R-015, R-055: roundabouts and stations ---------------------------------------------------


def test_R015_roundabout_passengers_enter_at_the_clock_step() -> None:
    """R-015: before the round-1 Clock step the four roundabout passengers wait (none in buildings);
    advance to office puts them in the offices at n17 (s4) and n18 (s7); n13 and n21 have no
    office so those wait."""
    s = H.load_fixture("resolve_ck")
    assert occupied(s) == set()
    assert set(waiting(s)) == set(B.roundabouts)
    s.apply_action(CLOCK_ADVANCE)
    assert occupied(s) == {4, 7}
    assert waiting(s) == {13: 1, 21: 1}


def test_R015_roundabout_passenger_enters_after_pause_with_same_type() -> None:
    """R-015, R-055: pause at house: n13's passenger enters house s3, n18's enters house s6."""
    s = H.load_fixture("resolve_ck")
    s.apply_action(CLOCK_PAUSE)
    assert occupied(s) == {3, 6}
    assert waiting(s) == {17: 1, 21: 1}


@pytest.mark.parametrize("clock", TYPES)
@pytest.mark.parametrize("choice", [CLOCK_ADVANCE, CLOCK_PAUSE])
def test_R055_stations_never_fill(clock: str, choice: int) -> None:
    """R-012/R-046/R-055: stations have no sites; their waiting passengers (n5: 2, n25: 1) stay
    waiting whatever the clock does."""
    s = ck_state(clock=clock)
    for st in B.stations:
        assert not B.node_sites[st]
    s.apply_action(choice)
    assert s.waiting_at(B.stations[0]) == 2
    assert s.waiting_at(B.stations[1]) == 1


# ---- conservation, strings, hand-off ----------------------------------------------------------


@pytest.mark.parametrize("fixture", ["clock/ck_movement", "clock/ck_occupied_pub", "resolve_ck"])
@pytest.mark.parametrize("choice", [CLOCK_ADVANCE, CLOCK_PAUSE])
def test_R055_passengers_are_conserved(fixture: str, choice: int) -> None:
    """R-005/R-055: on-board + supply stays 15 and the supply does not change."""
    s = H.load_fixture(fixture)
    supply = s.passenger_supply
    s.apply_action(choice)
    assert s.passenger_supply == supply
    assert total_passengers(s) == PASSENGER_TOTAL


def test_R050_action_to_string() -> None:
    """ACTIONS 4: Clock choices print as 'P3: clock advance' / 'P3: clock pause'."""
    s = ck_state()
    assert s.action_to_string(3, CLOCK_ADVANCE) == "P3: clock advance"
    assert s.action_to_string(3, CLOCK_PAUSE) == "P3: clock pause"


@pytest.mark.parametrize("choice", [CLOCK_ADVANCE, CLOCK_PAUSE])
def test_R056_movement_is_done_before_vrrooomm_resolves(choice: int) -> None:
    """R-050 then R-056 (resolution order): with a non-final choice and a Vrrooomm! marker still
    on the board, the state after the choice is no longer CK, and the passengers have already
    moved when Vrrooomm! starts (the movement result is visible)."""
    doc = H.load_doc("clock/ck_last_stone_4p")
    doc["clock"] = "house"
    doc["stones_on_clock"] = 2  # so that a pause is not the last stone
    doc["seats"][1]["stones"] = 1
    s = H.load_doc_state(doc)
    s.apply_action(choice)
    assert not s.is_terminal()
    assert s.phase != "CK"
    if choice == CLOCK_PAUSE:
        # house clock: s5 (pub) passenger stepped out and waits at n17 (no house there)
        assert not s.is_occupied(5)
