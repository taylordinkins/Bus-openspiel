"""Phase 1 action selection (R-020, R-022..R-028; ACTIONS §2, §3.4, §4; SCENARIOS §3.4, §4.1).

Written ahead of the SEL code (Phase 1 unit 6); everything here is pending.

Public API assumed (no engine internals):
  state.round / .phase / .current_player() / .start_player / .legal_actions() / .apply_action(a)
  state.action_board        7 lists (resolution order) of the seats on letters a, b, ... in order
  state.seat(p).action_markers   Action markers in hand (a placed marker leaves the hand, R-024)
  state.action_to_string(player, action)
  bus.actions.SELECT_BASE (0) and DROP_OUT (7)

Positions come from the SEL fixtures (sel_round1, sel_pass1_mid, sel_pass2_dropped) or from a
4-player document derived from sel_round1 by ``doc4`` (board, pass, dropped, markers, ...).
3- and 5-player games are reached by playing a random setup (units 4-5).
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from bus.actions import DROP_OUT, SELECT_BASE
from tests import harness as H

NAMES = ["lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player"]
SPACES = [6, 1, 6, 6, 1, 6, 1]
LINES, BUSES, PASSENGERS, BUILDINGS, CLOCK, VRROOOMM, STARTING = range(7)
LABEL = ["Lines", "Buses", "Passengers", "Buildings", "Clock", "Vrrooomm!", "Starting Player"]
SEL_ALL = [SELECT_BASE + a for a in range(7)]
RESOLUTION_PHASES = ("LN", "PX", "BD", "CK", "VP", "VD")


# ---- helpers ---------------------------------------------------------------------------------


def order(n: int, start: int) -> list[int]:
    return [(start + i) % n for i in range(n)]


def doc4(
    board: dict[int, list[int]] | None = None,
    pass_: int = 0,
    dropped: tuple[int, ...] = (),
    markers: dict[int, int] | None = None,
    buses: dict[int, int] | None = None,
    waiting: dict[str, int] | None = None,
) -> dict[str, Any]:
    """4-player SEL document (start player 0) derived from sel_round1.

    ``board`` maps an action index to the seats on its letters in order.  Markers in hand default
    to 20 minus the markers the seat has on the board; ``markers`` overrides per seat."""
    d = H.load_doc("sel_round1")
    d.pop("expect", None)
    board = board or {}
    placed = [0, 0, 0, 0]
    ab: dict[str, list[int | None]] = {}
    for a, name in enumerate(NAMES):
        seats = board.get(a, [])
        assert len(seats) <= SPACES[a]
        for s in seats:
            placed[s] += 1
        ab[name] = list(seats) + [None] * (SPACES[a] - len(seats))
    d["action_board"] = ab
    for p in range(4):
        d["seats"][p]["markers_left"] = 20 - placed[p]
        if markers and p in markers:
            d["seats"][p]["markers_left"] = markers[p]
        if buses and p in buses:
            d["seats"][p]["buses"] = buses[p]
    if waiting is not None:
        d["waiting"] = waiting
    d["pending"] = {"pass": pass_, "dropped": list(dropped)}
    return d


def play_to_sel(n: int, start: int = 0, seed: int = 0) -> Any:
    doc: dict[str, Any] = {"format": 1, "parameters": {"players": n}}
    if start:
        doc["start_player"] = start
    state = H.load_doc_state(doc)
    rng = random.Random(seed)
    while state.phase in ("SB", "L1", "L2"):
        state.apply_action(rng.choice(state.legal_actions()))
    assert state.phase == "SEL" and state.round == 1
    return state


def step(state: Any, actor: int, action: int) -> None:
    """Assert ``actor`` is to move and ``action`` is legal, then apply it."""
    assert state.phase == "SEL"
    assert state.current_player() == actor
    assert action in state.legal_actions()
    state.apply_action(action)


def lowest_select(state: Any) -> int:
    return min(a for a in state.legal_actions() if a != DROP_OUT)


def board_of(state: Any) -> list[list[int]]:
    return [list(x) for x in state.action_board]


def left_sel(state: Any) -> bool:
    return state.phase != "SEL"


# ---- R-020 / R-021 / R-022 -------------------------------------------------------------------


def test_R020_round_1_starts_in_selection_phase() -> None:
    """R-020/R-021: Phase 1 (selection) is the first thing in round 1, after setup."""
    s = H.load_fixture("sel_round1")
    assert s.round == 1 and s.phase == "SEL"
    assert s.current_player() == s.start_player == 0


def test_R022_board_shape_and_empty_start() -> None:
    """R-022: 7 actions in resolution order; board empty at round start; first node offers all 7."""
    s = H.load_fixture("sel_round1")
    assert len(s.action_board) == 7
    assert all(len(x) == 0 for x in s.action_board)
    assert s.legal_actions() == SEL_ALL
    assert DROP_OUT not in s.legal_actions()


@pytest.mark.parametrize("a", range(7))
def test_R022_action_to_string_names(a: int) -> None:
    """ACTIONS §4: SELECT is rendered 'P<seat>: select <Name>'; DROP_OUT 'P<seat>: drop out'."""
    s = H.load_fixture("sel_round1")
    assert s.action_to_string(0, SELECT_BASE + a) == f"P0: select {LABEL[a]}"
    assert s.action_to_string(2, DROP_OUT) == "P2: drop out"


# ---- R-023 -----------------------------------------------------------------------------------


def test_R023_no_drop_out_in_pass_0() -> None:
    """R-023: pass 0 is mandatory; each node offers exactly the 7 SELECTs."""
    s = H.load_fixture("sel_round1")
    for p in range(4):
        assert s.current_player() == p
        legal = s.legal_actions()
        assert DROP_OUT not in legal
        board = board_of(s)
        free = [SELECT_BASE + a for a in range(7) if len(board[a]) < SPACES[a]]
        assert legal == free
        s.apply_action(SELECT_BASE + (p % 7))


def test_R023_no_drop_out_in_pass_1_fixture() -> None:
    """R-023: at a pass-1 node (sel_pass1_mid, seat 2): no DROP_OUT, all 7 SELECTs."""
    s = H.load_fixture("sel_pass1_mid")
    assert s.current_player() == 2
    assert s.legal_actions() == SEL_ALL


def test_R023_no_drop_out_through_whole_second_pass() -> None:
    """R-023: DROP_OUT is illegal for seats 2 and 3 of pass 1, then legal at seat 0, pass 2."""
    s = H.load_fixture("sel_pass1_mid")
    for p in (2, 3):
        assert s.current_player() == p
        assert DROP_OUT not in s.legal_actions()
        s.apply_action(SELECT_BASE + BUILDINGS)
    assert s.current_player() == 0
    assert DROP_OUT in s.legal_actions()


def test_R023_same_action_in_both_mandatory_passes() -> None:
    """R-023: a seat may choose the same action in both mandatory passes."""
    s = H.load_fixture("sel_round1")
    for p in range(4):
        step(s, p, SELECT_BASE + (BUSES if p == 1 else CLOCK if p == 2 else LINES))
    # pass 1: seat 0 picks Lines again; seat 1 Passengers; seat 2 Clock is full, Buses is full
    step(s, 0, SELECT_BASE + LINES)
    assert board_of(s)[LINES] == [0, 3, 0]


def test_R023_pass_2_offers_drop_out() -> None:
    """R-023/R-024: once both mandatory passes are done DROP_OUT joins the SELECTs."""
    s = H.load_fixture("sel_round1")
    for i in range(8):
        step(s, i % 4, lowest_select(s))
    assert s.phase == "SEL" and s.current_player() == 0
    assert DROP_OUT in s.legal_actions()


# ---- turn order ------------------------------------------------------------------------------


@pytest.mark.parametrize("n,start", [(3, 0), (3, 2), (4, 1), (5, 0), (5, 3)])
def test_R023_turn_order_from_start_player_each_pass(n: int, start: int) -> None:
    """R-023/R-024: each pass visits the seats in turn order from the start player."""
    s = play_to_sel(n, start)
    sp = s.start_player
    assert sp == start
    seq = order(n, sp)
    for _ in range(2):
        for p in seq:
            step(s, p, lowest_select(s))
    assert s.current_player() == sp
    assert DROP_OUT in s.legal_actions()


@pytest.mark.parametrize("n,start", [(3, 1), (5, 4)])
def test_R024_third_pass_from_start_player_with_dropouts(n: int, start: int) -> None:
    """R-024: later passes keep the turn order from the start player and skip dropped seats."""
    s = play_to_sel(n, start)
    seq = order(n, s.start_player)
    for _ in range(2):
        for p in seq:
            step(s, p, lowest_select(s))
    step(s, seq[0], DROP_OUT)
    for p in seq[1:]:
        step(s, p, lowest_select(s))
    # pass 3: seq[0] skipped, the others in order
    assert s.current_player() == seq[1]


# ---- R-024 ----------------------------------------------------------------------------------


def test_R024_rulebook_example_three_players() -> None:
    """R-024 example (3 players, start P0): P0 drops and P2 drops on pass 3, P1 on pass 4."""
    s = play_to_sel(3, 0)
    for p, a in [(0, 0), (1, 1), (2, 2), (0, 0), (1, 3), (2, 4)]:
        step(s, p, SELECT_BASE + a)
    step(s, 0, DROP_OUT)
    step(s, 1, SELECT_BASE + VRROOOMM)
    step(s, 2, DROP_OUT)
    assert s.phase == "SEL"
    assert s.current_player() == 1  # P0 and P2 dropped; pass 4 is P1 alone
    step(s, 1, DROP_OUT)
    assert left_sel(s)


def test_R024_drop_out_fixture_skips_dropped_seat() -> None:
    """R-024: sel_pass2_dropped (seat 3 dropped): seats 0,1,2 act, seat 3 never gets a node."""
    s = H.load_fixture("sel_pass2_dropped")
    assert s.current_player() == 0
    step(s, 0, SELECT_BASE + BUILDINGS)
    step(s, 1, SELECT_BASE + BUILDINGS)
    step(s, 2, SELECT_BASE + BUILDINGS)
    assert s.phase == "SEL"
    assert s.current_player() == 0  # next pass, seat 3 skipped


def test_R024_dropped_seat_stays_out_for_rest_of_round() -> None:
    """R-024: a seat that drops is skipped in every later pass of the round."""
    s = H.load_fixture("sel_pass2_dropped")
    step(s, 0, SELECT_BASE + BUILDINGS)
    step(s, 1, DROP_OUT)
    step(s, 2, SELECT_BASE + BUILDINGS)
    step(s, 0, SELECT_BASE + BUILDINGS)  # next pass: seat 1 and 3 skipped
    step(s, 2, DROP_OUT)
    assert s.phase == "SEL"
    assert s.current_player() == 0  # seats 1, 2 and 3 are out; only seat 0 remains
    step(s, 0, SELECT_BASE + BUILDINGS)
    assert s.current_player() == 0


def test_R024_drop_out_does_not_spend_a_marker() -> None:
    """R-024: dropping out places nothing: markers in hand and the board are unchanged."""
    s = H.load_fixture("sel_pass2_dropped")
    before_hand = [s.seat(p).action_markers for p in range(4)]
    before_board = board_of(s)
    step(s, 0, DROP_OUT)
    assert [s.seat(p).action_markers for p in range(4)] == before_hand
    assert board_of(s) == before_board


def test_R024_each_select_spends_one_marker() -> None:
    """R-024: every SELECT moves one Action marker from hand to board."""
    s = H.load_fixture("sel_round1")
    hand = [20, 20, 20, 20]
    assert [s.seat(p).action_markers for p in range(4)] == hand
    for i in range(8):
        p = i % 4
        step(s, p, lowest_select(s))
        hand[p] -= 1
        assert [s.seat(q).action_markers for q in range(4)] == hand
    on_board = sum(len(x) for x in s.action_board)
    assert on_board == 8 == 80 - sum(hand)


def test_R024_drop_out_rejected_in_mandatory_pass() -> None:
    """R-023/R-024: applying DROP_OUT in pass 0 is not allowed."""
    s = H.load_fixture("sel_round1")
    with pytest.raises(Exception):  # noqa: B017  (OpenSpiel raises its own error type)
        s.apply_action(DROP_OUT)


# ---- R-025 ----------------------------------------------------------------------------------


def test_R025_earliest_free_letter_one_seat_per_letter() -> None:
    """R-025: successive markers on Lines take letters a, b, c, d in order (seats 0..3)."""
    s = H.load_fixture("sel_round1")
    for p in range(4):
        step(s, p, SELECT_BASE + LINES)
        assert board_of(s)[LINES] == list(range(p + 1))


def test_R025_one_seat_on_several_letters() -> None:
    """R-025: one seat's two markers on an action occupy a and b (the board lists seat 0 twice)."""
    s = H.load_fixture("sel_round1")
    step(s, 0, SELECT_BASE + PASSENGERS)
    step(s, 1, SELECT_BASE + BUILDINGS)
    step(s, 2, SELECT_BASE + BUILDINGS)
    step(s, 3, SELECT_BASE + BUILDINGS)
    step(s, 0, SELECT_BASE + PASSENGERS)
    assert board_of(s)[PASSENGERS] == [0, 0]
    assert board_of(s)[BUILDINGS] == [1, 2, 3]


def test_R025_next_marker_goes_after_existing_letters() -> None:
    """R-025 example: Lines has a and b occupied; the next goes on c."""
    d = doc4({LINES: [0, 1]}, pass_=0)
    # seats 0,1 have placed once, seat 2 acts: pass 0
    s = H.load_doc_state(d)
    assert s.current_player() == 2
    s.apply_action(SELECT_BASE + LINES)
    assert board_of(s)[LINES] == [0, 1, 2]


def test_R025_player_chooses_action_not_letter() -> None:
    """R-025/ACTIONS §2: only 7 SELECT ids exist, none for a letter."""
    s = H.load_fixture("sel_pass1_mid")
    assert [a for a in s.legal_actions() if a != DROP_OUT] == SEL_ALL


@pytest.mark.parametrize("a", [LINES, PASSENGERS, BUILDINGS, VRROOOMM])
def test_R025_full_six_space_action_not_selectable(a: int) -> None:
    """R-025: with all six letters taken the action is not legal; the other six are, no DROP_OUT
    (pass 1 position: seats 0,1 placed twice, seat 2 acts)."""
    d = doc4({a: [0, 1, 2, 3, 0, 1]}, pass_=1)
    s = H.load_doc_state(d)
    assert s.current_player() == 2
    assert s.legal_actions() == [x for x in SEL_ALL if x != SELECT_BASE + a]
    with pytest.raises(Exception):  # noqa: B017
        s.apply_action(SELECT_BASE + a)


def test_R025_fills_to_six_then_blocks_in_play() -> None:
    """R-025: after six markers on Lines (letters a..f) the seventh SELECT(Lines) is gone."""
    s = H.load_fixture("sel_round1")
    for i in range(6):
        step(s, i % 4, SELECT_BASE + LINES)
    assert board_of(s)[LINES] == [0, 1, 2, 3, 0, 1]
    assert s.current_player() == 2
    assert SELECT_BASE + LINES not in s.legal_actions()


@pytest.mark.parametrize("a", [BUSES, CLOCK, STARTING])
def test_R025_single_space_actions_take_one_marker(a: int) -> None:
    """R-022/R-025: Buses, Clock and Starting Player have one space; once taken the next seat
    cannot choose them (and the board shows the taker)."""
    s = H.load_fixture("sel_round1")
    step(s, 0, SELECT_BASE + a)
    assert board_of(s)[a] == [0]
    assert s.legal_actions() == [x for x in SEL_ALL if x != SELECT_BASE + a]
    with pytest.raises(Exception):  # noqa: B017
        s.apply_action(SELECT_BASE + a)


def test_R025_single_space_taken_by_same_seat_blocks_it_too() -> None:
    """R-025: a seat that already holds Clock cannot take it a second time in pass 1."""
    s = H.load_fixture("sel_round1")
    step(s, 0, SELECT_BASE + CLOCK)
    for p in (1, 2, 3):
        step(s, p, SELECT_BASE + LINES)
    assert s.current_player() == 0
    assert SELECT_BASE + CLOCK not in s.legal_actions()


def test_R025_board_is_empty_for_unselected_actions() -> None:
    """R-025: only the selected action's list changes."""
    s = H.load_fixture("sel_round1")
    step(s, 0, SELECT_BASE + BUILDINGS)
    b = board_of(s)
    assert b[BUILDINGS] == [0]
    assert all(b[a] == [] for a in range(7) if a != BUILDINGS)


# ---- R-026 ----------------------------------------------------------------------------------


def test_R026_buses_choosable_with_five_buses() -> None:
    """R-026: Buses may be chosen by a seat that already has 5 buses (blocking choice)."""
    d = doc4(buses={0: 5})
    s = H.load_doc_state(d)
    assert s.current_player() == 0
    assert SELECT_BASE + BUSES in s.legal_actions()
    s.apply_action(SELECT_BASE + BUSES)
    assert board_of(s)[BUSES] == [0]


def test_R026_passengers_choosable_with_empty_supply() -> None:
    """R-026: Passengers may be chosen when the supply is empty (all 15 passengers waiting)."""
    d = doc4(waiting={"13": 15})
    s = H.load_doc_state(d)
    assert s.passenger_supply == 0
    assert SELECT_BASE + PASSENGERS in s.legal_actions()
    s.apply_action(SELECT_BASE + PASSENGERS)
    assert board_of(s)[PASSENGERS] == [0]


def test_R026_select_depends_only_on_free_space() -> None:
    """R-026: legality of SELECT depends only on a free space, not on any effect: with every seat
    holding 5 buses, Buses is selectable by each seat in turn until taken."""
    d = doc4(buses={0: 5, 1: 5, 2: 5, 3: 5})
    s = H.load_doc_state(d)
    assert s.legal_actions() == SEL_ALL
    s.apply_action(SELECT_BASE + BUSES)
    assert SELECT_BASE + BUSES not in s.legal_actions()  # taken, not 'pointless'
    assert s.legal_actions() == [x for x in SEL_ALL if x != SELECT_BASE + BUSES]


# ---- R-027 ----------------------------------------------------------------------------------


def test_R027_zero_markers_skipped_in_mandatory_passes() -> None:
    """R-027 (O-3): a seat with no markers at round start gets no node, not even in pass 0 or 1;
    the other seats' mandatory passes continue."""
    s = H.load_doc_state(doc4(markers={1: 0}))
    for p in (0, 2, 3, 0, 2, 3):
        step(s, p, lowest_select(s))
    assert s.seat(1).action_markers == 0
    assert s.phase == "SEL"
    assert s.current_player() == 0
    assert DROP_OUT in s.legal_actions()
    step(s, 0, DROP_OUT)
    assert s.current_player() == 2  # seat 1 still skipped


def test_R027_one_marker_places_in_pass_0_and_is_skipped_in_pass_1() -> None:
    """R-027 example: a seat with 1 marker left places it in pass 0 and is skipped in pass 1."""
    s = H.load_doc_state(doc4(markers={2: 1}))
    for p in (0, 1, 2, 3):
        step(s, p, lowest_select(s))
    assert s.seat(2).action_markers == 0
    for p in (0, 1, 3):
        step(s, p, lowest_select(s))
    assert s.current_player() == 0  # pass 2; seat 2 never appeared in pass 1
    assert sum(1 for x in s.action_board for q in x if q == 2) == 1


def test_R027_one_marker_seat_out_from_third_pass() -> None:
    """R-027/R-028: the 1-marker seat stays out in later passes, and selection ends when the
    others have dropped."""
    s = H.load_doc_state(doc4(markers={2: 1}))
    for p in (0, 1, 2, 3, 0, 1, 3):
        step(s, p, lowest_select(s))
    step(s, 0, DROP_OUT)
    step(s, 1, DROP_OUT)
    assert s.current_player() == 3  # seat 2 skipped
    step(s, 3, DROP_OUT)
    assert left_sel(s)


def test_R027_runs_out_after_third_marker_without_dropping() -> None:
    """R-027/R-028: a seat spending its last marker in pass 2 is out afterwards (no DROP_OUT
    needed); selection ends once the others drop."""
    s = H.load_doc_state(doc4(markers={0: 3}))
    for p in (0, 1, 2, 3, 0, 1, 2, 3):
        step(s, p, lowest_select(s))
    step(s, 0, lowest_select(s))  # third marker, pass 2
    assert s.seat(0).action_markers == 0
    for p in (1, 2, 3):
        step(s, p, DROP_OUT)
    assert left_sel(s)


def test_R027_skipped_seat_gets_no_legal_actions_node() -> None:
    """ACTIONS §3.4: skipped seats are never current_player; legal actions always belong to an
    active seat that has markers."""
    s = H.load_doc_state(doc4(markers={0: 0, 3: 0}))
    seen = set()
    for _ in range(4):
        seen.add(s.current_player())
        s.apply_action(lowest_select(s))
    assert seen == {1, 2}


def test_R027_all_spaces_full_forces_drop_out() -> None:
    """R-027/ACTIONS §3.4: a seat with markers but every space full has exactly [DROP_OUT].
    27 markers fill the 27 spaces: seats 0..2 placed 7, seat 3 placed 6 (pass 6)."""
    board: dict[int, list[int]] = {}
    i = 0
    for a in range(7):
        board[a] = []
        for _ in range(SPACES[a]):
            board[a].append(i % 4)
            i += 1
    assert i == 27
    s = H.load_doc_state(doc4(board, pass_=6))
    assert s.current_player() == 3
    assert s.legal_actions() == [DROP_OUT]
    step(s, 3, DROP_OUT)
    assert s.phase == "SEL" and s.current_player() == 0
    assert s.legal_actions() == [DROP_OUT]


# ---- R-028 ----------------------------------------------------------------------------------


def test_R028_selection_continues_while_a_seat_is_active() -> None:
    """R-028: with one seat still active after the others dropped, the node is that seat's."""
    s = H.load_fixture("sel_pass2_dropped")
    step(s, 0, DROP_OUT)
    step(s, 1, DROP_OUT)
    assert s.phase == "SEL"
    assert s.current_player() == 2
    assert DROP_OUT in s.legal_actions()


def test_R028_ends_when_everyone_dropped() -> None:
    """R-028: the last active seat dropping out ends Phase 1 (round stays 1, board stays filled)."""
    s = H.load_fixture("sel_pass2_dropped")
    for p in (0, 1, 2):
        step(s, p, DROP_OUT)
    assert left_sel(s)
    assert s.round == 1
    assert not s.is_terminal()
    assert board_of(s)[LINES] == [0, 1, 2, 3]
    assert board_of(s)[PASSENGERS] == [0, 1, 2, 3]


def test_R028_ends_when_everyone_dropped_or_out_of_markers() -> None:
    """R-028: a mix of dropped and marker-less seats also ends selection."""
    s = H.load_doc_state(doc4(markers={1: 0, 3: 1}))
    for p in (0, 2, 3, 0, 2):
        step(s, p, lowest_select(s))
    step(s, 0, DROP_OUT)
    step(s, 2, DROP_OUT)
    assert left_sel(s)


def test_R028_not_ended_before_last_pass() -> None:
    """R-028: while any seat remains active the phase stays SEL."""
    s = H.load_fixture("sel_pass2_dropped")
    step(s, 0, DROP_OUT)
    step(s, 1, DROP_OUT)
    assert s.phase == "SEL"
    step(s, 2, SELECT_BASE + BUILDINGS)
    assert s.phase == "SEL" and s.current_player() == 2  # only seat 2 left


def test_R028_phase_2_starts_with_first_marker_in_resolution_order() -> None:
    """R-028/R-029/ACTIONS §3.4: Phase 2 starts at Lines letter a (seat 0, LN)."""
    s = H.load_fixture("sel_pass2_dropped")
    for p in (0, 1, 2):
        step(s, p, DROP_OUT)
    assert s.phase in RESOLUTION_PHASES
    assert s.phase == "LN"
    assert s.current_player() == 0


def test_R028_three_players_full_round_of_selection() -> None:
    """R-028: 3 players, everyone places the mandatory markers then drops: Phase 1 ends."""
    s = play_to_sel(3, 0)
    for _ in range(2):
        for p in range(3):
            step(s, p, lowest_select(s))
    for p in range(3):
        step(s, p, DROP_OUT)
    assert left_sel(s) and s.round == 1


def test_R028_five_players_full_round_of_selection_start_3() -> None:
    """R-028: 5 players with start player 3; 10 mandatory markers then all drop in turn order."""
    s = play_to_sel(5, 3)
    seq = order(5, 3)
    for _ in range(2):
        for p in seq:
            step(s, p, lowest_select(s))
    for p in seq:
        assert s.phase == "SEL"
        step(s, p, DROP_OUT)
    assert left_sel(s)
    assert sum(len(x) for x in s.action_board) == 10


def test_R028_three_players_board_capacity() -> None:
    """R-022/R-025 at 3 players: same 7-action board; Buses/Clock/Starting Player hold one seat."""
    s = play_to_sel(3, 0)
    assert len(s.action_board) == 7
    step(s, 0, SELECT_BASE + STARTING)
    step(s, 1, SELECT_BASE + CLOCK)
    assert SELECT_BASE + STARTING not in s.legal_actions()
    assert SELECT_BASE + CLOCK not in s.legal_actions()
    assert board_of(s)[STARTING] == [0] and board_of(s)[CLOCK] == [1]
