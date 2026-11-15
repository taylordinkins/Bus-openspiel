"""Starting Player, round end, game end, final scores, tiebreaks, returns (R-064..R-072).

Written ahead of Phase 1 unit 12; every test is pending.

Positions
  * ``ck_doc``  a round-1 position at the last decision of the round (a Clock marker, optionally
    a Starting Player marker after it) built from ``vrrooomm/base.json`` (4 seats, 3 by dropping
    the last).  ``hand`` is each seat's Action markers in hand, not counting the ones lying on the
    board (SCENARIOS 3.1 ``markers_left``).  CLOCK_ADVANCE ends Phase 2 and runs round end.
  * ``end_doc`` a terminal ``END`` position (round-end or last-stone) with chosen points, stones and
    last_score_move, built from a seeded random setup of the right size; used for scoring,
    tiebreaks and returns.  Named versions live in ``tests/scenarios/end/*.json``.

Accessors used beyond the harness: state.phase, round, start_player, current_player(),
is_terminal(), returns(), end_types (list/tuple of strings), sites_full, seat(p).points/.stones/
.action_markers, get_game().{utility_sum,min_utility,max_utility}; ``expect`` keys final_scores,
winners (list of seats, ascending) and returns of ``check_expect``; dump_scenario for the board,
clock and last_score_move.

Not asserted because the spec leaves it open (see the report): the order of ``end_types``, the
value of ``start_player`` once the game has ended, min/max utility values.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from bus.actions import (
    CLOCK_ADVANCE,
    CLOCK_PAUSE,
    DELIVER_TO_BASE,
    PICK_PASSENGER_BASE,
    encode_build,
)
from bus.board import load_board
from tests import harness as H

B = load_board()
T = len(B.sites)
TYPES = ["house", "office", "pub"]
STONES_TOTAL = {3: 4, 4: 5, 5: 5}


# ---- helpers ---------------------------------------------------------------------------------


def dump(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s)


def board_empty(s: Any) -> bool:
    ab = dump(s).get("action_board", {})
    return all(x is None for lst in ab.values() for x in lst)


def fill_sites(doc: dict[str, Any], leave: int | None = None, type_: str = "pub") -> None:
    have = {b["site"] for b in doc["buildings"]}
    for t in range(T):
        if t not in have and t != leave:
            doc["buildings"].append({"site": t, "type": type_})


def ck_doc(
    hand: list[int] | None = None,
    start: int = 0,
    sp: int | None = None,
    full: bool = False,
    short: bool = False,
    n: int = 4,
    points: dict[int, tuple[int, int]] | None = None,
    returns: str | None = None,
) -> dict[str, Any]:
    """Last decision of round 1: seat 0's Clock marker (advance / pause), then, if ``sp`` is
    given, that seat's Starting Player marker.  ``full``: every site has a building; ``short``:
    every site but one."""
    d = H.load_doc("vrrooomm/base")
    for k in ("expect", "description", "current_player"):
        d.pop(k, None)
    if n == 3:
        d["parameters"]["players"] = 3
        d["seats"].pop()
    if returns:
        d["parameters"]["returns"] = returns
    hand = hand if hand is not None else [10] * n
    assert len(hand) == n
    for p in range(n):
        d["seats"][p]["markers_left"] = hand[p]
    for p, (pts, lsm) in (points or {}).items():
        d["seats"][p]["points"] = pts
        d["seats"][p]["last_score_move"] = lsm
    d["phase"] = "CK"
    d["start_player"] = start
    d["action_board"] = {"clock": [0]}
    if sp is not None:
        d["action_board"]["starting_player"] = [sp]
    d["pending"] = {"cursor": {"letter": 0}}
    if full:
        fill_sites(d)
    if short:
        fill_sites(d, leave=max(range(T), key=lambda t: (B.sites[t].tier, t)))
    return d


def finish(doc: dict[str, Any]) -> Any:
    s = H.load_doc_state(doc)
    assert s.phase == "CK" and not s.is_terminal()
    s.apply_action(CLOCK_ADVANCE)
    return s


_SETUP: dict[int, dict[str, Any]] = {}


def setup_doc(n: int) -> dict[str, Any]:
    if n not in _SETUP:
        st = H.load_doc_state({"format": 1, "parameters": {"players": n}})
        rng = random.Random(3)
        while st.phase in ("SB", "L1", "L2"):
            st.apply_action(rng.choice(st.legal_actions()))
        _SETUP[n] = dump(st)
    return copy.deepcopy(_SETUP[n])


def end_doc(
    pts: list[int],
    stones: list[int] | None = None,
    lsm: list[int | None] | None = None,
    types: tuple[str, ...] = ("sites_full",),
    returns: str = "win_loss",
) -> dict[str, Any]:
    """Terminal position with the given points, stones and last scoring move numbers (by seat).
    ``lsm`` defaults to distinct move numbers 10, 20, ... for seats with points > 0."""
    n = len(pts)
    stones = stones if stones is not None else [0] * n
    d = setup_doc(n)
    for k in ("expect", "description", "current_player", "move_offset", "supply"):
        d.pop(k, None)
    d["parameters"]["returns"] = returns
    d["phase"] = "END"
    d["pending"] = {}
    d["action_board"] = {}
    d.pop("action_board")
    d["end"] = {"types": list(types)}
    for p in range(n):
        d["seats"][p]["points"] = pts[p]
        d["seats"][p]["stones"] = stones[p]
        if lsm is not None:
            d["seats"][p]["last_score_move"] = lsm[p]
        else:
            d["seats"][p]["last_score_move"] = 10 * (p + 1) if pts[p] > 0 else None
    d["stones_on_clock"] = STONES_TOTAL[n] - sum(stones)
    assert d["stones_on_clock"] >= 0
    return d


def end_state(*a: Any, **kw: Any) -> Any:
    s = H.load_doc_state(end_doc(*a, **kw))
    assert s.is_terminal()
    return s


def win_loss(n: int, winners: list[int]) -> list[float]:
    k = len(winners)
    return [(n - k) / (k * (n - 1)) if p in winners else -1 / (n - 1) for p in range(n)]


def margin(finals: list[int]) -> list[float]:
    n = len(finals)
    out = []
    for p in range(n):
        others = (sum(finals) - finals[p]) / (n - 1)
        out.append(max(-1.0, min(1.0, (finals[p] - others) / 10)))
    return out


# ---- R-064 Starting Player ---------------------------------------------------------------------


@pytest.mark.parametrize("n,start", [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_R064_default_start_passes_to_next_seat(n: int, start: int) -> None:
    """R-064: with no Starting Player marker the start passes to the next seat, ascending with
    wrap-around (4 players: P2 started, no marker, P3 starts next)."""
    s = finish(ck_doc(n=n, start=start))
    nxt = (start + 1) % n
    assert s.round == 2 and s.phase == "SEL"
    assert s.start_player == nxt
    assert s.current_player() == nxt


@pytest.mark.parametrize("owner", [0, 2, 3])
def test_R064_starting_player_marker_owner_starts(owner: int) -> None:
    """R-064: a marker on Starting Player makes its owner the next start player, whatever the
    default would have been (start is P1, default successor P2)."""
    s = finish(ck_doc(start=1, sp=owner))
    assert s.start_player == owner
    assert s.current_player() == owner


def test_R064_starting_player_marker_of_the_current_start_player_keeps_the_start() -> None:
    """R-064: P1 started and takes Starting Player: P1 starts again (not P2)."""
    s = finish(ck_doc(start=1, sp=1))
    assert s.start_player == 1 and s.current_player() == 1


def test_R064_starting_player_marker_three_players() -> None:
    """R-064 with 3 players: P2 started, P1 took the action: P1 starts next (no wrap default)."""
    s = finish(ck_doc(n=3, start=2, sp=1))
    assert s.start_player == 1


def test_R064_start_player_does_not_change_before_the_round_ends() -> None:
    """R-064: the Starting Player marker resolves after the Clock; until then the start player
    of the round is unchanged (the Clock decision node is still the round's start player P1)."""
    s = H.load_doc_state(ck_doc(start=1, sp=3))
    assert s.start_player == 1


# ---- R-065 spent markers are removed -----------------------------------------------------------


def test_R065_spent_markers_are_not_returned_to_hands() -> None:
    """R-065: markers lying on the action board (seat 0 Clock, seat 2 Starting Player) are
    removed at round end; the hands (10 each, board markers not included) do not grow."""
    s = finish(ck_doc(sp=2))
    assert s.phase == "SEL"
    assert [s.seat(p).action_markers for p in range(4)] == [10, 10, 10, 10]


def test_R065_next_round_starts_with_an_empty_action_board() -> None:
    """R-065, ACTIONS 3.11: the next round's SEL begins with an empty board, pass 0, nobody
    dropped."""
    s = finish(ck_doc(sp=2))
    assert board_empty(s)
    d = dump(s)
    assert d["pending"].get("pass") == 0
    assert d["pending"].get("dropped") == []


def test_R065_markers_never_increase_across_the_round_end() -> None:
    """R-065: a seat's markers-left count only decreases; it is not larger after round end."""
    s0 = H.load_doc_state(ck_doc(hand=[7, 3, 0, 5], sp=1))
    before = [s0.seat(p).action_markers for p in range(4)]
    s0.apply_action(CLOCK_ADVANCE)
    after = [s0.seat(p).action_markers for p in range(4)]
    assert all(a <= b for a, b in zip(after, before, strict=True))
    assert after == [7, 3, 0, 5]


def test_R065_game_over_board_is_empty_after_round_end() -> None:
    """R-065/R-067/SCENARIOS 9.9: a game that ends at round end has an empty action board."""
    s = finish(ck_doc(hand=[0, 0, 0, 0], sp=2))
    assert s.is_terminal()
    assert board_empty(s)


# ---- R-066 / R-067 end conditions --------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
def test_R066_sites_full_ends_the_game_at_round_end(n: int) -> None:
    """R-066 cond 1: every site has a building (hands plentiful): the game ends at the end of
    the round, end type exactly sites_full, no round 2."""
    s = finish(ck_doc(n=n, full=True, sp=1))
    assert s.is_terminal() and s.phase == "END"
    assert s.round == 1
    assert sorted(s.end_types) == ["sites_full"]


def test_R066_one_site_short_does_not_end_the_game() -> None:
    """R-066 cond 1 boundary: all sites but one have a building: no end; round 2 starts."""
    s = finish(ck_doc(short=True))
    assert not s.is_terminal()
    assert s.round == 2 and s.phase == "SEL"


def test_R066_sites_full_at_round_start_ends_only_at_round_end() -> None:
    """R-066/R-067: with every site already built the Clock marker still resolves (a decision
    node, not terminal); the game ends only after it, at round end."""
    s = H.load_doc_state(ck_doc(full=True))
    assert not s.is_terminal()
    assert sorted(s.legal_actions()) == sorted([CLOCK_ADVANCE, CLOCK_PAUSE])
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal()
    assert sorted(s.end_types) == ["sites_full"]


@pytest.mark.parametrize(
    "hand",
    [[5, 0, 0, 0], [0, 0, 0, 5], [0, 4, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    ids=["only-p0", "only-p3", "only-p1", "only-one-marker", "nobody"],
)
def test_R066_markers_condition_one_or_zero_seats_with_markers(hand: list[int]) -> None:
    """R-066 cond 3: at round end at most one seat still has Action markers (0 counts too):
    the game ends with end type exactly markers; sites not full."""
    s = finish(ck_doc(hand=hand))
    assert s.is_terminal() and s.phase == "END"
    assert s.round == 1
    assert sorted(s.end_types) == ["markers"]


@pytest.mark.parametrize("hand", [[0, 0, 5], [0, 5, 0], [0, 0, 0]])
def test_R066_markers_condition_three_players(hand: list[int]) -> None:
    """R-066 cond 3 with 3 players."""
    s = finish(ck_doc(n=3, hand=hand))
    assert s.is_terminal() and sorted(s.end_types) == ["markers"]


@pytest.mark.parametrize(
    "hand",
    [[5, 5, 0, 0], [1, 0, 1, 0], [0, 0, 3, 3], [4, 4, 4, 4], [0, 1, 0, 1]],
    ids=["two", "two-single", "two-high", "all", "two-single-b"],
)
def test_R066_two_or_more_seats_with_markers_no_end(hand: list[int]) -> None:
    """R-066 cond 3 must NOT trigger when two (or more) seats still have markers: round 2
    starts in SEL, not terminal."""
    s = finish(ck_doc(hand=hand))
    assert not s.is_terminal()
    assert s.round == 2 and s.phase == "SEL"
    assert s.end_types in ((), [], None) or list(s.end_types) == []


def test_R066_board_markers_are_removed_before_the_markers_check() -> None:
    """R-066 cond 3 'after R-065': seat 0 has 0 in hand but its Clock marker lies on the board;
    P1 has 3 in hand.  The spent marker is gone, so only P1 has markers: the game ends.  (If
    board markers still counted, two seats would have markers and the game would go on.)"""
    s = finish(ck_doc(hand=[0, 3, 0, 0]))
    assert s.is_terminal()
    assert sorted(s.end_types) == ["markers"]


def test_R066_starting_player_owner_hand_empty_counts_as_without_markers() -> None:
    """R-066 cond 3: P2's only remaining marker was the Starting Player marker on the board;
    P2 has 0 in hand afterwards, P1 has 2: one seat with markers, game ends."""
    s = finish(ck_doc(hand=[0, 2, 0, 0], sp=2))
    assert s.is_terminal() and sorted(s.end_types) == ["markers"]


def test_R066_markers_condition_mid_round_does_not_end_the_round() -> None:
    """R-066/R-067: nobody has markers in hand while Phase 2 is still running: not terminal,
    the Clock marker still resolves; the game ends only at round end."""
    s = H.load_doc_state(ck_doc(hand=[0, 0, 0, 0], sp=3))
    assert not s.is_terminal()
    assert sorted(s.legal_actions()) == sorted([CLOCK_ADVANCE, CLOCK_PAUSE])


@pytest.mark.parametrize("n", [3, 4])
def test_R066_both_round_end_conditions_are_both_recorded(n: int) -> None:
    """R-066: sites full and markers can hold together; the end type is the list of both."""
    s = finish(ck_doc(n=n, full=True, hand=[0] * n))
    assert s.is_terminal()
    assert sorted(s.end_types) == ["markers", "sites_full"]
    assert len(list(s.end_types)) == 2


def test_R066_sites_full_with_markers_left_is_only_sites_full() -> None:
    """R-066: sites full, two seats with markers: only sites_full is recorded."""
    s = finish(ck_doc(full=True, hand=[3, 3, 0, 0]))
    assert sorted(s.end_types) == ["sites_full"]


def test_R066_end_types_are_plain_strings_without_duplicates() -> None:
    """R-066/SCENARIOS 4.7: end_types entries are drawn from the three names, no duplicates."""
    s = finish(ck_doc(full=True, hand=[0, 0, 0, 0]))
    types = list(s.end_types)
    assert len(types) == len(set(types))
    assert set(types) <= {"sites_full", "last_stone", "markers"}


def _last_stone_doc() -> dict[str, Any]:
    d = H.load_doc("clock/ck_last_stone_4p")
    for k in ("expect", "description"):
        d.pop(k, None)
    return d


def test_R066_last_stone_is_alone_even_if_other_conditions_hold() -> None:
    """R-066: the time-stone end is immediate, so it is always alone in the set even though
    every site has a building and nobody has markers in hand."""
    d = _last_stone_doc()
    fill_sites(d)
    for p in range(4):
        d["seats"][p]["markers_left"] = 0
    s = H.load_doc_state(d)
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal()
    assert list(s.end_types) == ["last_stone"]


def test_R066_last_stone_with_plenty_of_markers_is_alone() -> None:
    """R-066 cond 2: the plain last-stone end records exactly last_stone."""
    s = H.load_doc_state(_last_stone_doc())
    s.apply_action(CLOCK_PAUSE)
    assert list(s.end_types) == ["last_stone"]
    assert s.legal_actions() == []


def test_R066_sites_full_filled_mid_round_round_still_finishes() -> None:
    """R-066/R-067/R-049 example: Buildings fills the last site in Phase 2.  The rest of the
    round resolves first (P1 Clock advance to office, P2 Vrrooomm! delivers one passenger and
    scores, P3 Starting Player), then the game ends with sites_full, still in round 1."""
    d = H.load_doc("vrrooomm/base")
    for k in ("expect", "description"):
        d.pop(k, None)
    last = max(range(T), key=lambda t: (B.sites[t].tier, t))
    have = {b["site"] for b in d["buildings"]}
    assert last not in have
    fill_sites(d, leave=last)
    d["waiting"] = {"22": 1}
    d.pop("supply", None)
    for p in range(4):
        d["seats"][p]["markers_left"] = 10
    d["phase"] = "BD"
    d["action_board"] = {
        "buildings": [0, None, None, None, None, None],
        "clock": [1],
        "vrrooomm": [2, None, None, None, None, None],
        "starting_player": [3],
    }
    d["pending"] = {"cursor": {"letter": 0}, "units_left": 1}
    s = H.load_doc_state(d)
    assert s.phase == "BD" and s.current_player() == 0
    s.apply_action(encode_build(last, 2))
    assert s.sites_full is True
    assert not s.is_terminal()
    assert s.phase == "CK" and s.current_player() == 1
    s.apply_action(CLOCK_ADVANCE)
    assert not s.is_terminal()
    assert s.phase == "VP" and s.current_player() == 2
    s.apply_action(PICK_PASSENGER_BASE + 22)
    assert s.phase == "VD"
    site = next(a for a in s.legal_actions() if a >= DELIVER_TO_BASE)
    s.apply_action(site)
    assert s.is_terminal(), "Starting Player resolves automatically, then the game ends"
    assert s.round == 1
    assert sorted(s.end_types) == ["sites_full"]
    assert s.seat(2).points == 1
    assert dump(s)["clock"] == "office"
    assert board_empty(s)


# ---- R-067 timing -------------------------------------------------------------------------------


def test_R067_no_further_round_or_action_after_a_round_end_game_over() -> None:
    """R-067: a game that ends at round end starts no new round: phase END, round unchanged,
    no legal actions, nobody is asked to select."""
    s = finish(ck_doc(full=True, sp=1))
    assert s.phase == "END" and s.round == 1
    assert s.legal_actions() == []
    assert s.is_terminal()


# ---- R-068 final score --------------------------------------------------------------------------


def test_R068_final_score_is_points_minus_stones_spec_example() -> None:
    """R-068 example: 12 points and 2 stones: final score 10; the disc stays at 12."""
    s = end_state([12, 5, 0, 0], [2, 0, 0, 1])
    H.scenario_module().check_expect(s, {"final_scores": [10, 5, 0, -1]})
    assert s.seat(0).points == 12 and s.seat(0).stones == 2
    assert s.seat(3).points == 0


def test_R068_final_scores_can_be_negative_and_have_no_floor() -> None:
    """R-068: zero points and 3 stones is -3."""
    s = end_state([0, 4, 0, 0], [3, 1, 1, 0])
    H.scenario_module().check_expect(s, {"final_scores": [-3, 3, -1, 0]})


@pytest.mark.parametrize("n", [3, 5])
def test_R068_final_scores_other_player_counts(n: int) -> None:
    """R-068 at 3 and 5 seats."""
    pts = [9, 4, 7, 0, 2][:n]
    stones = [1, 0, 2, 1, 1][:n] if n == 5 else [1, 0, 2]
    s = end_state(pts, stones)
    H.scenario_module().check_expect(
        s, {"final_scores": [p - t for p, t in zip(pts, stones, strict=True)]}
    )


def test_R068_final_score_after_a_natural_round_end() -> None:
    """R-068: scores after a round-end game over: points 4 for P1 with no stones: final 4."""
    s = finish(ck_doc(hand=[0, 0, 0, 0], points={1: (4, 7)}))
    H.scenario_module().check_expect(s, {"final_scores": [0, 4, 0, 0], "winners": [1]})


# ---- R-069 winner and tiebreaks -----------------------------------------------------------------

# (points, stones, last_score_move, winners)
CASES: list[tuple[str, list[int], list[int], list[int | None] | None, list[int]]] = [
    ("high score wins", [10, 8, 5, 3], [0, 0, 0, 0], None, [0]),
    ("score beats stones", [5, 10, 0, 0], [3, 0, 0, 0], None, [1]),
    ("stones tiebreak spec example", [11, 9, 0, 0], [2, 0, 0, 0], [30, 5, None, None], [0]),
    ("more stones wins, not fewer", [9, 11, 0, 0], [0, 2, 0, 0], [5, 30, None, None], [1]),
    ("stones tiebreak among three", [8, 7, 6, 0], [2, 1, 0, 0], [30, 20, 10, None], [0]),
    ("move tiebreak spec example", [0, 8, 0, 8], [0, 0, 0, 0], [None, 340, None, 412], [1]),
    ("move tiebreak later seat first", [0, 8, 0, 8], [0, 0, 0, 0], [None, 412, None, 340], [3]),
    ("move tiebreak three-way", [6, 6, 6, 0], [0, 0, 0, 0], [30, 10, 20, None], [1]),
    ("stones filter before move", [8, 8, 6, 0], [2, 2, 0, 0], [20, 10, 1, None], [1]),
    ("stones decide before move", [9, 8, 0, 0], [1, 0, 0, 0], [50, 3, None, None], [0]),
    (
        "never scored loses on stones to scorer",
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [4, None, None, None],
        [0],
    ),
]


@pytest.mark.parametrize(
    "pts,stones,lsm,winners",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_R069_winners_follow_score_then_stones_then_move(
    pts: list[int], stones: list[int], lsm: list[int | None] | None, winners: list[int]
) -> None:
    """R-069/R-070: highest final score; ties by most stones; then earlier last_score_move.
    The winners list and the win_loss returns agree."""
    s = end_state(pts, stones, lsm)
    H.scenario_module().check_expect(s, {"winners": winners})
    assert list(s.returns()) == pytest.approx(win_loss(4, winners))


def test_R069_spec_example_stones_break_the_tie() -> None:
    """R-069 example: P0 and P2 both final 9; P0 took 2 stones, P2 none: P0 wins (P0 = 11 points,
    P2 = 9 points)."""
    s = end_state([11, 0, 9, 0], [2, 0, 0, 0], [40, None, 10, None])
    H.scenario_module().check_expect(s, {"final_scores": [9, 0, 9, 0], "winners": [0]})
    assert s.returns()[0] == pytest.approx(1.0)


def test_R069_a_lower_score_never_wins_on_a_tiebreak() -> None:
    """R-069 must NOT: P1 has far more stones and the earlier move but a lower final score."""
    s = end_state([9, 8, 0, 0], [0, 3, 0, 0], [50, 5, None, None])
    H.scenario_module().check_expect(s, {"final_scores": [9, 5, 0, 0], "winners": [0]})


def test_R069_all_zero_points_all_share_the_win() -> None:
    """R-069/R-070 (Q3, O-4): nobody scored, nobody took a stone: all four are tied to the end
    and share the win; nobody is arbitrarily ranked."""
    s = end_state([0, 0, 0, 0])
    H.scenario_module().check_expect(s, {"winners": [0, 1, 2, 3]})
    assert list(s.returns()) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_R069_zero_point_tie_with_equal_stones_is_shared_but_not_with_others() -> None:
    """R-069/R-070: P0 and P1 never scored and have equal stones (0); P2, P3 each hold a stone.
    P0 and P1 share the win (final 0 > -1)."""
    s = end_state([0, 0, 0, 0], [0, 0, 1, 1])
    H.scenario_module().check_expect(s, {"winners": [0, 1]})
    assert list(s.returns()) == pytest.approx([1 / 3, 1 / 3, -1 / 3, -1 / 3])


def test_R069_all_negative_scores_still_have_a_winner() -> None:
    """R-069: everybody on 0 points but with different stones: the one with the fewest stones
    (highest final score) wins alone."""
    s = end_state([0, 0, 0, 0], [1, 1, 0, 2])
    H.scenario_module().check_expect(s, {"final_scores": [-1, -1, 0, -2], "winners": [2]})


def test_R069_shared_win_three_players() -> None:
    """R-069 at 3 players: P0 and P1 tie on everything (0 points, 1 stone each); P2 holds 2."""
    s = end_state([0, 0, 0], [1, 1, 2])
    H.scenario_module().check_expect(s, {"winners": [0, 1]})
    assert list(s.returns()) == pytest.approx([0.25, 0.25, -0.5])


def test_R069_five_players_winner() -> None:
    """R-069 at 5 players: P3 is highest; ties lower down are irrelevant."""
    s = end_state([3, 3, 0, 9, 0], [0, 0, 0, 1, 0])
    H.scenario_module().check_expect(s, {"winners": [3]})


# ---- R-070 reached the points total first --------------------------------------------------------


def test_R070_earlier_last_score_move_wins_spec_example() -> None:
    """R-070 example: P1 and P3 tie on points and stones with 8 points; last scoring moves 340
    and 412: P1 wins, alone."""
    s = end_state([0, 8, 0, 8], [0, 0, 0, 0], [None, 340, None, 412])
    H.scenario_module().check_expect(s, {"winners": [1]})
    assert s.returns()[1] == pytest.approx(1.0)
    assert s.returns()[3] == pytest.approx(-1 / 3)


def test_R070_last_score_move_is_recorded_through_play_and_decides() -> None:
    """R-070/R-062: scoring moves played after loading get later move numbers than the loaded
    ones (move offset 10, pick, deliver: 12).  P0 (7 points, last scoring move 3) and P2
    (6 points, move 4) tie at 7 after P2 scores again at VD:
    P2's last move is then later than P0's, so P0 wins the tie."""
    d = H.load_doc("vrrooomm/base")
    for k in ("expect", "description"):
        d.pop(k, None)
    d["move_offset"] = 10
    d["seats"][0]["points"] = 7
    d["seats"][0]["last_score_move"] = 3
    d["seats"][2]["points"] = 6
    d["seats"][2]["last_score_move"] = 4
    for p in range(4):
        d["seats"][p]["markers_left"] = 0
    d["phase"] = "VP"
    d["action_board"] = {"vrrooomm": [2, None, None, None, None, None]}
    d["pending"] = {"cursor": {"letter": 0}, "moves_left": 1}
    d["seats"][2]["markers_left"] = 0
    d["seats"][1]["markers_left"] = 1
    s = H.load_doc_state(d)
    s.apply_action(PICK_PASSENGER_BASE + 22)
    site = next(a for a in s.legal_actions() if a >= DELIVER_TO_BASE)
    s.apply_action(site)
    assert s.is_terminal(), "only P1 has a marker in hand: markers end"
    assert s.seat(2).points == 7
    assert dump(s)["seats"][2]["last_score_move"] == 12
    H.scenario_module().check_expect(s, {"winners": [0]})


def test_R070_zero_point_seats_never_beat_each_other() -> None:
    """R-070: seats that never scored are treated as reaching 0 at the same instant: two of
    them tied on stones share; neither is ranked above the other by seat number."""
    s = end_state([0, 0, 0, 0], [0, 0, 0, 0])
    assert sorted(set(round(r, 9) for r in s.returns())) == [0.0]


# ---- R-071 win_loss returns -------------------------------------------------------------------


def test_R071_sole_winner_gets_one_others_minus_one_third() -> None:
    """R-071 example n=4: sole winner +1, the other three -1/3 each."""
    s = end_state([6, 4, 3, 1])
    assert list(s.returns()) == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])


def test_R071_two_winners_get_one_third_each() -> None:
    """R-071 example n=4: k=2: (4-2)/(2*3) = +1/3 each, the other two -1/3."""
    s = end_state([0, 0, 0, 0], [0, 0, 1, 2])
    assert list(s.returns()) == pytest.approx([1 / 3, 1 / 3, -1 / 3, -1 / 3])


@pytest.mark.parametrize(
    "pts,stones,winners",
    [
        ([5, 2, 1], [0, 0, 0], [0]),
        ([0, 0, 0], [0, 0, 0], [0, 1, 2]),
        ([0, 0, 0], [1, 1, 2], [0, 1]),
        ([3, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0]),
        ([0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 1, 2]),
        ([0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 1]),
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 2, 3, 4]),
    ],
)
def test_R071_formula_at_three_four_five_players(
    pts: list[int], stones: list[int], winners: list[int]
) -> None:
    """R-071: winners get (n-k)/(k(n-1)), the rest -1/(n-1); the vector sums to 0 (everybody
    tied, k=n, gives zeros)."""
    n = len(pts)
    s = end_state(pts, stones)
    r = list(s.returns())
    assert r == pytest.approx(win_loss(n, winners))
    assert sum(r) == pytest.approx(0.0)


def test_R071_five_players_sole_winner_values() -> None:
    """R-071 n=5, k=1: +1 and four times -1/4."""
    s = end_state([2, 7, 0, 0, 0])
    assert list(s.returns()) == pytest.approx([-0.25, 1.0, -0.25, -0.25, -0.25])


def test_R071_returns_are_zero_before_the_game_ends() -> None:
    """R-071: returns() exists but is all zeros at a non-terminal state (any returns mode)."""
    for mode in ("win_loss", "margin"):
        s = H.load_doc_state(ck_doc(returns=mode, points={1: (4, 7)}))
        assert not s.is_terminal()
        assert list(s.returns()) == [0.0, 0.0, 0.0, 0.0]
    s2 = H.load_doc_state({"format": 1})
    assert list(s2.returns()) == [0.0] * 4


def test_R071_returns_after_a_natural_round_end() -> None:
    """R-071: a game ended by the markers condition: P1 (4 points) wins: [-1/3, 1, -1/3, -1/3]."""
    s = finish(ck_doc(hand=[0, 0, 0, 0], points={1: (4, 7)}))
    assert list(s.returns()) == pytest.approx([-1 / 3, 1.0, -1 / 3, -1 / 3])


def test_R071_returns_for_a_last_stone_end() -> None:
    """R-071 + R-053/R-068: ck_last_stone_4p, P3 takes the last stone.  Final scores
    P0 2-2=0, P1 0-2=-2, P2 0, P3 -1.  P0 and P2 tie at 0; P0 holds 2 stones, P2 none: P0 wins
    alone: [1, -1/3, -1/3, -1/3]."""
    s = H.load_doc_state(_last_stone_doc())
    s.apply_action(CLOCK_PAUSE)
    assert s.is_terminal() and list(s.end_types) == ["last_stone"]
    H.scenario_module().check_expect(s, {"final_scores": [0, -2, 0, -1], "winners": [0]})
    assert list(s.returns()) == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])


@pytest.mark.parametrize("name", ["end_sites_full", "end_both", "end_last_stone"])
def test_R071_existing_end_fixtures_have_sole_winner_p0(name: str) -> None:
    """R-071 on the unit-3 END fixtures: P0 (3 points) beats P1 (2 points) in every one (stones
    2 and 3 in the last-stone fixture give finals 1 and -1)."""
    s = H.load_fixture(name)
    assert list(s.returns()) == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])


def test_R071_win_loss_is_the_default_returns_mode() -> None:
    """R-002/R-071: no `returns` parameter means win_loss."""
    d = end_doc([6, 4, 3, 1])
    d["parameters"].pop("returns")
    s = H.load_doc_state(d)
    assert list(s.returns()) == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])


@pytest.mark.parametrize(
    "pts,stones",
    [([6, 4, 3, 1], [0] * 4), ([0, 0, 0, 0], [0, 0, 1, 2]), ([2, 2, 2], [0, 0, 0])],
)
def test_R071_utility_bounds_and_zero_sum(pts: list[int], stones: list[int]) -> None:
    """R-071, OPENSPIEL_NOTES: the game's utility_sum is 0 for win_loss and every return lies in
    [min_utility, max_utility]."""
    s = end_state(pts, stones, lsm=[10 * (i + 1) if p else None for i, p in enumerate(pts)])
    g = s.get_game()
    assert g.utility_sum() == pytest.approx(0.0)
    r = list(s.returns())
    assert g.min_utility() <= min(r) + 1e-9
    assert g.max_utility() >= max(r) - 1e-9


# ---- R-072 margin returns ---------------------------------------------------------------------


def test_R072_spec_example_three_players() -> None:
    """R-072 example (n=3): finals 14, 8, 5: seat 0 gets (14 - 6.5)/10 = 0.75; seat 1
    (8 - 9.5)/10 = -0.15; seat 2 (5 - 11)/10 = -0.6."""
    s = end_state([14, 8, 5], returns="margin")
    assert list(s.returns()) == pytest.approx([0.75, -0.15, -0.6])


def test_R072_four_players_mean_is_over_the_other_three() -> None:
    """R-072 n=4, finals 10, 0, 0, 0: seat 0 (10 - 0)/10 = 1; others (0 - 10/3)/10 = -1/3."""
    s = end_state([10, 0, 0, 0], returns="margin")
    assert list(s.returns()) == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])


def test_R072_five_players_with_stones() -> None:
    """R-072 n=5, points 4,2,0,0,0 with stones 0,0,0,2,3: finals 4,2,0,-2,-3.  Seat 0:
    (4 - (-3/4))/10 = 0.475; seat 4: (-3 - 1)/10 = -0.4."""
    s = end_state([4, 2, 0, 0, 0], [0, 0, 0, 2, 3], returns="margin")
    r = list(s.returns())
    assert r[0] == pytest.approx(0.475)
    assert r[4] == pytest.approx(-0.4)
    assert r == pytest.approx(margin([4, 2, 0, -2, -3]))


def test_R072_clipped_to_plus_minus_one() -> None:
    """R-072: finals 30, 0, 0 give raw 3.0 and -1.5, -1.5: clipped to 1, -1, -1."""
    s = end_state([30, 0, 0], returns="margin")
    assert list(s.returns()) == pytest.approx([1.0, -1.0, -1.0])


def test_R072_exact_boundary_values_are_not_distorted() -> None:
    """R-072: raw value exactly 1.0 (finals 10, 0, 0, 0 is 1.0 for seat 0) stays 1.0, and raw
    -1.0 (seat 0 at -10 against zeros... via stones) stays -1.0 with no off-by-one clipping."""
    s = end_state([0, 0, 0, 0], [5, 0, 0, 0], returns="margin")
    r = list(s.returns())
    assert r[0] == pytest.approx(-0.5)
    s2 = end_state([10, 0, 0, 0], returns="margin")
    assert s2.returns()[0] == pytest.approx(1.0)


def test_R072_ties_and_shared_wins_do_not_change_margin() -> None:
    """R-072: margin uses only final scores: tied top scores get equal returns, whatever the
    tiebreak says (P0 and P1 both final 5; P0 won on move number)."""
    s = end_state([5, 5, 0, 0], [0, 0, 0, 0], [10, 20, None, None], returns="margin")
    r = list(s.returns())
    assert r[0] == pytest.approx(r[1])
    assert r == pytest.approx(margin([5, 5, 0, 0]))
    assert r[0] == pytest.approx((5 - 5 / 3) / 10)


def test_R072_all_equal_scores_give_zeros() -> None:
    """R-072: nobody ahead of the mean of the others: all returns 0."""
    s = end_state([0, 0, 0, 0], returns="margin")
    assert list(s.returns()) == pytest.approx([0.0] * 4)


def test_R072_is_not_zero_sum() -> None:
    """R-072: the game is general-sum under margin: finals 14, 8, 5 sum to 0.75 - 0.15 - 0.6 = 0
    here, but finals 10, 0, 0, 0 sum to 1 - 1 = 0 too; use clipped finals 30, 0, 0 (sum -1)."""
    s = end_state([30, 0, 0], returns="margin")
    assert sum(s.returns()) == pytest.approx(-1.0)


def test_R072_returns_zero_before_terminal_and_bounded_after() -> None:
    """R-072: zeros before the end; after the end every return lies in [-1, 1] and within the
    game's min/max utility."""
    s = H.load_doc_state(ck_doc(returns="margin", points={1: (4, 7)}))
    assert list(s.returns()) == [0.0] * 4
    s.apply_action(CLOCK_ADVANCE)
    t = end_state([40, 0, 0, 0, 0], [0, 0, 0, 0, 0], returns="margin")
    g = t.get_game()
    for r in list(t.returns()):
        assert -1.0 <= r <= 1.0
        assert g.min_utility() - 1e-9 <= r <= g.max_utility() + 1e-9


def test_R072_margin_after_a_natural_round_end() -> None:
    """R-072: a margin game ending on the markers condition: P1 with 4 points, others 0:
    P1 (4 - 0)/10 = 0.4, others (0 - 4/3)/10."""
    s = finish(ck_doc(hand=[0, 0, 0, 0], points={1: (4, 7)}, returns="margin"))
    assert list(s.returns()) == pytest.approx([-0.13333333, 0.4, -0.13333333, -0.13333333])


def test_R072_margin_for_a_last_stone_end() -> None:
    """R-072 + R-053: last-stone end in a margin game: finals [0, -2, 0, -1]."""
    d = _last_stone_doc()
    d["parameters"]["returns"] = "margin"
    s = H.load_doc_state(d)
    s.apply_action(CLOCK_PAUSE)
    assert list(s.returns()) == pytest.approx(margin([0, -2, 0, -1]))


# ---- named END fixtures (tests/scenarios/end/) --------------------------------------------------

NAMED = [
    "end/tiebreak_stones",
    "end/tiebreak_move",
    "end/tiebreak_shared",
    "end/returns_win_loss_3p_shared",
    "end/returns_margin_3p",
]


@pytest.mark.parametrize("name", NAMED)
def test_R069_named_end_fixtures_match_their_expect_blocks(name: str) -> None:
    """R-068..R-072: each END fixture loads as terminal and its expect block (final_scores,
    winners, returns) holds."""
    s = H.run_fixture(name)
    assert s.is_terminal()
