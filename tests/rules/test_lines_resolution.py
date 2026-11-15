"""Resolving a Lines marker in phase LN (R-040, R-042; R-038 sequential evaluation, O-2).

Written ahead of Phase 1 unit 8; every behaviour test is pending.  Geometry (R-036..R-039, R-041)
is already implemented and tested in test_lines_geometry.py; this file adds the *per-unit* decision
node of phase LN and the maximal-resolution restriction (user decision Q-B, ACTIONS 3.5):

  * a unit offers only (end, street) choices whose ``best`` equals the maximum over all choices,
    where best(e, s) = 1 + the most further placements reachable within the marker's remaining
    power and the seat's Bus Line markers left (25 minus its line length);
  * every EXTEND spends one Bus Line marker and one unit; a street placed earlier in the same
    resolution counts as occupied for later units (O-2) and for other seats' later markers;
  * units that can no longer be placed are lost with no node (R-040).

``best`` is checked against ``restricted`` below, an independent brute-force oracle written from
the spec text (depth-first over all sequential placements; no reuse of bus.lines).  Concrete
street ids appear only in the fixtures under tests/scenarios/lines/ (res_*.json and the older
geometry fixtures); each test also asserts role-based facts (a choice is dropped, a choice is
kept, counts of placements) so a shared misreading cannot hide.

State API assumed (from the unit brief): phase, current_player(), units_left, cursor_letter,
bus_max, seat(p).line, seat(p).line_markers (= 25 - len(line)), legal_actions(), clone(),
action_to_string.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from bus.actions import DROP_OUT, encode_extend
from bus.board import load_board
from tests import harness as H

B = load_board()
LinesT = tuple[tuple[int, ...], ...]
BUS_LINE_MARKERS = 25

# fixtures added for this unit (units_left > 1) and the older geometry fixtures
RES_FIXTURES = [
    "res_dead_end_power3",
    "res_bite_dead_end",
    "res_bite_both_ends",
    "res_bite_power3",
    "res_bite_short",
    "res_equal_best",
    "res_supply_one_left",
    "res_supply_two_left",
]
GEOMETRY_FIXTURES = [
    "single_street_midline",
    "red_cd",
    "three_player_per_end",
    "shared_end",
    "shared_end_same_arrival",
    "shared_end_single_street_other",
    "shared_end_in_loop",
    "five_player_shared",
    "full_loop",
    "full_loop_forced",
    "full_loop_shared",
    "other_full_loop",
    "partial_loop",
    "dead_one_end",
    "sequential",
]
ALL_ACTIONS = {encode_extend(e, s): (e, s) for e in (0, 1) for s in range(len(B.streets))}


# ---- independent oracle (spec text only) --------------------------------------------------------


def inc(v: int) -> set[int]:
    return {i for i, s in enumerate(B.streets) if v in (s.a, s.b)}


def far(street: int, v: int) -> int:
    s = B.streets[street]
    assert v in (s.a, s.b)
    return s.b if v == s.a else s.a


def walk(line: tuple[int, ...]) -> list[int]:
    s0 = B.streets[line[0]]
    if len(line) == 1:
        return [s0.a, s0.b]
    s1 = B.streets[line[1]]
    shared = ({s0.a, s0.b} & {s1.a, s1.b}).pop()
    nodes = [far(line[0], shared), shared]
    for st in line[1:]:
        nodes.append(far(st, nodes[-1]))
    return nodes


def ends_info(line: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """(end index, node, arrival street) for end 0 and end 1."""
    n = walk(line)
    return [(0, n[0], line[0]), (1, n[-1], line[-1])]


def geo(lines: LinesT, seat: int) -> set[tuple[int, int]]:
    """R-036..R-039, R-041 (no resolution restriction)."""
    line = lines[seat]
    occ = {s for ln in lines for s in ln}
    out: set[tuple[int, int]] = set()
    for e, v, arr in ends_info(line):
        cands = inc(v) - set(line)
        unocc = cands - occ
        allowed = set(unocc) if unocc else set(cands)
        for o, other in enumerate(lines):
            if o == seat or not other:
                continue
            for _oe, ov, oarr in ends_info(other):
                if ov == v and oarr != arr and oarr in cands:
                    allowed.add(oarr)
        out |= {(e, s) for s in allowed}
    return out


def ext(line: tuple[int, ...], e: int, s: int) -> tuple[int, ...]:
    return (s, *line) if e == 0 else (*line, s)


def with_line(lines: LinesT, seat: int, line: tuple[int, ...]) -> LinesT:
    return tuple(line if i == seat else ln for i, ln in enumerate(lines))


def best_values(lines: LinesT, seat: int, r: int, supply: int) -> dict[tuple[int, int], int]:
    """best(e, s) = 1 + max further placements (<= r - 1, bounded by supply), every geo choice."""
    if r <= 0 or supply <= 0:
        return {}
    out = {}
    for e, s in geo(lines, seat):
        nl = with_line(lines, seat, ext(lines[seat], e, s))
        out[(e, s)] = 1 + max_placements(nl, seat, r - 1, supply - 1)
    return out


def max_placements(lines: LinesT, seat: int, r: int, supply: int) -> int:
    return max(best_values(lines, seat, r, supply).values(), default=0)


def supply_of(lines: LinesT, seat: int) -> int:
    return BUS_LINE_MARKERS - len(lines[seat])


def restricted(lines: LinesT, seat: int, r: int, supply: int | None = None) -> set[tuple[int, int]]:
    """R-042: only choices whose best equals the maximum over all choices."""
    sup = supply_of(lines, seat) if supply is None else supply
    vals = best_values(lines, seat, r, sup)
    if not vals:
        return set()
    top = max(vals.values())
    return {k for k, v in vals.items() if v == top}


def ids(choices: set[tuple[int, int]]) -> set[int]:
    return {encode_extend(e, s) for e, s in choices}


# ---- state helpers ----------------------------------------------------------------------


def doc(name: str, units: int | None = None) -> dict[str, Any]:
    d = H.load_doc(f"lines/{name}")
    if units is not None:
        d["pending"]["units_left"] = units
    return d


def load(name: str, units: int | None = None) -> Any:
    return H.load_doc_state(doc(name, units))


def lines_of_state(s: Any) -> LinesT:
    return tuple(tuple(s.seat(p).line) for p in range(s.num_players()))


def lines_of_doc(d: dict[str, Any]) -> LinesT:
    return tuple(tuple(x["line"]) for x in d["seats"])


def chosen(action: int) -> tuple[int, int]:
    return ALL_ACTIONS[action]


def in_marker(s: Any, letter: int) -> bool:
    return s.phase == "LN" and s.cursor_letter == letter


def leaves(s: Any, letter: int, cap: int = 3) -> list[Any]:
    """States reached when the marker at ``letter`` stops asking (<= cap choices per node)."""
    if not in_marker(s, letter):
        return [s]
    out: list[Any] = []
    for a in s.legal_actions()[:cap]:
        c = s.clone()
        c.apply_action(a)
        out += leaves(c, letter, cap)
    return out


def extend_str(seat: int, line: tuple[int, ...], e: int, s: int) -> str:
    """ACTIONS 4: 'P{seat}: extend end e via s# (na-nb) to n<new end>'."""
    st = B.streets[s]
    end_node = walk(line)[0 if e == 0 else -1]
    return f"P{seat}: extend end {e} via s{s} (n{st.a}-n{st.b}) to n{far(s, end_node)}"


# ---- converting a fixture into a "last decision of selection" document ---------------------------

LINES, BUSES, PASSENGERS, BUILDINGS, CLOCK, VRROOOMM, STARTING = range(7)
NAMES = ["lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player"]
SPACES = [6, 1, 6, 6, 1, 6, 1]


def to_sel(
    base: dict[str, Any],
    board: dict[int, list[int]],
    buses: dict[int, int] | None = None,
) -> dict[str, Any]:
    """SEL document where one seat is left to act: its DROP_OUT enters Phase 2 (R-028).

    ``board`` maps an action index to the seats on letters a, b, ...; ``buses`` maps seat to its
    bus count (others 1).  Seats short of two markers are padded on Starting Player, Vrrooomm!,
    Clock (they resolve after every action the tests below look at)."""
    d = copy.deepcopy(base)
    for k in ("expect", "current_player", "description"):
        d.pop(k, None)
    n = len(d["seats"])
    lists = {a: list(board.get(a, [])) for a in range(7)}
    count = [0] * n
    for seats in lists.values():
        for p in seats:
            count[p] += 1
    for p in range(n):
        for a in (STARTING, VRROOOMM, CLOCK):
            while count[p] < 2 and len(lists[a]) < SPACES[a]:
                lists[a].append(p)
                count[p] += 1
    spent = {p for p in range(n) if count[p] < 2}
    last = min((p for p in range(n) if p not in spent), key=lambda p: (count[p], p))
    d["phase"] = "SEL"
    d["round"] = 1
    d["action_board"] = {
        NAMES[a]: lists[a] + [None] * (SPACES[a] - len(lists[a])) for a in range(7)
    }
    d["pending"] = {
        "pass": count[last],
        "dropped": [p for p in range(n) if p != last and p not in spent],
    }
    for p in range(n):
        d["seats"][p]["markers_left"] = 0 if p in spent else 20 - count[p]
        d["seats"][p]["buses"] = (buses or {}).get(p, 1)
    return d


def enter(d: dict[str, Any]) -> Any:
    s = H.load_doc_state(d)
    assert s.phase == "SEL"
    assert s.legal_actions().count(DROP_OUT) == 1
    s.apply_action(DROP_OUT)
    return s


def first_choice(lines: LinesT, seat: int, r: int) -> tuple[int, int]:
    """The choice the lowest action id picks among the oracle's legal set."""
    return min(restricted(lines, seat, r), key=lambda c: encode_extend(*c))


# ===== fixture hygiene =====


@pytest.mark.parametrize(
    "name",
    [*RES_FIXTURES, "res_two_markers_drop", "res_two_markers_widen", "res_line_25"],
)
def test_res_fixture_loads_and_lines_are_paths(name: str) -> None:
    """Fixture hygiene: every new res_*.json loads and its lines are paths of distinct streets."""
    H.load_fixture(f"lines/{name}")
    for ln in lines_of_doc(doc(name)):
        assert len(set(ln)) == len(ln)
        walk(ln)


def test_oracle_sanity_expected_values_of_the_new_fixtures() -> None:
    """Oracle self-check on the fixtures' intended shape (not a rule test): dead_end_power3 has one
    live choice and top 2; bite_dead_end mixes best values 1 and 2; supply_one_left bounds to 1."""
    d = doc("res_dead_end_power3")
    ls = lines_of_doc(d)
    vals = best_values(ls, 0, 3, supply_of(ls, 0))
    assert len(geo(ls, 0)) == 1
    assert set(vals.values()) == {2}
    d = doc("res_bite_dead_end")
    ls = lines_of_doc(d)
    assert set(best_values(ls, 0, 2, supply_of(ls, 0)).values()) == {1, 2}
    d = doc("res_bite_short")
    ls = lines_of_doc(d)
    assert set(best_values(ls, 0, 3, supply_of(ls, 0)).values()) == {1, 2}
    d = doc("res_supply_one_left")
    ls = lines_of_doc(d)
    assert supply_of(ls, 0) == 1
    assert set(best_values(ls, 0, 2, 25).values()) == {1, 2}
    assert set(best_values(ls, 0, 2, 1).values()) == {1}


# ===== R-040 example =====


def test_R040_example_power3_two_placements_then_third_is_lost() -> None:
    """R-040 example: power 3, the line's only live end allows exactly 2 further placements.  Two
    EXTEND nodes happen (units_left 3 then 2); the third unit is lost: no node, resolution of the
    marker ends."""
    s = load("res_dead_end_power3")
    ls = lines_of_state(s)
    start_len = len(ls[0])
    start_hist = len(s.history())
    assert s.phase == "LN" and s.current_player() == 0 and s.units_left == 3
    # forced node (one legal action) is still emitted (ACTIONS 1.5)
    assert len(s.legal_actions()) == 1
    assert set(s.legal_actions()) == ids(restricted(ls, 0, 3))
    s.apply_action(s.legal_actions()[0])
    assert in_marker(s, 0) and s.units_left == 2
    assert len(s.seat(0).line) == start_len + 1
    assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), 0, 2))
    s.apply_action(s.legal_actions()[0])
    assert len(s.seat(0).line) == start_len + 2
    assert len(s.history()) == start_hist + 2  # exactly two decision nodes for this marker
    assert not in_marker(s, 0)  # third unit lost: this was the only Lines marker
    assert s.seat(0).line_markers == BUS_LINE_MARKERS - start_len - 2


def test_R040_dead_end_third_unit_lost_has_no_node_even_when_more_markers_follow() -> None:
    """R-040/R-042: the lost unit leaves no node and resolution moves straight to the next marker
    (here a Passengers marker of another seat), whatever the leftover power."""
    d = to_sel(doc("res_dead_end_power3"), {LINES: [0], PASSENGERS: [1]}, buses={2: 3})
    s = enter(d)
    assert s.phase == "LN" and s.units_left == 3  # full power on entry (SCENARIOS 9.8)
    s.apply_action(s.legal_actions()[0])
    s.apply_action(s.legal_actions()[0])
    assert s.phase == "PX" and s.current_player() == 1


def test_R040_both_ends_dead_marker_produces_no_node() -> None:
    """R-040: both ends dead -> every extension of the marker is lost; no LN node at all, the
    next marker's node comes first."""
    base = H.load_doc("lines/dead_both_ends")
    d = to_sel(base, {LINES: [0], PASSENGERS: [1]}, buses={2: 2})
    s = enter(d)
    assert s.phase == "PX" and s.current_player() == 1


def test_R040_no_bus_line_markers_left_marker_produces_no_node() -> None:
    """R-040: the player has no Bus Line markers left (25 on the board): remaining extensions are
    lost although the line has live ends; the marker produces no node."""
    base = doc("res_line_25")
    ls = lines_of_doc(base)
    assert len(ls[0]) == BUS_LINE_MARKERS
    assert geo(ls, 0)  # geometry alone would allow an extension
    s = enter(to_sel(base, {LINES: [0], PASSENGERS: [1]}, buses={2: 2}))
    assert s.phase == "PX" and s.current_player() == 1
    assert len(s.seat(0).line) == BUS_LINE_MARKERS


def test_R040_dead_marker_is_skipped_and_the_next_lines_marker_resolves() -> None:
    """R-040/R-030: Lines b (seat 0, both ends dead) resolves first and is lost, so the first node
    is Lines a (seat 1) with its full power 2 (M = 2)."""
    base = H.load_doc("lines/dead_both_ends")
    s = enter(to_sel(base, {LINES: [1, 0]}, buses={2: 2}))
    assert (s.phase, s.cursor_letter, s.current_player(), s.units_left) == ("LN", 0, 1, 2)


# ===== R-042 maximal resolution =====


@pytest.mark.parametrize("name", RES_FIXTURES)
def test_R042_legal_actions_equal_the_maximal_resolution_oracle(name: str) -> None:
    """R-042 / ACTIONS 3.5 items 1-4: at the first unit, legal_actions() is exactly the encoded
    (end, street) pairs whose best equals the maximum, with the supply bound applied."""
    s = load(name)
    ls = lines_of_state(s)
    r = s.units_left
    assert set(s.legal_actions()) == ids(restricted(ls, 0, r))


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("name", GEOMETRY_FIXTURES)
def test_R042_older_geometry_fixtures_with_more_units(name: str, r: int) -> None:
    """R-042: the geometry fixtures with units_left raised to 2 or 3 still match the maximal
    resolution oracle (per-end rule, shared-end exception, loops included)."""
    s = load(name, r)
    assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), s.current_player(), r))


@pytest.mark.parametrize("name", GEOMETRY_FIXTURES)
def test_R042_one_unit_left_is_never_restricted(name: str) -> None:
    """R-042: with one unit left every geometrically legal choice places it, so the restriction
    never bites at units_left 1 (the 15 geometry-wiring tests rely on this): legal == geometry."""
    s = load(name, 1)
    ls = lines_of_state(s)
    assert set(s.legal_actions()) == ids(geo(ls, s.current_player()))
    assert set(s.legal_actions()) == ids(restricted(ls, s.current_player(), 1))


@pytest.mark.parametrize("name", ["res_bite_dead_end", "res_bite_both_ends"])
def test_R042_choice_that_dead_ends_after_one_placement_is_not_legal(name: str) -> None:
    """R-042 example: power 2; one geometric choice leaves no second placement, the others allow
    two.  The dead-ending choice is NOT legal, the others are."""
    s = load(name)
    ls = lines_of_state(s)
    vals = best_values(ls, 0, 2, supply_of(ls, 0))
    losers = {k for k, v in vals.items() if v == 1}
    winners = {k for k, v in vals.items() if v == 2}
    assert losers and winners  # fixture sanity: the restriction really bites
    legal = set(s.legal_actions())
    assert legal.isdisjoint(ids(losers))
    assert legal == ids(winners)
    assert legal <= ids(geo(ls, 0))  # restriction only removes choices, never adds


def test_R042_equal_best_choices_are_all_legal() -> None:
    """R-042: choices with the same (maximal) best are all legal, not one canonical choice:
    on this line all six choices reach 2 placements."""
    s = load("res_equal_best")
    legal = set(s.legal_actions())
    assert len(legal) >= 3
    assert legal == ids(geo(lines_of_state(s), 0))
    ends = {chosen(a)[0] for a in legal}
    assert ends == {0, 1}  # both ends offered


def test_R042_two_equal_best_choices_survive_next_to_a_dropped_one() -> None:
    """R-042: in res_bite_both_ends two choices (one per end) reach 2 and one reaches 1: exactly
    the two best are legal."""
    s = load("res_bite_both_ends")
    legal = {chosen(a) for a in s.legal_actions()}
    assert len(legal) == 2
    assert {e for e, _ in legal} == {0, 1}
    assert len(geo(lines_of_state(s), 0)) == 3


def test_R042_power3_dead_end_choice_excluded_when_others_reach_three() -> None:
    """R-042: power 3 on the same line: the choice that stops after one placement is excluded;
    the two others (each reaching 3) remain."""
    s = load("res_bite_power3")
    ls = lines_of_state(s)
    vals = best_values(ls, 0, 3, supply_of(ls, 0))
    assert sorted(vals.values()) == [1, 3, 3]
    assert set(s.legal_actions()) == ids({k for k, v in vals.items() if v == 3})


def test_R042_when_no_choice_reaches_r_those_reaching_the_maximum_are_legal() -> None:
    """R-042 / ACTIONS 3.5: power 3 but no sequence places 3 (maximum 2).  Choices with best 2 are
    legal; the choice with best 1 is not."""
    s = load("res_bite_short")
    ls = lines_of_state(s)
    vals = best_values(ls, 0, 3, supply_of(ls, 0))
    assert max(vals.values()) == 2 and min(vals.values()) == 1
    legal = set(s.legal_actions())
    assert legal == ids({k for k, v in vals.items() if v == 2})
    assert legal.isdisjoint(ids({k for k, v in vals.items() if v == 1}))


@pytest.mark.parametrize("name", RES_FIXTURES)
def test_R042_every_legal_path_places_the_maximum_number_of_markers(name: str) -> None:
    """R-040/R-042: whatever legal choices the owner makes, the marker places exactly the maximal
    achievable number of streets (at most power, at most the supply)."""
    s = load(name)
    ls = lines_of_state(s)
    r, start = s.units_left, len(ls[0])
    top = max_placements(ls, 0, r, supply_of(ls, 0))
    finals = leaves(s, s.cursor_letter)
    assert finals
    for f in finals:
        assert len(f.seat(0).line) == start + top
        assert f.seat(0).line_markers == BUS_LINE_MARKERS - start - top


def test_R042_unrestricted_play_could_place_fewer_the_restriction_is_what_prevents_it() -> None:
    """R-042 (guard): on res_bite_dead_end an unrestricted choice sequence can place only 1 street
    (best value 1 exists); the engine's legal set never offers that sequence."""
    s = load("res_bite_dead_end")
    ls = lines_of_state(s)
    assert 1 in best_values(ls, 0, 2, supply_of(ls, 0)).values()
    assert s.legal_actions()
    for f in leaves(s, 0, cap=10):
        assert len(f.seat(0).line) == len(ls[0]) + 2


# ===== bound by Bus Line markers left =====


def test_R042_one_marker_left_bounds_best_so_all_geometric_choices_are_legal() -> None:
    """R-040/R-042: 24 streets in the line (1 marker left), power 2.  Without the supply bound
    some choices would be dropped (best 1 vs 2); with it every choice places the one possible
    street, so all geometric choices are legal."""
    s = load("res_supply_one_left")
    ls = lines_of_state(s)
    assert s.seat(0).line_markers == 1
    assert restricted(ls, 0, 2, supply=25) != geo(ls, 0)  # unbounded would bite
    assert set(s.legal_actions()) == ids(geo(ls, 0))
    s.apply_action(s.legal_actions()[0])
    assert len(s.seat(0).line) == BUS_LINE_MARKERS
    assert s.seat(0).line_markers == 0
    assert not in_marker(s, 0)  # second unit lost: no node


def test_R042_two_markers_left_power4_places_two_and_best_uses_that_bound() -> None:
    """R-040/R-042: 2 markers left, power 4: exactly 2 nodes, then the other 2 units are lost.
    The first-unit legal set is computed with the bound 2, which differs from the unbounded one."""
    s = load("res_supply_two_left")
    ls = lines_of_state(s)
    assert s.units_left == 4 and s.seat(0).line_markers == 2
    bounded = restricted(ls, 0, 4, supply=2)
    assert bounded != restricted(ls, 0, 4, supply=25)
    assert set(s.legal_actions()) == ids(bounded)
    start_hist = len(s.history())
    s.apply_action(s.legal_actions()[0])
    assert in_marker(s, 0) and s.units_left == 3
    assert s.seat(0).line_markers == 1
    assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), 0, 3))
    s.apply_action(s.legal_actions()[0])
    assert len(s.history()) == start_hist + 2
    assert len(s.seat(0).line) == BUS_LINE_MARKERS and s.seat(0).line_markers == 0
    assert not in_marker(s, 0)


def test_R042_supply_is_read_per_seat_other_seats_markers_do_not_count() -> None:
    """R-042: the supply is the acting seat's own (25 minus its line); the three other short lines
    do not reduce it."""
    s = load("res_supply_two_left")
    assert [s.seat(p).line_markers for p in range(1, 4)] == [23, 23, 23]
    assert s.seat(0).line_markers == 2
    assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), 0, 4))


# ===== each EXTEND =====


@pytest.mark.parametrize("name", ["res_equal_best", "res_bite_both_ends", "res_bite_power3"])
def test_R042_each_extend_spends_a_unit_a_marker_and_grows_the_line(name: str) -> None:
    """R-036/R-042: every EXTEND decrements units_left, spends one Bus Line marker, and puts the
    street at the chosen end of the stored path (end 0 prepends, end 1 appends)."""
    s0 = load(name)
    ls0 = lines_of_state(s0)
    r = s0.units_left
    assert s0.legal_actions()
    for a in s0.legal_actions():
        e, st = chosen(a)
        s = s0.clone()
        s.apply_action(a)
        assert tuple(s.seat(0).line) == ext(ls0[0], e, st)
        assert s.seat(0).line_markers == s0.seat(0).line_markers - 1
        assert in_marker(s, 0) and s.units_left == r - 1 and s.current_player() == 0
        for p in (1, 2, 3):  # nobody else changes
            assert tuple(s.seat(p).line) == ls0[p]


# ===== O-2 sequential evaluation =====


@pytest.mark.parametrize("name", ["res_equal_best", "res_bite_both_ends", "res_bite_short"])
def test_R042_O2_next_unit_is_evaluated_with_the_street_just_placed(name: str) -> None:
    """O-2 / R-038 / R-042: after unit k the legal set of unit k+1 equals the oracle on the line
    that includes the street placed in unit k (new end, placed street never offered again)."""
    s0 = load(name)
    r = s0.units_left
    assert s0.legal_actions()
    for a in s0.legal_actions():
        e, st = chosen(a)
        s = s0.clone()
        s.apply_action(a)
        ls = lines_of_state(s)
        assert s.legal_actions()
        assert st in ls[0]
        legal = set(s.legal_actions())
        assert legal == ids(restricted(ls, 0, r - 1))
        assert all(chosen(x)[1] != st for x in legal)  # R-037: never repeat an own street


def test_R042_O2_new_end_candidates_follow_the_end_that_was_extended() -> None:
    """R-036/O-2: after extending end 1 via street x, the next unit's end-1 choices start at x's far
    node; end 0 choices still start at the old end 0 node (the first line has two ends)."""
    s0 = load("res_equal_best")
    ls0 = lines_of_state(s0)
    nodes0 = walk(ls0[0])
    for a in s0.legal_actions():
        e, st = chosen(a)
        if e != 1:
            continue
        s = s0.clone()
        s.apply_action(a)
        new_end = far(st, nodes0[-1])
        for x in s.legal_actions():
            ee, ss = chosen(x)
            v = new_end if ee == 1 else nodes0[0]
            assert v in (B.streets[ss].a, B.streets[ss].b)
        assert tuple(walk(lines_of_state(s)[0])[-1:]) == (new_end,)
        break
    else:
        pytest.fail("fixture has no end-1 choice")


def test_R042_O2_other_seat_sees_street_just_placed_as_occupied() -> None:
    """O-2 / R-038 (sequential evaluation, other seat): seat 1 (Lines b) resolves before seat 0
    (Lines a).  For each of seat 1's choices, seat 0's options at its node afterwards equal the
    oracle on the updated lines; in at least one case the street seat 1 placed is dropped from
    seat 0's options because it is now occupied and a free candidate remains."""
    d = doc("res_two_markers_drop")
    s0 = H.load_doc_state(d)
    assert (s0.phase, s0.cursor_letter, s0.current_player(), s0.units_left) == ("LN", 1, 1, 1)
    # seat 0's options before seat 1 acts: same position, cursor on seat 0's marker
    d_before = copy.deepcopy(d)
    d_before["pending"] = {"cursor": {"letter": 0}, "units_left": 2}
    before = set(H.load_doc_state(d_before).legal_actions())
    assert before  # seat 0 has options
    dropped_somewhere = False
    for a in s0.legal_actions():
        e, st = chosen(a)
        s = s0.clone()
        s.apply_action(a)
        assert (s.phase, s.cursor_letter, s.current_player()) == ("LN", 0, 0)
        assert s.units_left == 2  # full power of letter a on entry (M = 2)
        ls = lines_of_state(s)
        assert st in ls[1]
        legal = set(s.legal_actions())
        assert legal == ids(restricted(ls, 0, 2))
        dropped = {x for x in before if chosen(x)[1] == st} - legal
        dropped_somewhere = dropped_somewhere or bool(dropped)
        if dropped:  # the placed street is occupied now, so it is no longer an option for seat 0
            assert all(chosen(x)[1] != st for x in legal if x in dropped)
    assert dropped_somewhere


def test_R042_O2_other_seat_gains_options_from_the_new_end_of_the_placed_street() -> None:
    """O-2 / R-039 / R-038: seat 1's earlier placement can also ADD options for seat 0 (its new end
    or the all-occupied widening).  In res_two_markers_widen one of seat 1's choices makes a street
    legal for seat 0 that was not legal before; the state matches the oracle in every case."""
    d = doc("res_two_markers_widen")
    s0 = H.load_doc_state(d)
    d_before = copy.deepcopy(d)
    d_before["pending"] = {"cursor": {"letter": 0}, "units_left": 2}
    before = set(H.load_doc_state(d_before).legal_actions())
    gained_somewhere = False
    for a in s0.legal_actions():
        s = s0.clone()
        s.apply_action(a)
        legal = set(s.legal_actions())
        assert legal == ids(restricted(lines_of_state(s), 0, 2))
        gained_somewhere = gained_somewhere or bool(legal - before)
    assert gained_somewhere


def test_R042_O2_placed_street_alone_does_not_reach_back_into_the_owners_earlier_state() -> None:
    """O-2: the sequence is strictly one street at a time: before seat 1 acts, seat 0's legal set
    does not yet contain anything caused by seat 1's pending marker (checked by loading the same
    board with the cursor on seat 0: it equals the oracle on the unmodified lines)."""
    d = doc("res_two_markers_drop")
    d["pending"] = {"cursor": {"letter": 0}, "units_left": 2}
    s = H.load_doc_state(d)
    assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), 0, 2))


# ===== marker order, hand-over =====


def test_R042_after_last_unit_cursor_moves_to_next_lines_marker_highest_letter_first() -> None:
    """R-030/R-042: with Lines a (seat 0), b (seat 1) the order is b then a; after the last unit
    of b the cursor is on letter 0 owned by seat 0 with its full power."""
    s = load("res_two_markers_drop")
    assert (s.cursor_letter, s.current_player(), s.units_left) == (1, 1, 1)
    s.apply_action(s.legal_actions()[0])
    assert (s.phase, s.cursor_letter, s.current_player(), s.units_left) == ("LN", 0, 0, 2)
    s.apply_action(s.legal_actions()[0])
    assert (s.cursor_letter, s.current_player(), s.units_left) == (0, 0, 1)


def test_R042_chain_of_three_lines_markers_then_buses_then_next_action() -> None:
    """R-029/R-030/R-042: Lines a (seat 0), b (seat 1), c (seat 2) with M = 3 resolve c (1 unit),
    b (2), a (3), nodes in that order, each completely before the next; then Buses (seat 3: 1 -> 4
    buses), then Passengers (seat 1).  Placements are followed with the oracle's first choice."""
    base = doc("res_two_markers_drop")
    d = to_sel(base, {LINES: [0, 1, 2], BUSES: [3], PASSENGERS: [1]}, buses={3: 3})
    s = enter(d)
    seen = []
    while s.phase == "LN":
        seat, r = s.current_player(), s.units_left
        seen.append((s.cursor_letter, seat, r))
        ls = lines_of_state(s)
        assert set(s.legal_actions()) == ids(restricted(ls, seat, r))
        s.apply_action(encode_extend(*first_choice(ls, seat, r)))
    assert seen == [(2, 2, 1), (1, 1, 2), (1, 1, 1), (0, 0, 3), (0, 0, 2), (0, 0, 1)]
    assert s.phase == "PX" and s.current_player() == 1
    assert s.seat(3).buses == 4  # Buses ran between Lines and Passengers (R-043)


def test_R042_power_is_computed_when_the_marker_starts_full_units_on_entry() -> None:
    """SCENARIOS 9.8 / R-031: on entering a marker units_left is its full power even when fewer
    placements are possible (dead end line, M = 3)."""
    d = to_sel(doc("res_dead_end_power3"), {LINES: [0]}, buses={1: 3})
    s = enter(d)
    assert (s.phase, s.cursor_letter, s.current_player(), s.units_left) == ("LN", 0, 0, 3)


def test_R042_five_player_power_plus_one_reaches_six_units() -> None:
    """R-033/R-042: 5 players, M = 5, Lines a: power 6.  Six EXTEND nodes in a row (units_left 6
    down to 1), one Bus Line marker each, then the marker is done.  (Checked step by step against
    the geometry set; the 6-deep oracle is skipped for speed.)"""
    base = H.load_doc("lines/five_player_shared")
    s = enter(to_sel(base, {LINES: [0]}, buses={1: 5}))
    start = len(s.seat(0).line)
    assert (s.phase, s.cursor_letter, s.current_player(), s.units_left) == ("LN", 0, 0, 6)
    for k in range(6, 0, -1):
        assert in_marker(s, 0) and s.units_left == k
        assert s.seat(0).line_markers == BUS_LINE_MARKERS - start - (6 - k)
        legal = s.legal_actions()
        assert legal
        assert set(legal) <= ids(geo(lines_of_state(s), 0))  # restriction only removes
        s.apply_action(legal[0])
    assert len(s.seat(0).line) == start + 6
    assert not in_marker(s, 0)


# ===== loops =====


@pytest.mark.parametrize("name", ["full_loop", "full_loop_forced", "full_loop_shared"])
def test_R041_R042_full_loop_end_choice_under_resolution(name: str) -> None:
    """R-041/R-042: for a full-line loop (both ends at one node) both ends are separate legal
    choices at every unit; extending end 0 prepends, end 1 appends, giving different stored paths;
    the following unit matches the oracle for each of them."""
    s0 = load(name, 2)
    ls0 = lines_of_state(s0)
    e0, e1 = ends_info(ls0[0])
    assert e0[1] == e1[1]  # both ends at one node
    legal = {chosen(a) for a in s0.legal_actions()}
    assert legal == restricted(ls0, 0, 2)
    assert {e for e, _ in legal} == {0, 1}
    by_street: dict[int, set[int]] = {}
    for e, st in legal:
        by_street.setdefault(st, set()).add(e)
    st, es = next((st, es) for st, es in sorted(by_street.items()) if es == {0, 1})
    results = []
    for e in (0, 1):
        s = s0.clone()
        s.apply_action(encode_extend(e, st))
        line = tuple(s.seat(0).line)
        assert line == ext(ls0[0], e, st)
        assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), 0, 1))
        results.append(line)
    assert results[0] != results[1]


def test_R041_R042_partial_loop_ends_stay_distinct_through_two_units() -> None:
    """R-035/R-041/R-042: a line that loops in the middle keeps two separate ends over two units;
    each of the first unit's choices leads to the oracle's second-unit set."""
    s0 = load("partial_loop", 2)
    assert s0.legal_actions()
    for a in s0.legal_actions():
        s = s0.clone()
        s.apply_action(a)
        assert set(s.legal_actions()) == ids(restricted(lines_of_state(s), 0, 1))


def test_R042_dead_one_end_with_power2_forced_chain_keeps_dead_end_out_of_the_legal_set() -> None:
    """R-040/R-042: dead_one_end (end 1 dead) with 2 units: every legal action is on end 0."""
    s = load("dead_one_end", 2)
    assert {chosen(a)[0] for a in s.legal_actions()} == {0}


# ===== must NOT be legal =====


@pytest.mark.parametrize("name", [*RES_FIXTURES, *GEOMETRY_FIXTURES[:6]])
def test_R042_never_legal_own_street_non_candidate_or_second_setup_style_choices(
    name: str,
) -> None:
    """R-037/R-042: no legal EXTEND uses a street already in the line or a street not incident to
    that end's node, and (e, s) pairs with e not in {0, 1} do not exist; nothing outside the
    geometric set is offered."""
    s = load(name)
    ls = lines_of_state(s)
    seat = s.current_player()
    nodes = walk(ls[seat])
    legal = s.legal_actions()
    assert legal
    for a in legal:
        e, st = chosen(a)
        assert st not in ls[seat]
        v = nodes[0] if e == 0 else nodes[-1]
        assert v in (B.streets[st].a, B.streets[st].b)
    assert set(legal) <= ids(geo(ls, seat))


def test_R042_non_extend_actions_are_never_legal_in_ln() -> None:
    """R-042: in LN only EXTEND actions are legal (no select, drop out, build, clock...)."""
    s = load("res_equal_best")
    assert s.legal_actions()
    assert set(s.legal_actions()) <= set(ALL_ACTIONS)
    assert DROP_OUT not in s.legal_actions()


def test_R042_restricted_out_choice_is_not_in_legal_actions_but_is_geometrically_valid() -> None:
    """R-042: the excluded dead-end choice is a valid geometry extension (bus.lines would allow
    it) yet absent from legal_actions(): the restriction is what removes it."""
    s = load("res_bite_dead_end")
    ls = lines_of_state(s)
    vals = best_values(ls, 0, 2, supply_of(ls, 0))
    (loser,) = [k for k, v in vals.items() if v == 1]
    assert loser in geo(ls, 0)
    assert s.legal_actions()
    assert encode_extend(*loser) not in s.legal_actions()


# ===== action_to_string =====


@pytest.mark.parametrize("name", ["res_equal_best", "res_bite_both_ends", "full_loop"])
def test_R042_extend_action_string_format_in_ln(name: str) -> None:
    """ACTIONS 4: an LN EXTEND reads 'P{seat}: extend end e via s# (na-nb) to n<new end>', with
    the street's endpoints in board a-b order and the new end node of the extended end."""
    s = load(name, 2)
    ls = lines_of_state(s)
    assert s.legal_actions()
    for a in s.legal_actions():
        e, st = chosen(a)
        assert s.action_to_string(s.current_player(), a) == extend_str(
            s.current_player(), ls[s.current_player()], e, st
        )


def test_R042_extend_action_string_uses_acting_seat_label_p1() -> None:
    """ACTIONS 4 / CLAUDE.md: seats are labelled from P0; seat 1's Lines marker reads 'P1: ...'."""
    s = load("res_two_markers_drop")
    assert s.current_player() == 1
    a = s.legal_actions()[0]
    text = s.action_to_string(1, a)
    assert text.startswith("P1: extend end ")
    assert text == extend_str(1, lines_of_state(s)[1], *chosen(a))


# ===== determinism / property =====


def test_R042_random_playouts_of_all_new_fixtures_match_oracle_every_unit() -> None:
    """R-040/R-042: random legal play over every res_* fixture (seeded): at every unit the legal set
    equals the oracle's, units_left falls by one per EXTEND, and the number of nodes equals the
    maximal placement count."""
    rng = random.Random(42)
    for name in RES_FIXTURES:
        for _ in range(4):
            s = load(name)
            ls = lines_of_state(s)
            r = s.units_left
            expected_nodes = max_placements(ls, 0, r, supply_of(ls, 0))
            nodes = 0
            while in_marker(s, 0):
                cur = lines_of_state(s)
                assert set(s.legal_actions()) == ids(restricted(cur, 0, s.units_left))
                before = s.units_left
                s.apply_action(rng.choice(s.legal_actions()))
                nodes += 1
                if in_marker(s, 0):
                    assert s.units_left == before - 1
            assert nodes == expected_nodes
