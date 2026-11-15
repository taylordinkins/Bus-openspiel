"""Line geometry (R-035..R-039, R-041), pure functions of ``bus.lines`` (Phase 1 unit 4).

Expected API
  line_nodes(board, line) -> list[int]       node walk, len(line) + 1 nodes (ACTIONS 1.3)
  line_ends(board, line) -> (int, int)       node of end 0, node of end 1
  legal_extensions(board, lines, seat) -> frozenset[(end, street)]
                                             R-036..R-039, R-041 given every seat's ordered line;
                                             an empty line gives frozenset() (first marker is R-017)
  extend(board, line, end, street) -> tuple  end 0 prepends, end 1 appends; ValueError if the
                                             street is not incident to that end's node or is
                                             already in line

Positions are the JSON fixtures in tests/scenarios/lines/ (lines are read from the fixtures;
concrete street ids never appear in this file).  ``ref_legal`` below is a small independent
oracle written from the spec text; each test also asserts role-based expectations (named streets of
the fixture's lines), so a shared misreading would still have to survive those.

Out of scope here: dead ends beyond "no candidate" (R-040), maximal resolution (R-042), setup flow.
"""

from __future__ import annotations

from typing import Any

import pytest

from bus.board import load_board
from tests import harness as H

B = load_board()

FIXTURES = [
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
    "dead_both_ends",
    "sequential",
]


def _L() -> Any:
    from bus import lines

    return lines


def doc(name: str) -> dict[str, Any]:
    return H.load_doc(f"lines/{name}")


def lines_of(name: str) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(s["line"]) for s in doc(name)["seats"])


# ---- tiny spec-derived helpers (independent of bus.lines) --------------------------------------


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


def occupied(lines: tuple[tuple[int, ...], ...]) -> set[int]:
    return {s for ln in lines for s in ln}


def ref_legal(lines: tuple[tuple[int, ...], ...], seat: int) -> set[tuple[int, int]]:
    line = lines[seat]
    if not line:
        return set()
    occ = occupied(lines)
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


def legal(name: str, seat: int = 0) -> set[tuple[int, int]]:
    return set(_L().legal_extensions(B, lines_of(name), seat))


def streets_at(le: set[tuple[int, int]], e: int) -> set[int]:
    return {s for (ee, s) in le if ee == e}


def end_at_node(line: tuple[int, ...], v: int) -> int:
    es = [e for e, n, _ in ends_info(line) if n == v]
    assert len(es) == 1, "helper expects exactly one end at node"
    return es[0]


def shared_end_node(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    common = {n for _, n, _ in ends_info(a)} & {n for _, n, _ in ends_info(b)}
    assert len(common) == 1
    return common.pop()


# ---- fixture sanity (not pending: the loader already exists) -----------------------------------


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_loads_and_lines_are_paths(name: str) -> None:
    """Fixture hygiene: the document loads and every line is a path of consecutive streets."""
    H.load_fixture(f"lines/{name}")
    for ln in lines_of(name):
        assert len(set(ln)) == len(ln)
        walk(ln)


# ============================== R-035 ==============================


def test_R035_single_street_line_ends_at_a_then_b() -> None:
    """R-035 / ACTIONS 1.3: a single-street line has two ends, one at each node; end 0 at `a`."""
    for name, seat in (("single_street_midline", 0), ("shared_end_single_street_other", 1)):
        (st,) = lines_of(name)[seat]
        s = B.streets[st]
        assert _L().line_nodes(B, (st,)) == [s.a, s.b]
        assert tuple(_L().line_ends(B, (st,))) == (s.a, s.b)
        assert s.a != s.b


@pytest.mark.parametrize("name", FIXTURES)
def test_R035_node_walk_has_len_plus_one_nodes_and_ends_are_its_extremes(name: str) -> None:
    """R-035: consecutive streets share a node; the walk has len+1 nodes; end 0 is the walk's start,
    end 1 its last node (path index 0 = end 0)."""
    for ln in lines_of(name):
        nodes = _L().line_nodes(B, ln)
        assert list(nodes) == walk(ln)
        assert len(nodes) == len(ln) + 1
        assert tuple(_L().line_ends(B, ln)) == (nodes[0], nodes[-1])


def test_R035_walk_starts_at_endpoint_of_first_street_not_shared_with_second() -> None:
    """R-035 / ACTIONS 1.3: the node walk starts at the endpoint of the first street that is not
    shared with the second street."""
    ln = lines_of("partial_loop")[0]
    s0, s1 = B.streets[ln[0]], B.streets[ln[1]]
    shared = ({s0.a, s0.b} & {s1.a, s1.b}).pop()
    nodes = _L().line_nodes(B, ln)
    assert nodes[1] == shared
    assert nodes[0] != shared


def test_R035_reversed_line_reverses_the_walk() -> None:
    """R-035: ends belong to the stored path: reversing a multi-street path swaps the ends."""
    for ln in lines_of("partial_loop")[0:1] + lines_of("shared_end")[1:3]:
        fwd = _L().line_nodes(B, ln)
        rev = _L().line_nodes(B, tuple(reversed(ln)))
        assert list(rev) == list(reversed(list(fwd)))


def test_R035_loop_passes_a_node_more_than_once_and_both_ends_are_real() -> None:
    """R-035: a line may pass through a node more than once; the stored path keeps both ends even
    when they coincide (full loop)."""
    full = lines_of("full_loop")[0]
    nodes = _L().line_nodes(B, full)
    e0, e1 = _L().line_ends(B, full)
    assert e0 == e1 == nodes[0] == nodes[-1]
    assert len(nodes) == len(full) + 1
    part = lines_of("partial_loop")[0]
    pn = _L().line_nodes(B, part)
    assert len(set(pn)) < len(pn)
    e0, e1 = _L().line_ends(B, part)
    assert e0 != e1
    interior = pn[1:-1]
    assert e0 not in interior and e1 not in interior


def test_R035_node_on_line_is_endpoint_of_any_street_of_the_line() -> None:
    """R-035: a node is on the line iff it is an endpoint of one of its streets (incl. mid)."""
    ln = lines_of("partial_loop")[0]
    on = set(_L().line_nodes(B, ln))
    assert on == {n for st in ln for n in (B.streets[st].a, B.streets[st].b)}
    mid = set(_L().line_nodes(B, ln)[1:-1]) - set(_L().line_ends(B, ln))
    assert mid and mid <= on


def test_R035_empty_line_has_no_extensions() -> None:
    """R-035/R-017: an empty line has no end, so legal_extensions is empty (setup is unit 5)."""
    ls = (tuple(), lines_of("full_loop")[0])
    assert _L().legal_extensions(B, ls, 0) == frozenset()


def test_R035_legal_extensions_returns_frozenset_of_int_pairs() -> None:
    """API shape: frozenset of (end, street) with end in {0, 1} and street a valid id."""
    r = _L().legal_extensions(B, lines_of("red_cd"), 0)
    assert isinstance(r, frozenset)
    assert r and all(e in (0, 1) and 0 <= s < len(B.streets) for e, s in r)


# ============================== R-036 ==============================


def test_R036_end0_prepends_end1_appends() -> None:
    """R-036 / ACTIONS 1.3: extending from end 0 prepends the street, from end 1 appends it."""
    for name in ("single_street_midline", "partial_loop", "full_loop"):
        ln = lines_of(name)[0]
        for e, s in sorted(legal(name)):
            new = _L().extend(B, ln, e, s)
            assert isinstance(new, tuple)
            assert new == ((s,) + ln if e == 0 else ln + (s,))


def test_R036_new_end_is_far_node_of_new_street_other_end_unchanged() -> None:
    """R-036: after placement that end is the far node of the new street; the other end stays."""
    for name in ("single_street_midline", "partial_loop", "shared_end"):
        ln = lines_of(name)[0]
        old = tuple(_L().line_ends(B, ln))
        for e, s in sorted(legal(name)):
            new = _L().extend(B, ln, e, s)
            ne = tuple(_L().line_ends(B, new))
            assert ne[e] == far(s, old[e])
            assert ne[1 - e] == old[1 - e]
            assert len(_L().line_nodes(B, new)) == len(ln) + 2


def test_R036_extend_does_not_mutate_input_and_accepts_list() -> None:
    """R-036: extend returns a new tuple; the input line is untouched."""
    ln = list(lines_of("partial_loop")[0])
    e, s = sorted(legal("partial_loop"))[0]
    before = list(ln)
    new = _L().extend(B, ln, e, s)
    assert ln == before and isinstance(new, tuple)


def test_R036_player_chooses_end_and_street() -> None:
    """R-036: both the end and the street are the player's choice: a line with two live ends offers
    pairs for both ends, several streets per end."""
    le = legal("single_street_midline")
    assert {e for e, _ in le} == {0, 1}
    le = legal("shared_end")
    assert all(len(streets_at(le, e)) >= 1 for e in (0, 1))
    assert any(len(streets_at(le, e)) >= 2 for e in (0, 1))


def test_R036_chained_extension_walks_the_ends() -> None:
    """R-036: two successive extensions at the same end grow the path by two streets, the second
    attached at the new far node."""
    name = "single_street_midline"
    ln = lines_of(name)[0]
    e0 = 0
    v0 = _L().line_ends(B, ln)[e0]
    cands = sorted(inc(v0) - set(ln))
    one = _L().extend(B, ln, e0, cands[0])
    v1 = _L().line_ends(B, one)[e0]
    assert v1 == far(cands[0], v0)
    two = _L().extend(B, one, e0, sorted(inc(v1) - set(one))[0])
    assert two[1] == cands[0] and two[2:] == ln
    assert len(_L().line_nodes(B, two)) == 4


def test_R036_extend_rejects_street_not_adjacent_to_the_end() -> None:
    """R-036: a street not incident to the chosen end's node cannot be added at that end.
    (API assumption: extend raises ValueError for geometry violations.)"""
    ln = lines_of("single_street_midline")[0]
    v = _L().line_ends(B, ln)[0]
    bad = sorted(set(range(len(B.streets))) - inc(v) - set(ln))[0]
    with pytest.raises(ValueError):
        _L().extend(B, ln, 0, bad)


def test_R036_extend_rejects_own_street() -> None:
    """R-036/R-037: a street already in the line cannot be added, even if incident to the end."""
    ln = lines_of("full_loop")[0]
    v = _L().line_ends(B, ln)[0]
    own_incident = sorted(inc(v) & set(ln))
    assert own_incident
    with pytest.raises(ValueError):
        _L().extend(B, ln, 0, own_incident[0])


def test_R036_not_legal_street_not_adjacent_to_any_end() -> None:
    """R-036: legal_extensions never offers a street that does not touch the chosen end."""
    for name in FIXTURES:
        L = lines_of(name)
        for seat in range(len(L)):
            ends = _L().line_ends(B, L[seat])
            for e, s in _L().legal_extensions(B, L, seat):
                assert ends[e] in (B.streets[s].a, B.streets[s].b)


# ============================== R-037 ==============================


@pytest.mark.parametrize("name", FIXTURES)
def test_R037_candidates_are_incident_and_not_own(name: str) -> None:
    """R-037: candidates are streets incident to the end node that the player does not already have;
    legal_extensions is a subset of them, for every seat."""
    L = lines_of(name)
    for seat in range(len(L)):
        ends = _L().line_ends(B, L[seat])
        for e, s in _L().legal_extensions(B, L, seat):
            assert s in inc(ends[e])
            assert s not in L[seat]


def test_R037_spec_example_candidates_exclude_the_street_in_line() -> None:
    """R-037 example: s1 already in the line, s2 and s3 also at v: the candidates are s2, s3.
    (Seat 0 of partial_loop at its end 1: the arrival street is excluded, all other streets at the
    node are offered because all are unoccupied.)"""
    L = lines_of("partial_loop")
    ln = L[0]
    v = _L().line_ends(B, ln)[1]
    expected = inc(v) - set(ln)
    assert expected and ln[-1] in inc(v)
    assert streets_at(legal("partial_loop"), 1) == expected


def test_R037_no_branch_streets_at_interior_nodes_never_offered() -> None:
    """R-037: the line stays a single non-branching path: a free street at a node in the middle of
    the line (not an end) is never a candidate."""
    L = lines_of("partial_loop")
    ln = L[0]
    ends = set(_L().line_ends(B, ln))
    interior = set(_L().line_nodes(B, ln)) - ends
    assert interior
    occ = occupied(L)
    offered = {s for _, s in legal("partial_loop")}
    free_at_interior = set().union(*(inc(n) for n in interior)) - occ
    assert free_at_interior, "fixture must leave a free street at the interior node"
    assert not (free_at_interior - set().union(*(inc(n) for n in ends))) & offered


def test_R037_full_loop_candidates_exclude_both_loop_streets_at_the_node() -> None:
    """R-037/R-041: at a full-loop node both of the line's streets there are excluded."""
    ln = lines_of("full_loop")[0]
    u = _L().line_ends(B, ln)[0]
    at_u_own = inc(u) & set(ln)
    assert len(at_u_own) == 2
    le = legal("full_loop")
    for e in (0, 1):
        assert streets_at(le, e) == inc(u) - set(ln)
        assert not (streets_at(le, e) & at_u_own)


def test_R037_end_without_candidate_is_dead_and_offers_nothing() -> None:
    """R-037/R-040: an end with no candidate street contributes no (end, street) pair; the live end
    still does."""
    L = lines_of("dead_one_end")
    ln = L[0]
    ends = _L().line_ends(B, ln)
    dead_end = [e for e in (0, 1) if not (inc(ends[e]) - set(ln))]
    live_end = [e for e in (0, 1) if inc(ends[e]) - set(ln)]
    assert dead_end == [1] and live_end == [0]
    le = legal("dead_one_end")
    assert streets_at(le, 1) == set()
    assert streets_at(le, 0) == inc(ends[0]) - set(ln)
    assert len(streets_at(le, 0)) == 1


def test_R037_both_ends_dead_gives_no_extensions() -> None:
    """R-037/R-040: a full loop at a degree-2 node leaves both ends dead: empty set."""
    assert legal("dead_both_ends") == set()


def test_R037_other_seats_streets_may_be_shared_when_all_candidates_occupied() -> None:
    """R-037/R-038: occupancy by other seats is not a candidate filter; only own streets are."""
    L = lines_of("red_cd")
    ends = _L().line_ends(B, L[0])
    all_occ_end = [e for e in (0, 1) if inc(ends[e]) - set(L[0]) <= occupied(L)]
    assert len(all_occ_end) == 1
    e = all_occ_end[0]
    assert streets_at(legal("red_cd"), e) == inc(ends[e]) - set(L[0])


def test_R037_each_seat_is_evaluated_with_its_own_line() -> None:
    """R-037: the `seat` argument selects whose line is extended; other seats' lines only matter as
    occupancy/ends.  Every seat's result equals the oracle."""
    for name in FIXTURES:
        L = lines_of(name)
        for seat in range(len(L)):
            assert set(_L().legal_extensions(B, L, seat)) == ref_legal(L, seat), (name, seat)


# ============================== R-038 ==============================


def test_R038_red_example_one_end_forced_other_end_free() -> None:
    """R-038 example (Red C/D): at an end with an unoccupied candidate the marker must go on an
    unoccupied candidate; at the other end, where all are occupied, any candidate may be chosen."""
    L = lines_of("red_cd")
    ln, occ = L[0], occupied(L)
    ends = _L().line_ends(B, ln)
    le = legal("red_cd")
    for e in (0, 1):
        cands = inc(ends[e]) - set(ln)
        free = cands - occ
        if free:
            assert streets_at(le, e) == free
            assert (cands - free) and not ((cands - free) & streets_at(le, e))
        else:
            assert streets_at(le, e) == cands and len(cands) >= 2


def test_R038_evaluated_per_end_not_across_both_ends() -> None:
    """R-038: a free candidate at one end does not restrict the other end."""
    L = lines_of("red_cd")
    ends = _L().line_ends(B, L[0])
    occ = occupied(L)
    free_ends = [e for e in (0, 1) if inc(ends[e]) - set(L[0]) - occ]
    full_ends = [e for e in (0, 1) if e not in free_ends]
    assert len(free_ends) == 1 and len(full_ends) == 1
    assert len(streets_at(legal("red_cd"), full_ends[0])) > len(
        streets_at(legal("red_cd"), free_ends[0])
    )


def test_R038_three_players_per_end() -> None:
    """R-038 at 3 players: one seat passes through (twice) the first end node, filling every
    candidate there; another passes through the second end node leaving one free."""
    L = lines_of("three_player_per_end")
    assert len(L) == 3
    ends = _L().line_ends(B, L[0])
    occ = occupied(L)
    le = legal("three_player_per_end")
    assert le == ref_legal(L, 0)
    for e in (0, 1):
        cands = inc(ends[e]) - set(L[0])
        free = cands - occ
        assert streets_at(le, e) == (free if free else cands)
    assert {len(inc(ends[e]) - set(L[0]) - occ) for e in (0, 1)} == {0, 1}


def test_R038_mid_line_occupancy_counts_as_occupied() -> None:
    """R-038 (R-013): a street under another line's middle is occupied; with an unoccupied
    candidate present it is not legal (no exception applies to mid-line, R-039)."""
    L = lines_of("single_street_midline")
    through = L[1]
    le = legal("single_street_midline")
    mid_nodes = set(_L().line_nodes(B, through)[1:-1])
    ends0 = _L().line_ends(B, L[0])
    v = next(n for n in ends0 if n in mid_nodes)
    e = end_at_node(L[0], v)
    assert set(through) <= inc(v)
    assert not (set(through) & streets_at(le, e))
    assert streets_at(le, e) == inc(v) - set(L[0]) - occupied(L)


def test_R038_all_candidates_occupied_allows_any_including_multiply_occupied() -> None:
    """R-038: when every candidate is occupied, any candidate may be chosen, also streets under
    several lines (seat 2 of three_player_per_end passes the node twice)."""
    L = lines_of("three_player_per_end")
    ends = _L().line_ends(B, L[0])
    occ = occupied(L)
    full = [e for e in (0, 1) if not (inc(ends[e]) - set(L[0]) - occ)]
    assert full
    e = full[0]
    assert streets_at(legal("three_player_per_end"), e) == inc(ends[e]) - set(L[0])


def test_R038_sequential_own_placement_counts_at_the_other_end() -> None:
    """R-038 (O-2): evaluation is sequential.  Full loop at u where only one candidate (x) is free:
    both ends are forced onto x.  After placing x at end 0 the other end's candidates (the two
    occupied streets) are all occupied, so both are allowed."""
    name = "full_loop_forced"
    L = lines_of(name)
    ln = L[0]
    u = _L().line_ends(B, ln)[0]
    cands = inc(u) - set(ln)
    free = cands - occupied(L)
    assert len(free) == 1 and len(cands) == 3
    (x,) = free
    le = legal(name)
    assert le == {(0, x), (1, x)}
    new = _L().extend(B, ln, 0, x)
    L2 = (new,) + L[1:]
    le2 = set(_L().legal_extensions(B, L2, 0))
    assert streets_at(le2, 1) == cands - {x}
    assert x not in streets_at(le2, 1)
    assert le2 == ref_legal(L2, 0)


def test_R038_placed_marker_is_occupied_for_other_seats() -> None:
    """R-038 (O-2): a street placed earlier counts as occupying for later evaluations, here for
    another seat whose end is at the same node: before, the placing seat's own end gave the
    exception; after the move that seat is mid-line there and the placed street is not free."""
    name = "sequential"
    L = lines_of(name)
    v = shared_end_node(L[0], L[1])
    e0 = end_at_node(L[0], v)
    e1 = end_at_node(L[1], v)
    cands0 = inc(v) - set(L[0])
    free0 = sorted(cands0 - occupied(L))
    assert len(free0) == 2
    before = streets_at(set(_L().legal_extensions(B, L, 1)), e1)
    assert set(L[0]) & before  # exception: seat 0's last street at v
    for x in free0:
        new0 = _L().extend(B, L[0], e0, x)
        L2 = (new0,) + L[1:]
        after = streets_at(set(_L().legal_extensions(B, L2, 1)), e1)
        other = (set(free0) - {x}) | set()
        assert x not in after
        assert set(L[0]) & after == set()
        assert after == other
        assert after < before


def test_R038_occupied_candidate_not_legal_when_unoccupied_exists() -> None:
    """R-038: negative case.  In shared_end, the street under seat 2's middle at v is not legal."""
    L = lines_of("shared_end")
    v = shared_end_node(L[0], L[1])
    e = end_at_node(L[0], v)
    mid_streets = set(L[2]) & inc(v)
    assert len(mid_streets) == 2
    assert not (mid_streets & streets_at(legal("shared_end"), e))


# ============================== R-039 ==============================


def test_R039_other_end_street_is_extra_option_unoccupied_stay_legal() -> None:
    """R-039 example: an unoccupied candidate and the street of another seat's end at v are both
    legal; occupied non-end streets (mid-line of a third seat) are not."""
    L = lines_of("shared_end")
    v = shared_end_node(L[0], L[1])
    e = end_at_node(L[0], v)
    y = L[1][-1] if _L().line_ends(B, L[1])[1] == v else L[1][0]
    free = inc(v) - set(L[0]) - occupied(L)
    assert free and y in inc(v) and y not in free
    le = streets_at(legal("shared_end"), e)
    assert le == free | {y}
    z = set(L[2]) & inc(v)
    assert z and not (z & le)


def test_R039_same_arrival_street_gives_no_exception() -> None:
    """R-039 counter-example: the other seat's last street at v is the very street seat 0 arrives
    on; that gives seat 0 nothing extra (the result is the plain unoccupied set)."""
    L = lines_of("shared_end_same_arrival")
    v = shared_end_node(L[0], L[1])
    assert L[0][-1] == L[1][-1]
    e = end_at_node(L[0], v)
    free = inc(v) - set(L[0]) - occupied(L)
    assert streets_at(legal("shared_end_same_arrival"), e) == free
    assert L[0][-1] not in streets_at(legal("shared_end_same_arrival"), e)


def test_R039_middle_of_another_line_gives_no_exception() -> None:
    """R-039 counter-example (rulebook p.2 Blue on Red's line): the end touches the middle of
    another line, so no street of that line is available while a free candidate exists."""
    L = lines_of("single_street_midline")
    through = L[1]
    mid = set(_L().line_nodes(B, through)[1:-1])
    v = next(n for n in _L().line_ends(B, L[0]) if n in mid)
    e = end_at_node(L[0], v)
    got = streets_at(legal("single_street_midline"), e)
    assert not (got & set(through))
    assert got and got == inc(v) - set(L[0]) - occupied(L)


def test_R039_other_seats_single_street_line_gives_exception_through_either_end() -> None:
    """R-039: a one-street line has an end at both of its nodes; its only street is that end's last
    street (b).  Seat 0 may use it from its end at either node, never a mid-line street."""
    L = lines_of("shared_end_single_street_other")
    (y,) = L[1]
    v = next(n for n in _L().line_ends(B, L[0]) if n in (B.streets[y].a, B.streets[y].b))
    e = end_at_node(L[0], v)
    free = inc(v) - set(L[0]) - occupied(L)
    got = streets_at(legal("shared_end_single_street_other"), e)
    assert got == free | {y}
    assert not (got & set(L[2]))


def test_R039_other_end_inside_own_loop_is_still_an_end_but_only_last_street() -> None:
    """R-039: an end of another seat that sits at a node its own path passed earlier (partial-line
    loop) is still an end: its last street is available.  The streets it uses to pass through that
    node are not (they are not the last street at that end)."""
    L = lines_of("shared_end_in_loop")
    other = L[1]
    v = shared_end_node(L[0], other)
    nodes = _L().line_nodes(B, other)
    assert nodes.count(v) == 2 and nodes[-1] == v
    y = other[-1]
    passing = set(other) & inc(v) - {y}
    assert passing
    e = end_at_node(L[0], v)
    free = inc(v) - set(L[0]) - occupied(L)
    got = streets_at(legal("shared_end_in_loop"), e)
    assert got == free | {y}
    assert not (got & passing)


def test_R039_end_of_other_full_loop_gives_both_last_streets() -> None:
    """R-039/R-041: a full-loop owner has both ends at v with different last streets there; each is
    an end, so both streets are available to seat 0."""
    L = lines_of("other_full_loop")
    loop = L[1]
    v = _L().line_ends(B, loop)[0]
    assert _L().line_ends(B, loop)[1] == v
    e = end_at_node(L[0], v)
    free = inc(v) - set(L[0]) - occupied(L)
    got = streets_at(legal("other_full_loop"), e)
    assert got == free | {loop[0], loop[-1]}
    assert loop[0] != loop[-1]


def test_R039_several_other_ends_each_street_available_mid_line_street_not() -> None:
    """R-039 at 5 players: two other seats have ends at v over different streets (each street is an
    allowed extra); a third seat passes through v (not allowed); one free street remains."""
    L = lines_of("five_player_shared")
    assert len(L) == 5
    v = shared_end_node(L[0], L[1])
    assert shared_end_node(L[0], L[2]) == v
    e = end_at_node(L[0], v)
    free = inc(v) - set(L[0]) - occupied(L)
    got = streets_at(legal("five_player_shared"), e)
    ya = L[1][-1] if _L().line_ends(B, L[1])[1] == v else L[1][0]
    yb = L[2][-1] if _L().line_ends(B, L[2])[1] == v else L[2][0]
    assert ya != yb
    assert got == free | {ya, yb}
    assert not (got & set(L[3]))
    assert len(free) == 1


def test_R039_exception_also_applies_with_a_full_loop_owner_and_a_foreign_end() -> None:
    """R-039 + R-041: a full-loop owner's both ends sit at u; another seat's end at u adds that
    seat's last street at both of the loop owner's ends, as an extra to the unoccupied ones."""
    L = lines_of("full_loop_shared")
    u = _L().line_ends(B, L[0])[0]
    y = next(s for s in L[1] if s in inc(u))
    free = inc(u) - set(L[0]) - occupied(L)
    le = legal("full_loop_shared")
    for e in (0, 1):
        assert streets_at(le, e) == free | {y}


def test_R039_after_use_new_end_is_evaluated_afresh() -> None:
    """R-039: after taking the shared street the new end is evaluated afresh; with no foreign end
    at its far node, only unoccupied candidates there are legal."""
    L = lines_of("shared_end")
    v = shared_end_node(L[0], L[1])
    e = end_at_node(L[0], v)
    y = L[1][-1] if _L().line_ends(B, L[1])[1] == v else L[1][0]
    new = _L().extend(B, L[0], e, y)
    L2 = (new,) + L[1:]
    nv = _L().line_ends(B, new)[e]
    assert nv == far(y, v)
    foreign_ends = {n for o in L2[1:] for _, n, _ in ends_info(o)}
    assert nv not in foreign_ends
    got = streets_at(set(_L().legal_extensions(B, L2, 0)), e)
    free = inc(nv) - set(new) - occupied(L2)
    assert free
    assert got == free


def test_R039_applies_to_one_street_lines_as_in_second_setup_marker() -> None:
    """R-039 last line / R-018: the exception applies to the second setup marker, i.e. for a
    one-street line.  Seat 1 (one street) has a foreign end at one of its nodes."""
    name = "shared_end_single_street_other"
    L = lines_of(name)
    (own,) = L[1]
    v = next(n for n in _L().line_ends(B, L[1]) if n in {m for _, m, _ in ends_info(L[0])})
    e = end_at_node(L[1], v)
    y = L[0][-1]
    got = streets_at(set(_L().legal_extensions(B, L, 1)), e)
    free = inc(v) - {own} - occupied(L)
    assert got == free | {y}
    assert own not in got


# ============================== R-041 ==============================


def test_R041_full_loop_both_ends_are_a_real_choice() -> None:
    """R-041: with both ends at one node both ends are offered with the same candidate streets;
    end 0 attaches at the start of the path, end 1 at its end."""
    ln = lines_of("full_loop")[0]
    le = legal("full_loop")
    s0, s1 = streets_at(le, 0), streets_at(le, 1)
    assert s0 == s1 and len(s0) >= 2
    x = sorted(s0)[0]
    a = _L().extend(B, ln, 0, x)
    b = _L().extend(B, ln, 1, x)
    assert a == (x,) + ln and b == ln + (x,) and a != b
    u = _L().line_ends(B, ln)[0]
    assert tuple(_L().line_ends(B, a)) == (far(x, u), u)
    assert tuple(_L().line_ends(B, b)) == (u, far(x, u))


def test_R041_after_full_loop_is_extended_the_ends_split() -> None:
    """R-041: once a street is added to a full loop the ends are at different nodes (a partial
    loop); the loop node is then interior and is no longer a place to extend from."""
    ln = lines_of("full_loop")[0]
    u = _L().line_ends(B, ln)[0]
    x = sorted(streets_at(legal("full_loop"), 0))[0]
    new = _L().extend(B, ln, 1, x)
    L = (new,) + lines_of("full_loop")[1:]
    ends = _L().line_ends(B, new)
    assert ends[0] != ends[1]
    offered = {(e, s) for e, s in _L().legal_extensions(B, L, 0)}
    assert offered == ref_legal(L, 0)
    for e, s in offered:
        assert s in inc(ends[e])
    assert u in set(_L().line_nodes(B, new)[1:-1])
    assert not any(s in inc(u) and s not in inc(ends[0]) | inc(ends[1]) for _, s in offered)


def test_R041_closing_a_loop_makes_both_ends_coincide() -> None:
    """R-041: a new street ending at a node already on the same line is allowed; when it joins the
    two ends the line becomes a full loop, with both ends still stored as end 0 and end 1."""
    full = lines_of("full_loop")[0]
    open_ln, closer = full[:-1], full[-1]
    n0, n1 = _L().line_ends(B, open_ln)
    assert n0 != n1 and {n0, n1} == {B.streets[closer].a, B.streets[closer].b}
    L = (open_ln,) + lines_of("full_loop")[1:]
    le = set(_L().legal_extensions(B, L, 0))
    assert (1, closer) in le and (0, closer) in le
    c1 = _L().extend(B, open_ln, 1, closer)
    assert c1 == full
    e0, e1 = _L().line_ends(B, c1)
    assert e0 == e1 == n0
    c0 = _L().extend(B, open_ln, 0, closer)
    assert c0 == (closer,) + open_ln
    f0, f1 = _L().line_ends(B, c0)
    assert f0 == f1 == n1


def test_R041_partial_loop_ends_differ_and_each_is_evaluated_from_its_own_node() -> None:
    """R-041: for a line whose ends are at different nodes the choice is determined by the node;
    the loop node in the middle offers nothing; each end evaluates R-038/R-039 itself (end at which
    a foreign line ends gets the exception)."""
    L = lines_of("partial_loop")
    ln = L[0]
    e0n, e1n = _L().line_ends(B, ln)
    assert e0n != e1n
    le = legal("partial_loop")
    assert le == ref_legal(L, 0)
    foreign = shared_end_node(L[0], L[1])
    ef = end_at_node(L[0], foreign)
    y = L[1][-1] if _L().line_ends(B, L[1])[1] == foreign else L[1][0]
    other_end = 1 - ef
    assert y in streets_at(le, ef) and y not in streets_at(le, other_end)
    assert streets_at(le, other_end) == inc(_L().line_ends(B, ln)[other_end]) - set(ln)


def test_R041_full_loop_forced_street_chooses_which_end_is_attached() -> None:
    """R-041 + R-038: forced single free street x at the loop node: (0, x) and (1, x) are both
    legal and lead to different stored paths."""
    ln = lines_of("full_loop_forced")[0]
    le = legal("full_loop_forced")
    assert len(le) == 2 and {e for e, _ in le} == {0, 1}
    (x,) = {s for _, s in le}
    assert _L().extend(B, ln, 0, x) != _L().extend(B, ln, 1, x)


def test_R041_dead_full_loop_has_no_ends_to_extend() -> None:
    """R-041/R-037: a full loop at a node whose every street is already the loop's is dead at both
    ends even though both ends exist."""
    ln = lines_of("dead_both_ends")[0]
    e0, e1 = _L().line_ends(B, ln)
    assert e0 == e1
    assert inc(e0) <= set(ln)
    assert legal("dead_both_ends") == set()


# ============================== whole-fixture agreement ==============================


@pytest.mark.parametrize("name", FIXTURES)
def test_R038_every_fixture_matches_the_spec_oracle_for_seat_zero(name: str) -> None:
    """R-036..R-039, R-041: legal_extensions equals the independent spec oracle."""
    assert legal(name) == ref_legal(lines_of(name), 0)


# ============================== state wiring (unit 8) ==============================


@pytest.mark.parametrize("name", [n for n in FIXTURES if n != "dead_both_ends"])
def test_R042_lines_phase_legal_actions_are_the_geometry_extensions(name: str) -> None:
    """ACTIONS 3.5 / R-042: in phase LN with one unit left, legal_actions() is exactly the encoded
    (end, street) pairs of legal_extensions for the acting seat (seat 0 in these fixtures)."""
    from bus.actions import encode_extend

    state = H.load_fixture(f"lines/{name}")
    expected = {encode_extend(e, s) for e, s in _L().legal_extensions(B, lines_of(name), 0)}
    assert state.current_player() == 0
    assert set(state.legal_actions()) == expected
