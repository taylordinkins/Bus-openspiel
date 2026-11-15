"""Setup phases SB, L1, L2 (R-015..R-019, R-021; ACTIONS §3.1-§3.3, §4; SCENARIOS §4, §4.3).

Written ahead of the L1/L2 code (Phase 1 unit 5).  SB (R-016) is already playable, so SB-only tests
carry no pending marker.  Everything that reaches L1, L2 or the SEL hand-over is pending.

Public API assumed (no engine internals):
  state.round / .phase / .current_player() / .start_player / .move_number() / .history()
  state.seat(p).line / .line_markers / .buildings ; state.site_building(t) ; state.is_occupied(t)
  state.waiting_at(node) ; state.passenger_supply ; state.action_board ; state.action_to_string
  bus.lines.legal_extensions(board, lines, seat) -> frozenset[(end, street)]
  bus.lines.line_ends(board, line) -> (node of end 0, node of end 1)
  bus.lines.extend(board, line, end, street) -> tuple

Ids are never hard-coded: streets, sites and nodes come from the board and from fixtures.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from bus import constants as C
from bus.actions import BUILD_BASE, EXTEND_BASE, encode_build, encode_extend
from bus.board import load_board
from tests import harness as H

B = load_board()
S = len(B.streets)
T = len(B.sites)
PLAYER_COUNTS = [3, 4, 5]


# ---- helpers ---------------------------------------------------------------------------------


def _L() -> Any:
    from bus import lines

    return lines


def fresh(n: int, start_player: int | None = None) -> Any:
    doc: dict[str, Any] = {"format": 1, "parameters": {"players": n}}
    if start_player is not None:
        doc["start_player"] = start_player
    return H.load_doc_state(doc)


def order(n: int, start: int) -> list[int]:
    return [(start + i) % n for i in range(n)]


def expected_actor_sequence(n: int, start: int) -> list[tuple[str, int]]:
    fwd = order(n, start)
    seq = [("SB", p) for p in fwd for _ in range(2)]
    seq += [("L1", p) for p in fwd]
    seq += [("L2", p) for p in reversed(fwd)]
    return seq


def seat_lines(state: Any) -> tuple[tuple[int, ...], ...]:
    n = state.get_game().num_players()
    return tuple(tuple(state.seat(p).line) for p in range(n))


def street_str(s: int) -> str:
    st = B.streets[s]
    return f"s{s} (n{st.a}-n{st.b})"


def lowest_open_tier_sites(state: Any) -> list[int]:
    for tier in "ABCD":
        empty = [t for t in range(T) if B.sites[t].tier == tier and state.site_building(t) is None]
        if empty:
            return empty
    return []


def pick(phase: str, legal: list[int], rng: random.Random) -> int:
    return rng.choice(legal)


def play_setup(state: Any, seed: int = 0, check: bool = True) -> list[tuple[str, int, int]]:
    """Play random legal setup moves until the phase leaves SB/L1/L2; returns (phase, seat, action)
    per node.  With ``check`` the per-step expectations (actor, tier, L2 legality) are asserted."""
    rng = random.Random(seed)
    n = state.get_game().num_players()
    seq = expected_actor_sequence(n, state.start_player)
    out: list[tuple[str, int, int]] = []
    i = 0
    while state.phase in ("SB", "L1", "L2"):
        assert i < len(seq), "setup ran longer than 4n nodes"
        ph, seat = seq[i]
        if check:
            assert state.phase == ph, f"node {i}"
            assert state.current_player() == seat, f"node {i}"
            assert state.round == 0
            assert state.move_number() == i
        legal = state.legal_actions()
        if check and ph == "SB":
            want = sorted(
                encode_build(t, y) for t in lowest_open_tier_sites(state) for y in range(3)
            )
            assert legal == want
        if check and ph == "L2":
            want_l2 = sorted(
                encode_extend(e, s) for e, s in _L().legal_extensions(B, seat_lines(state), seat)
            )
            assert legal == want_l2
        a = pick(ph, legal, rng)
        out.append((ph, seat, a))
        state.apply_action(a)
        i += 1
    assert i == len(seq)
    return out


# ---- R-016 : first buildings (SB already playable) -------------------------------------------


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R016_two_buildings_per_seat_in_turn_order(n: int) -> None:
    """R-016: in turn order each seat places both its buildings before the next seat starts."""
    s = fresh(n)
    seen: list[int] = []
    for _ in range(2 * n):
        assert s.phase == "SB"
        seen.append(s.current_player())
        s.apply_action(s.legal_actions()[0])
    assert seen == [p for p in range(n) for _ in range(2)]
    assert s.phase != "SB"


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R016_lowest_open_tier_re_evaluated_per_building(n: int) -> None:
    """R-016/R-010: before every building the legal set is exactly (every type on) the empty sites
    of the lowest tier that still has one; a filled site is never offered again."""
    s = fresh(n)
    built: set[int] = set()
    for _ in range(2 * n):
        want = sorted(encode_build(t, y) for t in lowest_open_tier_sites(s) for y in range(3))
        assert s.legal_actions() == want
        a = s.legal_actions()[-1]
        t = (a - BUILD_BASE) // 3
        assert t not in built
        built.add(t)
        s.apply_action(a)
    assert {t for t in range(T) if s.site_building(t) is not None} == built


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R016_setup_buildings_are_tier_a_with_chosen_type(n: int) -> None:
    """R-016: with 2n <= 10 buildings and 12 tier-A sites, all setup buildings are tier A; the
    chosen type is the one built; seat building counts become 2 each."""
    s = fresh(n)
    for i in range(2 * n):
        y = i % 3
        t = lowest_open_tier_sites(s)[0]
        s.apply_action(encode_build(t, y))
        assert B.sites[t].tier == "A"
        assert s.site_building(t) == C.BUILDING_TYPES[y]
    assert all(s.seat(p).buildings == 2 for p in range(n))


def test_R016_site_just_built_is_not_legal_again() -> None:
    """R-016: a second building cannot go on the site the first one just took (no type of it)."""
    s = fresh(4)
    t = lowest_open_tier_sites(s)[0]
    s.apply_action(encode_build(t, 0))
    assert not set(s.legal_actions()) & {encode_build(t, y) for y in range(3)}


def test_R016_no_line_action_while_in_sb() -> None:
    """R-016/R-017: no EXTEND id is legal until all buildings are placed."""
    s = fresh(4)
    for _ in range(7):
        assert not [a for a in s.legal_actions() if EXTEND_BASE <= a < BUILD_BASE]
        s.apply_action(s.legal_actions()[0])
    assert s.phase == "SB"


def test_R016_fixture_midway_next_seat_and_excludes() -> None:
    """R-016, SCENARIOS §4: the shipped mid-SB fixture rests with seat 1 to act on a fresh site."""
    s = H.run_fixture("setup_sb_midway")
    assert s.phase == "SB" and s.current_player() == 1
    built = {t for t in range(T) if s.site_building(t) is not None}
    assert len(built) == 3
    assert not set(s.legal_actions()) & {encode_build(t, y) for t in built for y in range(3)}


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R016_after_last_building_phase_is_l1(n: int) -> None:
    """R-016/R-017: after the 2n-th building the phase is L1 with the start player to act."""
    s = fresh(n)
    for _ in range(2 * n):
        s.apply_action(s.legal_actions()[0])
    assert s.phase == "L1"
    assert s.current_player() == 0
    assert s.round == 0


# ---- R-017 : first Bus Line marker -----------------------------------------------------------


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R017_l1_legal_is_every_street_end_zero(n: int) -> None:
    """R-017, ACTIONS §3.2: the L1 legal set is EXTEND(0, s) for every street, and nothing else
    (in particular no EXTEND(1, s))."""
    s = fresh(n)
    for _ in range(2 * n):
        s.apply_action(s.legal_actions()[0])
    assert s.phase == "L1"
    assert s.legal_actions() == sorted(encode_extend(0, st) for st in range(S))
    assert not set(s.legal_actions()) & {encode_extend(1, st) for st in range(S)}


def test_R017_occupied_street_stays_legal_for_later_seats() -> None:
    """R-017: the street may already be occupied by another player's marker."""
    s = fresh(4)
    for _ in range(8):
        s.apply_action(s.legal_actions()[0])
    street = 5
    s.apply_action(encode_extend(0, street))
    assert s.current_player() == 1
    assert encode_extend(0, street) in s.legal_actions()
    assert len(s.legal_actions()) == S
    s.apply_action(encode_extend(0, street))
    assert list(s.seat(0).line) == [street] == list(s.seat(1).line)


def test_R017_l1_creates_single_street_line_and_spends_marker() -> None:
    """R-017/R-035: L1 puts the chosen street alone in the line and costs one Bus Line marker."""
    s = fresh(3)
    for _ in range(6):
        s.apply_action(s.legal_actions()[0])
    street = S - 1
    assert s.seat(0).line_markers == C.LINE_MARKERS_PER_PLAYER
    s.apply_action(encode_extend(0, street))
    assert list(s.seat(0).line) == [street]
    assert s.seat(0).line_markers == C.LINE_MARKERS_PER_PLAYER - 1
    assert list(s.seat(1).line) == []
    assert s.seat(1).line_markers == C.LINE_MARKERS_PER_PLAYER


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R017_l1_goes_once_around_forward_then_l2(n: int) -> None:
    """R-017/R-018: n first markers in turn order, then the phase is L2."""
    s = fresh(n)
    for _ in range(2 * n):
        s.apply_action(s.legal_actions()[0])
    seats = []
    for i in range(n):
        assert s.phase == "L1"
        seats.append(s.current_player())
        s.apply_action(encode_extend(0, i))
    assert seats == list(range(n))
    assert s.phase == "L2"


def test_R017_fixture_l1_mid_expect_blocks_hold() -> None:
    """R-017, SCENARIOS §4: seats 0 and 1 served, seat 2 acts, ids of every street legal."""
    s = H.run_fixture("setup_l1_mid")
    assert s.phase == "L1" and s.current_player() == 2
    assert s.legal_actions() == sorted(encode_extend(0, st) for st in range(S))


def test_R017_ids_outside_extend_zero_are_not_legal_in_l1() -> None:
    """R-017: BUILD, SELECT and the rest are never legal in L1."""
    s = H.load_fixture("setup_l1_mid")
    assert s.legal_actions()
    legal = set(s.legal_actions())
    assert not legal & set(range(EXTEND_BASE))
    assert not legal & set(range(BUILD_BASE, BUILD_BASE + 3 * T))


# ---- R-018 : second Bus Line marker ----------------------------------------------------------


def test_R018_l2_reverse_order() -> None:
    """R-018: second markers go in reverse turn order, the last L1 seat first."""
    n = 4
    s = fresh(n)
    for _ in range(2 * n):
        s.apply_action(s.legal_actions()[0])
    for i in range(n):
        s.apply_action(encode_extend(0, i))
    seats = []
    for _ in range(n):
        assert s.phase == "L2"
        seats.append(s.current_player())
        s.apply_action(s.legal_actions()[0])
    assert seats == [3, 2, 1, 0]


def test_R018_fixture_l2_mid_legal_equals_legal_extensions() -> None:
    """R-018, ACTIONS §3.3: L2 legality is the encoded legal_extensions of the one-street line."""
    s = H.run_fixture("setup_l2_mid")
    assert s.phase == "L2" and s.current_player() == 2
    want = sorted(encode_extend(e, st) for e, st in _L().legal_extensions(B, seat_lines(s), 2))
    assert want
    assert s.legal_actions() == want


def test_R018_candidates_are_incident_to_an_end_and_not_the_own_street() -> None:
    """R-018: only streets at one of the first marker's two end nodes; never the street itself."""
    s = H.load_fixture("setup_l2_mid")
    line = tuple(s.seat(2).line)
    assert len(line) == 1
    ends = _L().line_ends(B, line)
    assert ends[0] != ends[1]
    legal = s.legal_actions()
    assert legal
    for a in legal:
        e = (a - EXTEND_BASE) // S
        st = (a - EXTEND_BASE) % S
        assert st != line[0]
        assert ends[e] in (B.streets[st].a, B.streets[st].b)
    assert encode_extend(0, line[0]) not in legal
    assert encode_extend(1, line[0]) not in legal
    near = {st for st in range(S) if set(ends) & {B.streets[st].a, B.streets[st].b}}
    far_streets = [st for st in range(S) if st not in near]
    assert not set(legal) & {encode_extend(e, st) for e in range(2) for st in far_streets}


def test_R018_extension_updates_line_and_markers() -> None:
    """R-018/R-036: the second marker extends the line at the chosen end and costs one marker."""
    s = H.load_fixture("setup_l2_mid")
    before = tuple(s.seat(2).line)
    a = s.legal_actions()[0]
    e = (a - EXTEND_BASE) // S
    st = (a - EXTEND_BASE) % S
    s.apply_action(a)
    assert tuple(s.seat(2).line) == _L().extend(B, before, e, st)
    assert len(s.seat(2).line) == 2
    assert s.seat(2).line_markers == C.LINE_MARKERS_PER_PLAYER - 2
    assert s.phase == "L2" and s.current_player() == 1


def _hub() -> int:
    deg: dict[int, int] = {}
    for st in B.streets:
        deg[st.a] = deg.get(st.a, 0) + 1
        deg[st.b] = deg.get(st.b, 0) + 1
    return max(deg, key=lambda v: deg[v])


def _shared_l2_state() -> tuple[Any, dict[str, int]]:
    """4p L2, seat 2 to act (order 3,2,1,0).  Hub node v: seat 2's line [a] ends at v; seat 1's
    line [y] has an end at v (y != a); seat 3's two-street line [w, z] passes THROUGH v (v is its
    middle node, not an end); seat 0's line [x] is elsewhere."""
    v = _hub()
    at_v = [i for i, st in enumerate(B.streets) if v in (st.a, st.b)]
    assert len(at_v) >= 5
    a, y, w, z = at_v[:4]
    other = lambda st: B.streets[st].b if B.streets[st].a == v else B.streets[st].a  # noqa: E731
    u = other(a)
    banned = {v, u, other(y), other(w), other(z)}
    x = next(i for i, st in enumerate(B.streets) if not ({st.a, st.b} & banned))
    doc = {
        "format": 1,
        "parameters": {"players": 4},
        "phase": "L2",
        "buildings": [{"site": t, "type": C.BUILDING_TYPES[t % 3]} for t in range(8)],
        "pending": {"done": 1},
        "seats": [{"line": [x]}, {"line": [y]}, {"line": [a]}, {"line": [w, z]}],
        "expect": {"phase": "L2", "current_player": 2},
    }
    s = H.load_doc_state(doc)
    return s, {"v": v, "a": a, "y": y, "w": w, "z": z, "x": x, "u": u}


def test_R018_shared_end_exception_applies_in_l2() -> None:
    """R-018/R-039: at v an unoccupied candidate exists, yet seat 1's last street y (an end of
    its line at v, different from the arriving street) is also legal; the mid-line streets w, z of
    seat 3 (v is not an end there) and the own street a are not."""
    s, r = _shared_l2_state()
    v = r["v"]
    end_v = 0 if _L().line_ends(B, (r["a"],))[0] == v else 1
    legal = set(s.legal_actions())
    unocc = {
        st
        for st in range(S)
        if v in (B.streets[st].a, B.streets[st].b) and st not in (r["a"], r["y"], r["w"], r["z"])
    }
    assert unocc
    assert encode_extend(end_v, r["y"]) in legal
    for st in unocc:
        assert encode_extend(end_v, st) in legal
    assert encode_extend(end_v, r["w"]) not in legal
    assert encode_extend(end_v, r["z"]) not in legal
    assert encode_extend(end_v, r["a"]) not in legal
    want = sorted(encode_extend(e, st) for e, st in _L().legal_extensions(B, seat_lines(s), 2))
    assert s.legal_actions() == want


def test_R018_sequential_evaluation_counts_earlier_markers_as_occupying() -> None:
    """R-018/R-038: markers placed earlier in L2 occupy their streets for later seats.  After
    seat 2 takes unoccupied street q at hub v, seat 1 (own line [y], end at v) must treat q as
    occupied: q is allowed only through the exception (it is seat 2's last street at v)."""
    s, r = _shared_l2_state()
    v = r["v"]
    end_v = 0 if _L().line_ends(B, (r["a"],))[0] == v else 1
    q = next(
        st
        for st in range(S)
        if v in (B.streets[st].a, B.streets[st].b) and st not in (r["a"], r["y"], r["w"], r["z"])
    )
    s.apply_action(encode_extend(end_v, q))
    assert s.current_player() == 1
    want = sorted(encode_extend(e, st) for e, st in _L().legal_extensions(B, seat_lines(s), 1))
    assert s.legal_actions() == want


# ---- R-019 / R-021 : start player, hand-over to round 1 --------------------------------------


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R021_full_setup_playout_seat_order(n: int) -> None:
    """R-016..R-018, R-021: 4n nodes (2 per seat SB, forward L1, reverse L2) then round 1 SEL."""
    s = fresh(n)
    moves = play_setup(s, seed=n)
    assert len(moves) == 4 * n
    assert [(ph, p) for ph, p, _ in moves] == expected_actor_sequence(n, 0)
    assert s.round == 1
    assert s.phase == "SEL"
    assert not s.is_terminal()
    assert s.current_player() == 0
    assert s.start_player == 0


@pytest.mark.parametrize("seed", range(5))
def test_R021_playout_end_state(seed: int) -> None:
    """R-021, R-015: after L2 every seat has 2 buildings and a 2-street line with 23 markers left,
    the action board is empty, and passengers are untouched."""
    n = 4
    s = fresh(n)
    play_setup(s, seed=seed)
    assert s.round == 1 and s.phase == "SEL"
    for p in range(n):
        v = s.seat(p)
        assert v.buildings == 2
        assert len(v.line) == 2
        assert v.line_markers == C.LINE_MARKERS_PER_PLAYER - 2
        assert v.points == 0
    assert sum(1 for t in range(T) if s.site_building(t) is not None) == 2 * n
    assert all(len(letters) == 0 for letters in s.action_board)


@pytest.mark.parametrize("n,start", [(3, 1), (4, 2), (4, 3), (5, 4), (5, 2)])
def test_R019_nonzero_start_player_changes_setup_order(n: int, start: int) -> None:
    """R-019, SCENARIOS §4.3: setup order is anchored at the start player; L2 reverses it; round 1
    opens with the start player."""
    s = fresh(n, start)
    assert s.start_player == start
    assert s.current_player() == start
    moves = play_setup(s, seed=start)
    assert [(ph, p) for ph, p, _ in moves] == expected_actor_sequence(n, start)
    assert s.round == 1 and s.phase == "SEL"
    assert s.current_player() == start
    assert s.start_player == start


def test_R019_sb_order_with_start_player() -> None:
    """R-019/R-016: with start player 2 of 4 the building order is 2,2,3,3,0,0,1,1."""
    s = fresh(4, 2)
    seen = []
    for _ in range(8):
        seen.append(s.current_player())
        s.apply_action(s.legal_actions()[0])
    assert seen == [2, 2, 3, 3, 0, 0, 1, 1]
    assert s.phase == "L1" and s.current_player() == 2


def test_R019_l2_first_actor_is_last_in_turn_order() -> None:
    """R-018/R-019: with start player 1 of 4, L1 runs 1,2,3,0 and L2 starts with seat 0."""
    s = fresh(4, 1)
    for _ in range(8):
        s.apply_action(s.legal_actions()[0])
    for i in range(4):
        s.apply_action(encode_extend(0, i))
    assert s.phase == "L2"
    assert s.current_player() == 0


# ---- move numbers, forced nodes --------------------------------------------------------------


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R021_move_number_counts_every_setup_node(n: int) -> None:
    """ACTIONS §1.5/§6: every node counts one move; setup is 4n nodes and history records them."""
    s = fresh(n)
    play_setup(s, seed=1)
    assert s.move_number() == 4 * n
    assert len(s.history()) == 4 * n


def test_R017_node_is_emitted_even_with_one_legal_action() -> None:
    """ACTIONS §1.5: a node is never skipped; every L1/L2 node has a legal action and every one
    of the 4n setup steps is a real decision of the seat in order (no auto-advance)."""
    s = fresh(4)
    count = 0
    rng = random.Random(3)
    while s.phase in ("SB", "L1", "L2"):
        assert s.legal_actions()
        s.apply_action(rng.choice(s.legal_actions()))
        count += 1
    assert count == 16


# ---- R-015 : passengers untouched by setup ---------------------------------------------------


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R015_setup_leaves_passengers_alone(n: int) -> None:
    """R-015: whatever is built, roundabout passengers keep waiting (none steps into a building at
    setup), the supply stays 11, and no other node gains passengers."""
    s = fresh(n)
    # build on every roundabout site first (those that are tier A), then finish setup
    rb_sites = [t for rb in B.roundabouts for t in B.node_sites[rb]]
    built_on_rb = 0
    rng = random.Random(n)
    while s.phase in ("SB", "L1", "L2"):
        legal = s.legal_actions()
        if s.phase == "SB":
            prefer = [a for a in legal if (a - BUILD_BASE) // 3 in rb_sites]
            if prefer:
                built_on_rb += 1
                legal = prefer
        s.apply_action(rng.choice(legal))
        for rb in B.roundabouts:
            assert s.waiting_at(rb) == 1
        assert s.passenger_supply == C.TOTAL_PASSENGERS - C.INITIAL_BOARD_PASSENGERS
        assert not any(s.is_occupied(t) for t in range(T))
    others = [nd for nd in range(len(B.nodes)) if nd not in set(B.roundabouts)]
    assert all(s.waiting_at(nd) == 0 for nd in others)
    assert s.phase == "SEL"


# ---- action_to_string ------------------------------------------------------------------------


def test_R017_action_to_string_l1_start_line() -> None:
    """ACTIONS §4: in L1 EXTEND reads 'P<seat>: start line on s<id> (n<a>-n<b>)'."""
    s = fresh(4)
    for _ in range(8):
        s.apply_action(s.legal_actions()[0])
    for seat, st in ((0, 0), (0, S - 1), (3, 17 % S)):
        assert s.action_to_string(seat, encode_extend(0, st)) == (
            f"P{seat}: start line on {street_str(st)}"
        )
    assert s.action_to_string(encode_extend(0, 4)) == f"P0: start line on {street_str(4)}"


def test_R018_action_to_string_l2_extend_wording() -> None:
    """ACTIONS §4: outside L1 EXTEND reads 'P<seat>: extend end <e> via s<id> (n<a>-n<b>) to
    n<new end>'; in L2 the 'start line' wording is not used."""
    s = H.load_fixture("setup_l2_mid")
    line = tuple(s.seat(2).line)
    ends = _L().line_ends(B, line)
    legal = s.legal_actions()
    assert legal
    for a in legal:
        e = (a - EXTEND_BASE) // S
        st = (a - EXTEND_BASE) % S
        sd = B.streets[st]
        new_end = sd.b if sd.a == ends[e] else sd.a
        assert s.action_to_string(2, a) == (
            f"P2: extend end {e} via {street_str(st)} to n{new_end}"
        )
        assert "start line" not in s.action_to_string(a)


def test_R017_action_to_string_follows_phase_not_id() -> None:
    """ACTIONS §4: the same EXTEND(0, s) id is 'start line' in L1 and 'extend' in L2."""
    s = fresh(3)
    for _ in range(6):
        s.apply_action(s.legal_actions()[0])
    st = 9
    a = encode_extend(0, st)
    assert "start line on" in s.action_to_string(0, a)
    for i in range(3):
        s.apply_action(encode_extend(0, i))
    assert s.phase == "L2"
    assert "start line" not in s.action_to_string(2, a)
    assert " extend end 0 via " in s.action_to_string(2, a)


@pytest.mark.parametrize("seat", [0, 1, 2])
def test_R016_action_to_string_sb(seat: int) -> None:
    """ACTIONS §4: SB BUILD reads 'P<seat>: build <type> at site <t> (<tier>, n<node>)'."""
    s = fresh(3)
    t = lowest_open_tier_sites(s)[2]
    for y, word in enumerate(C.BUILDING_TYPES):
        node = B.sites[t].node
        assert s.action_to_string(seat, encode_build(t, y)) == (
            f"P{seat}: build {word} at site {t} ({B.sites[t].tier}, n{node})"
        )
