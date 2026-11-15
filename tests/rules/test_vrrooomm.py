"""Vrrooomm! (phases VP / VD): R-056..R-063, with R-008, R-034, R-070 (move number).

Written ahead of Phase 1 unit 11 (now implemented).

Positions come from ``tests/scenarios/vrrooomm/base.json`` (round 1, four seats, clock house).
Since D22 (R-055, S-CLOCKSTEP) no passenger waits at a node beside an empty building of the
clock's type, so the waiting passengers stand at nodes of each line that have no building:
  P0 line nodes 6,7,8                empty houses at sites 36 (n6) 35 (n7) 15 25 (n8)
  P1 line nodes 12,17,20,16,19,24    empty houses at 17 (n16) 39 (n19) 8 27 (n20; canonical = 8);
                                     passengers wait at n24, n17, n12 (no buildings there)
  P2 line nodes 22,21,26,25          houses 18 (n22) 9 10 (n21; canonical 9); offices 11 30 (n26;
                                     canonical 11); pub 28 (n22); passengers wait at station n25
  P3 line nodes 28,29,30,35          houses 41 (n28) 20 (n29) 43 (n30); n35 has no building
  off every line: houses 22 (n33) 2 (n13, holds a passenger), office 40 (n23), pubs 31 29
  waiting: one passenger each at n24 n17 n12 n25 n13 n18, all buses 1.
Nodes 21, 13, 17, 18 are roundabouts, 25 a station.  Each test edits a copy of that document.

Two ways in:
  * ``enter`` builds a round-1 SEL position whose last DROP_OUT starts Phase 2 with the given
    Vrrooomm! markers (seats that cannot hold two markers are "spent", markers_left 0, as in
    test_resolution.py).  Because of those spent seats the game would end by R-066 at the first
    round end, so these tests never look past the last Vrrooomm! node.  Without a Clock marker
    the automatic Clock step (R-050) advances the clock before Vrrooomm!, so the SEL position
    starts one type earlier and ``clock`` names the type Vrrooomm! sees; passengers step out and
    in at that step (R-054, R-055).
  * ``at_vp`` loads a VP / VD position directly (SCENARIOS 4); used for scoring detail, move
    numbers and round-end tests (all seats keep markers).
Public API plus ``bus.actions`` constants only; pending data is read with dump_scenario
(SCENARIOS 4).
"""

from __future__ import annotations

from typing import Any

from bus.actions import CLOCK_ADVANCE, DELIVER_TO_BASE, DROP_OUT, PICK_PASSENGER_BASE
from tests import harness as H

LINES, BUSES, PASSENGERS, BUILDINGS, CLOCK, VRROOOMM, STARTING = range(7)
NAMES = ["lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player"]
SPACES = [6, 1, 6, 6, 1, 6, 1]

BASE = "vrrooomm/base"


def pick(n: int) -> int:
    return PICK_PASSENGER_BASE + n


def deliver(t: int) -> int:
    return DELIVER_TO_BASE + t


# ---- document builders -------------------------------------------------------------------------


def edit(
    d: dict[str, Any],
    *,
    buses: dict[int, int] | None = None,
    waiting: dict[int, int] | None = None,
    clock: str | None = None,
    set_b: dict[int, str | None] | None = None,
    occupied: tuple[int, ...] = (),
    lines: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    """Apply scenario edits.  ``set_b`` maps site -> type (None removes the building)."""
    for p, b in (buses or {}).items():
        d["seats"][p]["buses"] = b
    if waiting is not None:
        d["waiting"] = {str(n): c for n, c in waiting.items()}
    d.pop("supply", None)
    if clock is not None:
        d["clock"] = clock
    by_site = {b["site"]: b for b in d["buildings"]}
    for site, typ in (set_b or {}).items():
        if typ is None:
            by_site.pop(site, None)
        else:
            by_site[site] = {"site": site, "type": typ}
    for site in occupied:
        by_site[site]["occupied"] = True
    d["buildings"] = list(by_site.values())
    for p, ln in (lines or {}).items():
        d["seats"][p]["line"] = ln
    return d


def base_doc(**kw: Any) -> dict[str, Any]:
    return edit(H.load_doc(BASE), **kw)


def sel_doc(
    owners: list[int], extra: dict[int, list[int]] | None = None, **kw: Any
) -> dict[str, Any]:
    """SEL document whose last DROP_OUT starts Phase 2.  ``owners`` = seats on Vrrooomm! letters
    a, b, ...; ``extra`` = markers on other actions (action index -> seats)."""
    d = base_doc(**kw)
    lists: dict[int, list[int]] = {a: [] for a in range(7)}
    lists[VRROOOMM] = list(owners)
    for a, seats in (extra or {}).items():
        lists[a] = list(seats)
    n = len(d["seats"])
    count = [0] * n
    for seats in lists.values():
        for s in seats:
            count[s] += 1
    if max(count) < 2:  # someone must be able to make the final DROP_OUT
        lists[STARTING].append(owners[0])
        count[owners[0]] += 1
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
    return d


PREV = {"house": "pub", "office": "house", "pub": "office"}


def enter(owners: list[int], extra: dict[int, list[int]] | None = None, **kw: Any) -> Any:
    """Phase 2 from a SEL position.  ``clock`` is the type at Vrrooomm!: without a Clock marker
    the SEL position starts at the previous type and the automatic Clock step advances it
    (R-050, R-051); with a Clock marker it is the type the marker's owner sees at CK."""
    if CLOCK not in (extra or {}):
        kw["clock"] = PREV[kw.get("clock") or "house"]
    s = H.load_doc_state(sel_doc(owners, extra, **kw))
    assert s.phase == "SEL"
    assert DROP_OUT in s.legal_actions()
    s.apply_action(DROP_OUT)
    return s


def at_vp(
    owners: list[int],
    letter: int,
    moves_left: int,
    picked: int | None = None,
    offset: int | None = None,
    seat_points: dict[int, tuple[int, int]] | None = None,
    **kw: Any,
) -> Any:
    """Load a VP (or VD when ``picked`` is given) position directly.  ``seat_points`` maps seat ->
    (points, last_score_move)."""
    d = base_doc(**kw)
    n = len(d["seats"])
    d["phase"] = "VD" if picked is not None else "VP"
    d["round"] = 1
    board: dict[str, list[Any]] = {name: [None] * SPACES[a] for a, name in enumerate(NAMES)}
    d["action_board"] = board
    board["vrrooomm"] = list(owners) + [None] * (6 - len(owners))
    pend: dict[str, Any] = {"cursor": {"letter": letter}, "moves_left": moves_left}
    if picked is not None:
        pend["picked_node"] = picked
    d["pending"] = pend
    for p in range(n):
        d["seats"][p]["markers_left"] = 20 - owners.count(p)
    for p, (pts, lsm) in (seat_points or {}).items():
        d["seats"][p]["points"] = pts
        d["seats"][p]["last_score_move"] = lsm
    if offset is not None:
        d["move_offset"] = offset
    return H.load_doc_state(d)


# ---- readers -----------------------------------------------------------------------------------


def dump(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s)


def pend(s: Any) -> dict[str, Any]:
    return dump(s).get("pending", {})


def in_vr(s: Any) -> bool:
    return bool(s.phase in ("VP", "VD"))


def letter(s: Any) -> int:
    return int(pend(s)["cursor"]["letter"])


def moves_left(s: Any) -> int:
    return int(pend(s)["moves_left"])


def waiting(s: Any, n: int) -> int:
    return int(dump(s)["waiting"].get(str(n), 0))


def building(s: Any, site: int) -> dict[str, Any] | None:
    for b in dump(s)["buildings"]:
        if b["site"] == site:
            return b
    return None


def occupied(s: Any, site: int) -> bool:
    b = building(s, site)
    return bool(b and b.get("occupied"))


def points(s: Any, p: int) -> int:
    return int(dump(s)["seats"][p]["points"])


def last_score(s: Any, p: int) -> int | None:
    return dump(s)["seats"][p]["last_score_move"]


def legal(s: Any) -> set[int]:
    return set(s.legal_actions())


def picks(*nodes: int) -> set[int]:
    return {pick(n) for n in nodes}


def delivers(*sites: int) -> set[int]:
    return {deliver(t) for t in sites}


def play_through(s: Any, choose: Any = None) -> None:
    """Apply actions (first legal by default) until the Vrrooomm! node chain is over."""
    guard = 0
    while in_vr(s):
        la = s.legal_actions()
        s.apply_action(la[0] if choose is None else choose(s, la))
        guard += 1
        assert guard < 60


# ---- R-056: marker resolves a..f, not dwindling, destination type = clock ----------------------


def test_R056_first_marker_is_letter_a_and_its_owner_acts() -> None:
    """R-056/R-030: Vrrooomm! markers resolve a to f: a = P2, b = P1; the first node is VP for
    marker a (cursor 0) and P2 acts."""
    s = enter([2, 1])
    assert s.phase == "VP"
    assert letter(s) == 0
    assert s.current_player() == 2


def test_R056_markers_resolve_in_letter_order_for_several_owners() -> None:
    """R-056: owners a=P1, b=P2, c=P3 each with one bus, one passenger and one house on their
    line: nodes come in order P1 (cursor 0), P2 (1), P3 (2) and each scores 1."""
    s = enter([1, 2, 3], waiting={24: 1, 25: 1, 35: 1})
    seen: list[Any] = []
    while in_vr(s):
        if s.phase == "VP":
            seen.append((letter(s), s.current_player()))
        s.apply_action(s.legal_actions()[0])
    assert seen == [(0, 1), (1, 2), (2, 3)]
    assert [points(s, p) for p in range(4)] == [0, 1, 1, 1]


def test_R056_marker_f_gets_the_full_bus_count() -> None:
    """R-056/R-057: no dwindling.  Letters a..e belong to P0 (no passenger on P0's line, so they
    are skipped); marker f (cursor 5) of P1 with 3 buses has moves_left 3, not 3 - 5."""
    s = enter([0, 0, 0, 0, 0, 1], buses={1: 3}, waiting={24: 1, 17: 1, 12: 1})
    assert s.phase == "VP"
    assert letter(s) == 5
    assert s.current_player() == 1
    assert moves_left(s) == 3


def test_R056_second_marker_of_same_owner_is_not_reduced() -> None:
    """R-056: P1 owns a and b, 2 buses, 6 waiting passengers, 4 houses on the line.  Marker a
    moves 2, marker b still has moves_left 2 (a dwindling rule would give 1)."""
    s = enter([1, 1], buses={1: 2}, waiting={24: 2, 17: 2, 12: 2})
    assert (letter(s), moves_left(s)) == (0, 2)
    for _ in range(4):  # pick + deliver, twice
        s.apply_action(s.legal_actions()[0])
    assert s.phase == "VP"
    assert (letter(s), moves_left(s)) == (1, 2)


def test_R056_destination_is_clock_type_office() -> None:
    """R-056/R-059: with the clock on office P2 (line n22 n21 n26) may deliver only to offices:
    canonical site 11 (n26), not the houses (9, 18) nor the pub (28)."""
    s = enter([2], clock="office", waiting={21: 1})
    s.apply_action(pick(21))
    assert legal(s) == delivers(11)


def test_R056_destination_is_clock_type_pub() -> None:
    """R-056/R-059: clock on pub: only the pub at n22 (site 28) qualifies for P2."""
    s = enter([2], clock="pub", waiting={21: 1})
    s.apply_action(pick(21))
    assert legal(s) == delivers(28)


def test_R056_clock_advanced_this_round_sets_the_destination_type() -> None:
    """R-056/R-029: P2 holds Clock and Vrrooomm!.  The Clock marker resolves first; advancing
    house -> office means the Vrrooomm! destination is the office (site 11), not a house.
    Waiting only at n21 (houses, no office) so passenger step-in changes nothing."""
    s = enter([2], extra={CLOCK: [2]}, waiting={21: 1})
    assert s.phase == "CK"
    s.apply_action(CLOCK_ADVANCE)
    assert s.phase == "VP"
    s.apply_action(pick(21))
    assert legal(s) == delivers(11)


def test_R056_clock_advanced_office_to_pub_sets_the_destination_type() -> None:
    """R-056: clock office -> pub through this round's Clock marker: Vrrooomm! then targets the
    pub at n22 (site 28)."""
    s = enter([2], extra={CLOCK: [2]}, clock="office", waiting={21: 1})
    s.apply_action(CLOCK_ADVANCE)
    assert s.phase == "VP"
    s.apply_action(pick(21))
    assert legal(s) == delivers(28)


# ---- R-057 / R-060: forced move count --------------------------------------------------------


def test_R060_count_limited_by_buses() -> None:
    """R-060: 2 buses, 3 waiting passengers, 4 empty houses on the line -> moves_left 2."""
    s = enter([1], buses={1: 2}, waiting={24: 1, 17: 1, 12: 1})
    assert s.phase == "VP"
    assert moves_left(s) == 2


def test_R060_count_limited_by_waiting_passengers() -> None:
    """R-060: 4 buses, only 2 waiting passengers on the line (more off the line), 4 houses:
    moves_left 2.  Off-line passengers do not count."""
    s = enter([1], buses={1: 4}, waiting={24: 1, 17: 1, 13: 3, 21: 2})
    assert moves_left(s) == 2


def test_R060_count_limited_by_empty_buildings_of_clock_type() -> None:
    """R-060: 4 buses, 6 waiting passengers, only 2 houses left on the line (8 and 27 removed):
    moves_left 2."""
    s = enter(
        [1],
        buses={1: 4},
        waiting={24: 2, 17: 2, 12: 2},
        set_b={8: None, 27: None},
    )
    assert moves_left(s) == 2


def test_R060_occupied_buildings_do_not_count_as_destinations() -> None:
    """R-060/R-059: 4 houses on the line but 3 hold a passenger: one destination -> moves_left 1
    though 3 buses and 3 waiting."""
    s = enter(
        [1],
        buses={1: 3},
        waiting={17: 1, 12: 2},
        occupied=(17, 39, 8),
    )
    assert moves_left(s) == 1


def test_R060_spec_example_three_passengers_three_offices_two_buses() -> None:
    """R-060 example: 3 passengers, 3 eligible offices on the line, 2 buses -> exactly 2 moves and
    2 points; the third passenger stays waiting."""
    s = enter(
        [1],
        clock="office",
        buses={1: 2},
        waiting={24: 1, 17: 1, 12: 1},
        set_b={17: "office", 39: "office", 8: "office", 27: "pub"},
    )
    assert moves_left(s) == 2
    play_through(s)
    assert points(s, 1) == 2
    assert sum(waiting(s, n) for n in (24, 17, 12)) == 1
    assert sum(1 for t in (17, 39, 8) if occupied(s, t)) == 2


def test_R060_count_all_limits_equal_one() -> None:
    """R-060: default one bus, passengers and houses available: moves_left 1."""
    s = enter([1])
    assert s.phase == "VP"
    assert moves_left(s) == 1


def test_R060_zero_when_no_waiting_passenger_on_the_line() -> None:
    """R-060: count 0 (every waiting passenger is off P1's line) -> no VP node at all."""
    s = enter([1], waiting={13: 2, 18: 1, 21: 1, 6: 1})
    assert not in_vr(s)


def test_R060_zero_when_no_building_of_the_clock_type_on_the_line() -> None:
    """R-060: clock office, no office anywhere on P1's line (P1 has only houses) -> no node."""
    s = enter([1], clock="office")
    assert not in_vr(s)


def test_R060_zero_when_all_destinations_occupied() -> None:
    """R-060: every house on the line already holds a passenger -> no node."""
    s = enter([1], waiting={24: 1}, occupied=(17, 39, 8, 27))
    assert not in_vr(s)


def test_R060_zero_when_passengers_are_only_inside_buildings() -> None:
    """R-060/R-058: no waiting passenger on the line (the only passengers on the line are inside
    occupied houses) -> no node although an empty house exists."""
    s = enter([1], waiting={13: 1}, occupied=(17,))
    assert not in_vr(s)


def test_R060_zero_marker_is_skipped_and_next_marker_resolves() -> None:
    """R-060: marker a (P0, nothing on its line) has count 0 and makes no node; the first node is
    marker b (P1)."""
    s = enter([0, 1])
    assert s.phase == "VP"
    assert letter(s) == 1
    assert s.current_player() == 1


def test_R060_count_recomputed_for_each_marker_buildings() -> None:
    """R-060: P1 owns a and b, 2 buses, 3 houses on the line (27 removed), plenty of passengers.
    Marker a moves 2 and fills 2 houses; marker b is recomputed: 1 house left -> moves_left 1."""
    s = enter(
        [1, 1],
        buses={1: 2},
        waiting={24: 2, 17: 2, 12: 2},
        set_b={27: None},
    )
    assert moves_left(s) == 2
    for _ in range(4):
        s.apply_action(s.legal_actions()[0])
    assert (s.phase, letter(s), moves_left(s)) == ("VP", 1, 1)


def test_R060_count_recomputed_for_each_marker_passengers() -> None:
    """R-060: 2 buses, 3 waiting passengers, 4 houses.  Marker a moves 2; marker b has only 1
    waiting passenger left -> moves_left 1."""
    s = enter([1, 1], buses={1: 2}, waiting={24: 1, 17: 1, 12: 1})
    for _ in range(4):
        s.apply_action(s.legal_actions()[0])
    assert (s.phase, letter(s), moves_left(s)) == ("VP", 1, 1)


def test_R060_second_marker_with_nothing_left_is_skipped() -> None:
    """R-060: 1 bus, one passenger on the line, 4 houses: marker a uses it; marker b has count 0
    so the chain ends after one move."""
    s = enter([1, 1], waiting={24: 1})
    assert moves_left(s) == 1
    s.apply_action(pick(24))
    s.apply_action(s.legal_actions()[0])
    assert not in_vr(s)
    assert points(s, 1) == 1


def test_R060_owner_cannot_stop_early_and_total_equals_forced_count() -> None:
    """R-060/ACTIONS 3.10: whatever the owner picks, every VP/VD node has a legal action (never
    blocked) and the marker always scores exactly the forced count.  Exhaustive over all pick and
    deliver choices: 3 buses, 6 waiting passengers (2 per node), 4 houses -> 3 points always."""
    start = enter([1], buses={1: 3}, waiting={24: 2, 17: 2, 12: 2})
    assert moves_left(start) == 3
    stack = [start]
    leaves = 0
    while stack:
        s = stack.pop()
        if not in_vr(s):
            assert points(s, 1) == 3
            leaves += 1
            continue
        la = s.legal_actions()
        assert la, "VP/VD node with no legal action"
        for a in la:
            c = s.clone()
            c.apply_action(a)
            stack.append(c)
    assert leaves > 1


def test_R060_moves_left_counts_down_and_phase_alternates() -> None:
    """R-057/R-060: with 2 moves the sequence is VP(2) pick VD deliver VP(1) pick VD deliver, then
    the marker is over.  Same seat acts throughout."""
    s = enter([1], buses={1: 2}, waiting={24: 1, 17: 1, 12: 1})
    trace: list[Any] = []
    while in_vr(s):
        trace.append((s.phase, moves_left(s), s.current_player()))
        s.apply_action(s.legal_actions()[0])
    assert trace == [("VP", 2, 1), ("VD", 2, 1), ("VP", 1, 1), ("VD", 1, 1)]


# ---- R-058: which passengers ------------------------------------------------------------------


def test_R058_pick_set_is_exactly_the_online_nodes_with_waiting_passengers() -> None:
    """R-058: P1's line covers n12 n17 n20 n16 n19 n24.  Waiting at n24, n17, n12, and on another
    seat's line and roundabouts off the line: PICK legal exactly for nodes 24, 17, 12."""
    s = enter([1], buses={1: 2})
    assert legal(s) == picks(24, 17, 12)


def test_R058_node_with_several_passengers_has_one_pick_action() -> None:
    """R-058: three passengers at n24 only: a single PICK(24) action."""
    s = enter([1], buses={1: 2}, waiting={24: 3})
    assert legal(s) == picks(24)


def test_R058_node_without_waiting_passenger_is_not_offered() -> None:
    """R-058: passengers at n24 and n12 only: PICK(17) is not legal although n17 is on the line."""
    s = enter([1], waiting={24: 1, 12: 1})
    assert legal(s) == picks(24, 12)
    assert pick(17) not in legal(s)


def test_R058_passenger_inside_a_building_is_not_carried() -> None:
    """R-058: house 17 at n16 holds a passenger (not waiting; it steps out and back in at the
    Clock step); passengers wait at n17 and n12: PICK offers 17 and 12 but not 16."""
    s = enter([1], waiting={17: 1, 12: 1}, occupied=(17,))
    assert occupied(s, 17)
    assert legal(s) == picks(17, 12)
    assert pick(16) not in legal(s)


def test_R058_passenger_off_the_line_is_not_carried() -> None:
    """R-058: passengers at the other seats' nodes and roundabouts are never offered to P1; here
    n25 (P2's station), n13 and n18 are populated and none is a legal pick."""
    s = enter([1])
    la = legal(s)
    for n in (21, 22, 26, 25, 13, 18, 6, 7, 8):
        assert pick(n) not in la


def test_R058_passenger_at_a_station_on_the_line_can_be_carried() -> None:
    """R-058: P1's line runs to the station n25 (streets 42, 37, 38, 44: n24 n19 n16 n20 n25).  A
    passenger placed at the station may be carried: PICK(25) legal with PICK(24); then the
    destinations are the houses at n19, n16 and n20 (sites 39, 17, canonical 8)."""
    s = enter([1], lines={1: [42, 37, 38, 44]}, waiting={25: 1, 24: 1})
    assert legal(s) == picks(24, 25)
    s.apply_action(pick(25))
    assert s.phase == "VD"
    assert legal(s) == delivers(17, 39, 8)


def test_R058_station_off_the_line_is_not_offered() -> None:
    """R-058: a passenger at station n25 while P1's line does not reach it: not a legal pick."""
    s = enter([1], waiting={25: 2, 24: 1})
    assert legal(s) == picks(24)


def test_R058_roundabout_passenger_on_the_line_can_be_carried() -> None:
    """R-058: roundabout n21 is on P2's line (n22 n21 n26 n25): the waiting passenger there may be
    picked (it stands at the intersection, not in a building).  Clock office: n21 has only houses,
    so the passenger stays out at the Clock step."""
    s = enter([2], clock="office", waiting={21: 1})
    assert legal(s) == picks(21)


# ---- R-059 / R-063: which buildings, canonical site ---------------------------------------------


def test_R059_deliver_set_is_canonical_sites_on_the_line() -> None:
    """R-059/R-063: P1, clock house.  Houses on the line: 17 (n16), 39 (n19), 8 and 27 (n20).
    DELIVER legal for 17, 39 and 8 only: 27 is not canonical (8 is lower at n20)."""
    s = enter([1], waiting={24: 1})
    s.apply_action(pick(24))
    assert s.phase == "VD"
    assert legal(s) == delivers(17, 39, 8)
    assert deliver(27) not in legal(s)


def test_R059_buildings_off_the_line_are_excluded() -> None:
    """R-059: houses at n13 (site 2), n33 (site 22) and on other seats' lines (9, 18, 41, ...) are
    not destinations for P1."""
    s = enter([1], waiting={24: 1})
    s.apply_action(pick(24))
    la = legal(s)
    for t in (2, 22, 9, 10, 18, 41, 20, 43, 36, 35, 15, 25):
        assert deliver(t) not in la


def test_R059_other_building_types_are_excluded() -> None:
    """R-059: clock house, P2: the pub (28, n22) and the offices (11, 30, n26) on the line are not
    offered; only the houses 9 (n21, canonical over 10) and 18 (n22)."""
    s = enter([2], waiting={25: 1})
    s.apply_action(pick(25))
    assert legal(s) == delivers(9, 18)
    for t in (28, 11, 30, 10):
        assert deliver(t) not in legal(s)


def test_R059_occupied_buildings_are_excluded_and_canonical_moves_up() -> None:
    """R-059/R-063: house 8 at n20 holds a passenger, so the canonical empty house there is 27;
    house 17 at n16 is occupied and n16 offers nothing."""
    s = enter([1], waiting={17: 1}, occupied=(8, 17))
    s.apply_action(pick(17))
    assert legal(s) == delivers(39, 27)


def test_R063_only_lowest_site_at_a_node_is_offered() -> None:
    """R-063: two empty houses 9 and 10 at the roundabout n21 (P2): only DELIVER_TO(9) exists for
    that node."""
    s = enter([2], waiting={25: 1})
    s.apply_action(pick(25))
    assert deliver(9) in legal(s)
    assert deliver(10) not in legal(s)


def test_R063_two_offices_at_one_node_offer_the_lower() -> None:
    """R-063: clock office, P2: offices 11 and 30 are both at n26; only 11 is offered."""
    s = enter([2], clock="office", waiting={21: 1})
    s.apply_action(pick(21))
    assert legal(s) == delivers(11)


def test_R063_filling_the_canonical_site_exposes_the_next_one() -> None:
    """R-063/R-057: P2 with 2 buses, clock house, passengers at station n25 (2 of them).  First move
    can be DELIVER_TO(9); on the second move site 9 is occupied and site 10 is the canonical choice
    at n21 (along with 18 at n22)."""
    s = enter([2], buses={2: 2}, waiting={25: 2})
    s.apply_action(pick(25))
    s.apply_action(deliver(9))
    s.apply_action(pick(25))
    assert legal(s) == delivers(10, 18)


# ---- R-061 retired (D22): no delivery at the passenger's own node -----


def test_R059_passenger_beside_an_empty_house_steps_in_and_is_not_carried() -> None:
    """R-055/R-058/R-059 (R-061 retired): a passenger waiting at n21 beside the empty houses 9 and
    10 steps into house 9 at the Clock step, so it is not a pick; the station passenger is carried
    elsewhere on P2's line, to 10 (n21, canonical now that 9 is occupied) or 18 (n22)."""
    s = enter([2], waiting={21: 1, 25: 1})
    assert occupied(s, 9) and waiting(s, 21) == 0
    assert legal(s) == picks(25)
    s.apply_action(pick(25))
    assert legal(s) == delivers(10, 18)


def test_R060_passenger_beside_the_only_house_steps_in_and_the_count_is_zero() -> None:
    """R-055/R-060 (R-061 retired): P3's only house is site 20 at n29 (41 and 43 removed) and the
    only passenger on its line waits at n29.  It steps into house 20 at the Clock step, so P3's
    marker has no eligible passenger: count 0, no node, no point."""
    s = enter([3], waiting={29: 1}, set_b={41: None, 43: None})
    assert not in_vr(s)
    assert occupied(s, 20) and waiting(s, 29) == 0
    assert points(s, 3) == 0


# ---- R-062: effects of a move -----


def test_R062_move_effects_on_state() -> None:
    """R-062: pick at n24, deliver to house 39 (n19): +1 point for P1, house
    39 occupied (17 stays empty), waiting at n24 goes 1 -> 0, other seats' points unchanged."""
    s = enter([1], waiting={24: 1, 17: 1})
    s.apply_action(pick(24))
    s.apply_action(deliver(39))
    assert points(s, 1) == 1
    assert occupied(s, 39)
    assert not occupied(s, 17)
    assert waiting(s, 24) == 0
    assert waiting(s, 17) == 1  # untouched; the pick/deliver moved only the n24 passenger
    assert [points(s, p) for p in (0, 2, 3)] == [0, 0, 0]


def test_R062_picked_passenger_still_counted_in_waiting_during_vd() -> None:
    """R-062/SCENARIOS 9.2: after PICK_PASSENGER the passenger is still counted at the picked node
    until DELIVER_TO; pending shows phase VD with picked_node."""
    s = enter([1], waiting={24: 2})
    s.apply_action(pick(24))
    assert s.phase == "VD"
    assert pend(s)["picked_node"] == 24
    assert waiting(s, 24) == 2
    assert points(s, 1) == 0
    s.apply_action(deliver(17))
    assert waiting(s, 24) == 1


def test_R062_each_move_scores_one_point() -> None:
    """R-062: 3 buses, three passengers on the line: after each DELIVER_TO the owner has +1."""
    s = enter([1], buses={1: 3}, waiting={24: 1, 17: 1, 12: 1})
    seen: list[Any] = []
    while in_vr(s):
        if s.phase == "VD":
            s.apply_action(s.legal_actions()[0])
            seen.append(points(s, 1))
        else:
            s.apply_action(s.legal_actions()[0])
    assert seen == [1, 2, 3]


def test_R008_points_are_added_to_existing_points_and_never_decrease() -> None:
    """R-008/R-062: a seat with 5 points who scores one move has 6."""
    s = at_vp([1], 0, 1, seat_points={1: (5, 3)}, offset=3)
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert points(s, 1) == 6


def test_R062_last_score_move_is_the_history_length_after_deliver() -> None:
    """R-062/ACTIONS 1.7: last_score_move is the move number of the DELIVER_TO.  From a SEL
    position (nothing before): DROP_OUT = 1, PICK = 2, DELIVER_TO = 3."""
    s = enter([1])
    assert last_score(s, 1) is None
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert len(s.history()) == 3
    assert last_score(s, 1) == 3


def test_R062_last_score_move_adds_the_scenario_move_offset() -> None:
    """R-062/SCENARIOS 9.5: scenario with move_offset 50; after PICK and DELIVER_TO the delivery is
    move 52 and the score is recorded as 52."""
    s = at_vp([1], 0, 1, seat_points={1: (2, 7)}, offset=50)
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert last_score(s, 1) == 52
    assert points(s, 1) == 3


def test_R062_last_score_move_tracks_the_latest_scoring_move() -> None:
    """R-062: two moves from a loaded VP (offset 50): the second DELIVER_TO is move 54, so
    last_score_move is 54 (not 52)."""
    s = at_vp([1], 0, 2, offset=50, buses={1: 2}, waiting={24: 1, 17: 1})
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert last_score(s, 1) == 52
    s.apply_action(pick(17))
    s.apply_action(deliver(39))
    assert last_score(s, 1) == 54


def test_R062_pick_alone_does_not_change_points_or_last_score_move() -> None:
    """R-062: only DELIVER_TO scores; after PICK_PASSENGER the seat's points and last_score_move
    are unchanged."""
    s = at_vp([1], 0, 1, seat_points={1: (2, 7)}, offset=50)
    s.apply_action(pick(24))
    assert points(s, 1) == 2
    assert last_score(s, 1) == 7


def test_R062_other_seats_last_score_move_untouched() -> None:
    """R-062: only the scoring owner's last_score_move changes; seats that never scored keep None
    and a seat that scored earlier keeps its old value."""
    s = at_vp([1], 0, 1, seat_points={0: (1, 9)}, offset=50)
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert last_score(s, 0) == 9
    assert last_score(s, 2) is None
    assert last_score(s, 3) is None


def test_R062_vd_loaded_position_delivers_the_picked_passenger() -> None:
    """R-062/SCENARIOS 4: a scenario loaded directly in VD (picked_node 12) offers only deliveries;
    delivering to the canonical house 8 at n20 scores and removes the n12 passenger."""
    s = at_vp([1], 0, 1, picked=12)
    assert s.phase == "VD"
    assert delivers(8) <= legal(s)
    s.apply_action(deliver(8))
    assert occupied(s, 8)
    assert waiting(s, 12) == 0
    assert points(s, 1) == 1


# ---- sequencing, round end ---------------------------------------------------------------------


def test_R056_next_marker_starts_with_a_pick_step() -> None:
    """R-056/ACTIONS 3.10: when a marker's moves_left reaches 0 the next marker (b, P2) starts at
    VP with its own owner and cursor 1."""
    s = enter([1, 2], waiting={24: 1, 25: 1})
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert s.phase == "VP"
    assert (letter(s), s.current_player(), moves_left(s)) == (1, 2, 1)


def test_R056_round_ends_after_last_marker() -> None:
    """R-056/R-064: after the final move of the last marker (letter b of two) no further
    Vrrooomm! node exists and the next round's selection begins."""
    s = at_vp([1, 2], 1, 1, waiting={25: 1})
    assert s.current_player() == 2
    s.apply_action(pick(25))
    s.apply_action(deliver(9))
    assert not in_vr(s)
    assert s.phase == "SEL"
    assert s.round == 2


def test_R056_remaining_markers_after_a_skipped_one_still_resolve_then_round_ends() -> None:
    """R-056/R-064: markers a=P1 (VP loaded, 1 move), b=P0 (nothing to move, skipped), c=P2:
    after P1's move the next node is c (cursor 2, P2); after P2's move the round ends."""
    s = at_vp([1, 0, 2], 0, 1, waiting={24: 1, 25: 1})
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert s.phase == "VP"
    assert (letter(s), s.current_player()) == (2, 2)
    s.apply_action(pick(25))
    s.apply_action(deliver(9))
    assert s.phase == "SEL" and s.round == 2


def test_R056_last_marker_with_zero_count_ends_the_round_without_nodes() -> None:
    """R-056/R-060/R-064: after the last scoring marker, trailing letters with count 0 are skipped
    and the round ends in the same transition."""
    s = at_vp([1, 0, 0], 0, 1, waiting={24: 1})
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert s.phase == "SEL" and s.round == 2


def test_R056_filled_buildings_stay_filled_into_next_round() -> None:
    """R-062/R-064: the delivered passenger stays inside its building into the next round."""
    s = at_vp([1], 0, 1, waiting={24: 1})
    s.apply_action(pick(24))
    s.apply_action(deliver(17))
    assert s.phase == "SEL"
    assert occupied(s, 17)
    assert points(s, 1) == 1


# ---- illegal actions ---------------------------------------------------------------------------


def test_R058_illegal_actions_during_vp() -> None:
    """R-057/R-058: in VP no DELIVER_TO, no SELECT/DROP_OUT, no pick of an off-line node and no
    pick of a node without waiting passengers is legal."""
    s = enter([1], waiting={24: 1, 13: 1, 21: 1})
    assert s.phase == "VP"
    assert legal(s) == picks(24)
    for t in (17, 39, 8):
        assert deliver(t) not in legal(s)


def test_R059_illegal_actions_during_vd() -> None:
    """R-057/R-059: in VD no PICK_PASSENGER is legal (one passenger at a time) and the legal
    deliveries are only DELIVER_TO ids."""
    s = enter([1], buses={1: 2}, waiting={24: 1, 17: 1})
    s.apply_action(pick(24))
    assert s.phase == "VD"
    assert all(a >= DELIVER_TO_BASE for a in legal(s))
    assert pick(17) not in legal(s)


# ---- action_to_string --------------------------------------------------------------------------


def test_R057_action_to_string_pick_and_deliver() -> None:
    """ACTIONS 4: 'P2: pick passenger at n21' and 'P2: deliver to site 9 (A, n21)'.  Clock office,
    so the n21 passenger (houses only there) stays out at the Clock step."""
    s = enter([2], clock="office", waiting={21: 1})
    assert s.action_to_string(2, pick(21)) == "P2: pick passenger at n21"
    s.apply_action(pick(21))
    assert s.action_to_string(2, deliver(9)) == "P2: deliver to site 9 (A, n21)"
    assert s.action_to_string(2, deliver(18)) == "P2: deliver to site 18 (B, n22)"


def test_R057_state_strings_differ_between_vp_and_vd() -> None:
    """CLAUDE.md: distinct states never print alike: the VP and VD states of one marker, and the
    two moves of a 2-move marker, have different strings."""
    s = enter([1], buses={1: 2}, waiting={24: 1, 17: 1})
    seen = [str(s)]
    s.apply_action(pick(24))
    seen.append(str(s))
    s.apply_action(deliver(17))
    seen.append(str(s))
    s.apply_action(pick(17))
    seen.append(str(s))
    assert len(set(seen)) == 4


# ---- R-070: move number (tiebreak data) ----


def test_R070_move_number_counts_forced_nodes_too() -> None:
    """R-062/R-070: the move number counts every decision node, forced ones included.  P2 has a
    single legal PICK (forced) and then chooses DELIVER_TO; DROP_OUT, PICK, DELIVER_TO are
    moves 1, 2, 3 and last_score_move is 3."""
    s = enter([2], waiting={25: 1})
    assert len(legal(s)) == 1  # forced pick
    s.apply_action(pick(25))
    s.apply_action(deliver(18))
    assert last_score(s, 2) == len(s.history()) == 3
