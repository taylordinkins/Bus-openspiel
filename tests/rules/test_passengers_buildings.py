"""Passengers and Buildings resolution steps (R-045..R-049; ACTIONS 3.7, 3.8, 4).

Written ahead of Phase 1 unit 9; every test is pending.

Positions are the fixtures ``resolve_px`` / ``resolve_bd`` (a marker at letter a, units_left = full
power) adjusted by the helpers below: the number of filled sites, the bus maximum M, the supply,
which seats own which letters, a cursor letter.  Sites fill in (tier, id) order, as R-010 demands.

Accessors assumed beyond the harness (reported to the caller):
  state.sites_full        bool: R-049 sites-full condition recorded (set when the last site fills)
  state.end_types         list/tuple of strings, containing "sites_full" once the game has ended
  state.start_player      only through the existing scenario dump (not used here)
Pending-node data (cursor letter, units_left) is read through ``dump_scenario(state)["pending"]``
as in test_resolution.py.

Example numbers: 47 sites (A 12, B 11, C 9, D 15), 2 stations, 15 passengers.
"""

from __future__ import annotations

from typing import Any

import pytest

from bus.actions import DROP_OUT, encode_build, encode_passenger_to
from bus.board import load_board
from tests import harness as H

B = load_board()
T = len(B.sites)
K = len(B.stations)
SITE_ORDER = sorted(range(T), key=lambda t: (B.sites[t].tier, t))
TYPES = ["house", "office", "pub"]
CLOCK_OWNER = 3
PASSENGER_TOTAL = 15


# ---- helpers ---------------------------------------------------------------------------------


def _clean(doc: dict[str, Any]) -> None:
    for k in ("expect", "description", "current_player"):
        doc.pop(k, None)
    for s in doc["seats"]:
        s["markers_left"] = 10


def _set_board(doc: dict[str, Any], name: str, owners: list[int]) -> None:
    doc["action_board"][name] = owners + [None] * (6 - len(owners))


def _extras(doc: dict[str, Any], clock: bool, tail: bool) -> None:
    if clock:
        doc["action_board"]["clock"] = [CLOCK_OWNER]
    if tail:
        doc["action_board"]["vrrooomm"] = [0, None, None, None, None, None]
        doc["action_board"]["starting_player"] = [1]


def px_doc(
    supply: int = 11,
    M: int = 3,
    owners: list[int] | None = None,
    cursor: int = 0,
    buildings: list[int] | None = None,
    clock: bool = False,
) -> dict[str, Any]:
    """Passengers marker(s) at the cursor letter; units_left = power M - cursor."""
    doc = H.load_doc("resolve_px")
    _clean(doc)
    owners = owners if owners is not None else [2]
    _set_board(doc, "passengers", owners)
    if buildings:
        _set_board(doc, "buildings", buildings)
    _extras(doc, clock, False)
    doc["waiting"] = {"13": PASSENGER_TOTAL - supply}
    doc["seats"][0]["buses"] = M
    doc["pending"] = {"cursor": {"letter": cursor}, "units_left": M - cursor}
    return doc


def bd_doc(
    filled: int = 4,
    M: int = 3,
    owners: list[int] | None = None,
    cursor: int = 0,
    clock: bool = False,
    tail: bool = False,
) -> dict[str, Any]:
    """Buildings marker(s) at the cursor letter; ``filled`` sites occupied in (tier, id) order."""
    doc = H.load_doc("resolve_bd")
    _clean(doc)
    owners = owners if owners is not None else [1]
    _set_board(doc, "buildings", owners)
    _extras(doc, clock, tail)
    doc["buildings"] = [
        {"site": t, "type": TYPES[i % 3]} for i, t in enumerate(SITE_ORDER[:filled])
    ]
    doc["seats"][0]["buses"] = M
    doc["pending"] = {"cursor": {"letter": cursor}, "units_left": M - cursor}
    return doc


def pending(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s).get("pending", {})


def node(s: Any) -> tuple[str, int | None, int, int | None]:
    p = pending(s)
    return (s.phase, p.get("cursor", {}).get("letter"), s.current_player(), p.get("units_left"))


def empty_sites(s: Any) -> list[int]:
    return [t for t in range(T) if s.site_building(t) is None]


def lowest_tier_empty(s: Any) -> list[int]:
    for tier in "ABCD":
        e = [t for t in empty_sites(s) if B.sites[t].tier == tier]
        if e:
            return e
    return []


def build_ids(sites: list[int]) -> list[int]:
    return sorted(encode_build(t, y) for t in sites for y in range(3))


def station_ids() -> list[int]:
    return [encode_passenger_to(k) for k in range(K)]


def all_waiting(s: Any) -> int:
    return sum(s.waiting_at(n.id) for n in B.nodes)


# ---- R-045: Passengers action ----------------------------------------------------------------


def test_R045_legal_actions_are_exactly_the_two_stations() -> None:
    """R-045: each unit offers both stations (one PASSENGER_TO per station) and nothing else;
    the owner of marker a (seat 2) acts."""
    s = H.load_doc_state(px_doc())
    assert K == 2
    assert s.phase == "PX" and s.current_player() == 2
    assert sorted(s.legal_actions()) == sorted(station_ids())


@pytest.mark.parametrize("owner", [0, 1, 2, 3])
def test_R045_station_need_not_be_on_the_owners_line(owner: int) -> None:
    """R-045: whoever owns the marker, both stations are offered; lines play no part."""
    s = H.load_doc_state(px_doc(owners=[owner]))
    assert s.current_player() == owner
    assert sorted(s.legal_actions()) == sorted(station_ids())


@pytest.mark.parametrize("k", range(K))
def test_R045_placement_moves_one_passenger_from_supply_to_station(k: int) -> None:
    """R-045: a unit takes one passenger from the supply to the chosen station; units_left -1,
    same owner, still PX."""
    s = H.load_doc_state(px_doc(supply=11, M=3))
    st = B.stations[k]
    before = s.waiting_at(st)
    other = s.waiting_at(B.stations[1 - k])
    s.apply_action(encode_passenger_to(k))
    assert s.waiting_at(st) == before + 1
    assert s.waiting_at(B.stations[1 - k]) == other
    assert s.passenger_supply == 10
    assert node(s) == ("PX", 0, 2, 2)


def test_R045_each_passenger_may_go_to_a_different_station() -> None:
    """R-045: the station is chosen per passenger: stations 0, 1, 0 give 2 and 1."""
    s = H.load_doc_state(px_doc(supply=11, M=3))
    for k in (0, 1, 0):
        s.apply_action(encode_passenger_to(k))
    assert s.waiting_at(B.stations[0]) == 2
    assert s.waiting_at(B.stations[1]) == 1
    assert s.passenger_supply == 8


def test_R045_full_power_is_used_when_supply_allows() -> None:
    """R-045: power 3, supply 11: exactly three nodes, then resolution moves on (Buildings P3)."""
    s = H.load_doc_state(px_doc(supply=11, M=3, buildings=[3]))
    n = 0
    while s.phase == "PX":
        s.apply_action(encode_passenger_to(0))
        n += 1
        assert n <= 3
    assert n == 3
    assert s.phase == "BD" and s.current_player() == 3
    assert s.passenger_supply == 8


def test_R045_spec_example_supply_one_power_three() -> None:
    """R-045 example: M = 3, supply 1, marker a: one passenger placed, two units lost; no node
    for the lost units, the next marker (Buildings P3) is up."""
    s = H.load_doc_state(px_doc(supply=1, M=3, buildings=[3]))
    assert s.passenger_supply == 1
    s.apply_action(encode_passenger_to(1))
    assert s.passenger_supply == 0
    assert s.waiting_at(B.stations[1]) == 1
    assert s.phase == "BD" and s.current_player() == 3


def test_R045_units_left_is_not_capped_by_the_supply() -> None:
    """R-045/SCENARIOS 9.8: with supply 2 and power 3 the first placement leaves units_left 2
    (the shortfall is only discovered, and the unit lost, when reached)."""
    s = H.load_doc_state(px_doc(supply=2, M=3, buildings=[3]))
    s.apply_action(encode_passenger_to(0))
    assert node(s) == ("PX", 0, 2, 2)
    s.apply_action(encode_passenger_to(0))
    assert s.phase == "BD"


def test_R045_empty_supply_marker_after_a_placement_is_skipped() -> None:
    """R-045/R-034: a (P2) takes the last passenger; b (P0, power 2) finds an empty supply, so
    it produces no node and play moves on to Buildings (P3)."""
    s = H.load_doc_state(px_doc(supply=1, M=3, owners=[2, 0], buildings=[3]))
    s.apply_action(encode_passenger_to(0))
    assert s.phase == "BD" and s.current_player() == 3
    assert s.passenger_supply == 0


def test_R045_lost_units_do_not_leave_an_empty_pending_node() -> None:
    """R-045: no PX node ever has an empty action list; running supply down with several
    markers always ends in a non-PX phase."""
    s = H.load_doc_state(px_doc(supply=2, M=4, owners=[2, 0, 1]))
    steps = 0
    while s.phase == "PX":
        assert s.legal_actions()
        s.apply_action(s.legal_actions()[0])
        steps += 1
        assert steps <= 2
    assert steps == 2
    assert s.passenger_supply == 0


def test_R045_letters_resolve_a_before_b_before_c() -> None:
    """R-030/R-045: M = 4, markers a (P2), b (P0), c (P3): powers 4, 3, 2, each finished before
    the next; nine passengers leave the supply."""
    s = H.load_doc_state(px_doc(supply=11, M=4, owners=[2, 0, 3]))
    seen = []
    while s.phase == "PX":
        seen.append(node(s))
        s.apply_action(encode_passenger_to(0))
        assert len(seen) <= 9
    assert seen == [("PX", 0, 2, u) for u in (4, 3, 2, 1)] + [
        ("PX", 1, 0, u) for u in (3, 2, 1)
    ] + [("PX", 2, 3, u) for u in (2, 1)]
    assert s.passenger_supply == 2


def test_R045_no_other_action_is_legal_at_a_passengers_unit() -> None:
    """R-045/R-034: no pass or drop-out; the owner cannot do less than place."""
    s = H.load_doc_state(px_doc())
    assert DROP_OUT not in s.legal_actions()
    assert encode_build(0, 0) not in s.legal_actions()
    assert len(s.legal_actions()) == K


# ---- R-046: placed passengers are waiting passengers -----------------------------------------


def test_R046_placed_passengers_wait_and_conserve_total() -> None:
    """R-046: after the whole marker, the placed passengers are waiting at the stations; no
    building is occupied or gained; supply + waiting = 15."""
    s = H.load_doc_state(px_doc(supply=11, M=3, buildings=[3]))
    occupied_before = [s.is_occupied(t) for t in range(T)]
    kinds_before = [s.site_building(t) for t in range(T)]
    for k in (0, 0, 1):
        s.apply_action(encode_passenger_to(k))
    assert [s.is_occupied(t) for t in range(T)] == occupied_before
    assert [s.site_building(t) for t in range(T)] == kinds_before
    assert s.passenger_supply + all_waiting(s) == PASSENGER_TOTAL
    assert s.waiting_at(B.stations[0]) == 2 and s.waiting_at(B.stations[1]) == 1


def test_R046_passengers_never_step_into_buildings_by_themselves() -> None:
    """R-046: Passengers followed by Buildings: the passengers stay at the stations; no
    building site becomes occupied."""
    s = H.load_doc_state(px_doc(supply=11, M=1, buildings=[3]))
    s.apply_action(encode_passenger_to(0))
    assert s.phase == "BD"
    s.apply_action(s.legal_actions()[0])
    assert not any(s.is_occupied(t) for t in range(T))
    assert s.waiting_at(B.stations[0]) == 1


def test_R046_placed_passenger_adds_to_existing_waiting_count() -> None:
    """R-046: the station count stacks on passengers already there."""
    doc = px_doc(supply=9, M=2)
    doc["waiting"] = {str(B.stations[0]): 2, "13": 4}
    s = H.load_doc_state(doc)
    assert s.waiting_at(B.stations[0]) == 2
    s.apply_action(encode_passenger_to(0))
    assert s.waiting_at(B.stations[0]) == 3


def test_R045_action_to_string_passenger_to() -> None:
    """ACTIONS 4: PASSENGER_TO string is 'P<seat>: passenger to station n<node>'."""
    s = H.load_doc_state(px_doc())
    for k in range(K):
        assert (
            s.action_to_string(2, encode_passenger_to(k))
            == f"P2: passenger to station n{B.stations[k]}"
        )


# ---- R-047 / R-010: Buildings action ---------------------------------------------------------


def test_R047_legal_actions_every_type_on_every_empty_lowest_tier_site() -> None:
    """R-047/R-010: all (site, type) pairs on empty sites of tier A (4 filled), nothing else."""
    s = H.load_doc_state(bd_doc(filled=4, M=2))
    assert s.phase == "BD" and s.current_player() == 1
    sites = lowest_tier_empty(s)
    assert len(sites) == 8 and {B.sites[t].tier for t in sites} == {"A"}
    assert sorted(s.legal_actions()) == build_ids(sites)


def test_R047_higher_tier_sites_are_not_legal_while_a_lower_tier_is_open() -> None:
    """R-047/R-010: no tier B/C/D site and no occupied site is offered."""
    s = H.load_doc_state(bd_doc(filled=4, M=2))
    legal = set(s.legal_actions())
    for t in range(T):
        if B.sites[t].tier != "A" or s.site_building(t) is not None:
            for y in range(3):
                assert encode_build(t, y) not in legal


@pytest.mark.parametrize("y", range(3))
def test_R047_build_places_the_chosen_type_and_spends_a_unit(y: int) -> None:
    """R-047: BUILD(t, y) puts a building on site t (empty before), units_left -1, the site's
    passenger flag is not set, the site is no longer offered."""
    s = H.load_doc_state(bd_doc(filled=4, M=2))
    t = lowest_tier_empty(s)[0]
    assert s.site_building(t) is None
    s.apply_action(encode_build(t, y))
    assert s.site_building(t) is not None
    assert not s.is_occupied(t)
    assert node(s) == ("BD", 0, 1, 1)
    assert not any(a in s.legal_actions() for a in build_ids([t]))


def test_R047_buildings_of_the_three_types_are_distinguishable() -> None:
    """R-047: the type chosen is the type recorded (house, office, pub differ)."""
    seen = set()
    for y in range(3):
        s = H.load_doc_state(bd_doc(filled=4, M=1))
        t = lowest_tier_empty(s)[0]
        s.apply_action(encode_build(t, y))
        seen.add(repr(s.site_building(t)))
    assert len(seen) == 3


def test_R047_site_need_not_be_on_any_line() -> None:
    """R-047: a lowest-tier empty site whose node is on nobody's line is still offered."""
    s = H.load_doc_state(bd_doc(filled=4, M=1))
    line_nodes: set[int] = set()
    for p in range(s.num_players()):
        for st in s.seat(p).line:
            line_nodes |= {B.streets[st].a, B.streets[st].b}
    off = [t for t in lowest_tier_empty(s) if not any(t in B.node_sites[n] for n in line_nodes)]
    assert off, "fixture should leave an off-line tier A site"
    for t in off:
        assert encode_build(t, 0) in s.legal_actions()


def test_R047_units_use_full_power_and_resolution_moves_on() -> None:
    """R-047: power 3 gives three BUILD nodes, then the Clock marker (P3)."""
    s = H.load_doc_state(bd_doc(filled=4, M=3, clock=True))
    n = 0
    while s.phase == "BD":
        s.apply_action(s.legal_actions()[0])
        n += 1
        assert n <= 3
    assert n == 3
    assert s.phase == "CK" and s.current_player() == CLOCK_OWNER
    assert len(empty_sites(s)) == T - 4 - 3


def test_R047_no_other_action_is_legal_at_a_buildings_unit() -> None:
    """R-047/R-034: no pass, no passenger placement."""
    s = H.load_doc_state(bd_doc(filled=4, M=2))
    assert DROP_OUT not in s.legal_actions()
    assert not set(s.legal_actions()) & set(station_ids())


def test_R047_letters_resolve_highest_first_with_dwindling_power() -> None:
    """R-030/R-031/R-047: M = 5, markers a..f owned 0,1,2,3,0,1.  f has power 0 (skipped);
    e (P0, 1 unit), d (P3, 2), c (P2, 3), b (P1, 4), a (P0, 5)."""
    s = H.load_doc_state(bd_doc(filled=2, M=5, owners=[0, 1, 2, 3, 0, 1], cursor=4, clock=True))
    seen = []
    while s.phase == "BD":
        seen.append(node(s))
        s.apply_action(s.legal_actions()[0])
        assert len(seen) <= 15
    exp = [("BD", 4, 0, 1)]
    exp += [("BD", 3, 3, u) for u in (2, 1)]
    exp += [("BD", 2, 2, u) for u in (3, 2, 1)]
    exp += [("BD", 1, 1, u) for u in (4, 3, 2, 1)]
    exp += [("BD", 0, 0, u) for u in (5, 4, 3, 2, 1)]
    assert seen == exp
    assert s.phase == "CK"


# ---- R-048: tier advance within one action ---------------------------------------------------


def test_R048_tier_advances_between_units_of_one_marker() -> None:
    """R-047 example / R-048: tier A has one empty site, power 3.  Unit 1 offers only that site
    (3 types); after it, units 2 and 3 offer tier B sites only."""
    s = H.load_doc_state(bd_doc(filled=11, M=3))
    last_a = lowest_tier_empty(s)
    assert len(last_a) == 1 and B.sites[last_a[0]].tier == "A"
    assert sorted(s.legal_actions()) == build_ids(last_a)
    s.apply_action(encode_build(last_a[0], 0))
    tier_b = lowest_tier_empty(s)
    assert len(tier_b) == 11 and {B.sites[t].tier for t in tier_b} == {"B"}
    assert sorted(s.legal_actions()) == build_ids(tier_b)
    t = tier_b[0]
    s.apply_action(encode_build(t, 1))
    rest = lowest_tier_empty(s)
    assert len(rest) == 10 and t not in rest
    assert sorted(s.legal_actions()) == build_ids(rest)
    assert node(s) == ("BD", 0, 1, 1)


def test_R048_tier_does_not_advance_while_it_still_has_an_empty_site() -> None:
    """R-048: with two empty A sites, after one build the other A site is still the only
    offer."""
    s = H.load_doc_state(bd_doc(filled=10, M=2))
    a_sites = lowest_tier_empty(s)
    assert len(a_sites) == 2
    s.apply_action(encode_build(a_sites[0], 2))
    assert sorted(s.legal_actions()) == build_ids([a_sites[1]])


def test_R048_advance_across_two_markers_of_the_same_action() -> None:
    """R-048: marker b (P2, 1 unit) fills the last A site, then marker a (P1, 2 units) can
    only build on tier B."""
    s = H.load_doc_state(bd_doc(filled=11, M=2, owners=[1, 2], cursor=1))
    assert node(s) == ("BD", 1, 2, 1)
    last_a = lowest_tier_empty(s)
    s.apply_action(encode_build(last_a[0], 0))
    assert node(s) == ("BD", 0, 1, 2)
    assert {B.sites[a_t].tier for a_t in lowest_tier_empty(s)} == {"B"}
    assert sorted(s.legal_actions()) == build_ids(lowest_tier_empty(s))


def test_R048_advance_from_c_to_d() -> None:
    """R-048: the same recomputation at the C/D boundary (all of A, B filled, one C empty)."""
    nab = 12 + 11
    s = H.load_doc_state(bd_doc(filled=nab + 8, M=2))
    c_last = lowest_tier_empty(s)
    assert len(c_last) == 1 and B.sites[c_last[0]].tier == "C"
    s.apply_action(encode_build(c_last[0], 0))
    assert {B.sites[t].tier for t in lowest_tier_empty(s)} == {"D"}


# ---- R-049: sites full -----------------------------------------------------------------------


def test_R049_sites_full_false_while_a_site_is_empty() -> None:
    """R-049: with empty sites left after the marker, sites-full is not recorded."""
    s = H.load_doc_state(bd_doc(filled=4, M=2, clock=True))
    assert s.sites_full is False
    while s.phase == "BD":
        s.apply_action(s.legal_actions()[0])
    assert s.sites_full is False


def test_R049_filling_the_last_site_records_sites_full_and_play_continues() -> None:
    """R-049/ACTIONS 3.8: the unit that fills the last site records sites-full; the game is not
    over, and the next marker (Clock P3) resolves."""
    s = H.load_doc_state(bd_doc(filled=T - 1, M=1, clock=True))
    (last,) = empty_sites(s)
    assert s.sites_full is False
    s.apply_action(encode_build(last, 0))
    assert s.sites_full is True
    assert not s.is_terminal()
    assert s.phase == "CK" and s.current_player() == CLOCK_OWNER
    assert empty_sites(s) == []


def test_R049_remaining_units_are_lost_when_no_site_is_left() -> None:
    """R-049/ACTIONS 3.8: one empty site, power 3: one build, two units lost, no node."""
    s = H.load_doc_state(bd_doc(filled=T - 1, M=3, clock=True))
    (last,) = empty_sites(s)
    s.apply_action(encode_build(last, 2))
    assert s.phase == "CK"
    assert s.sites_full is True


def test_R049_two_empty_sites_power_three_loses_one_unit() -> None:
    """R-049: two builds happen as nodes, the third unit is lost."""
    s = H.load_doc_state(bd_doc(filled=T - 2, M=3, clock=True))
    s.apply_action(s.legal_actions()[0])
    assert s.phase == "BD" and s.sites_full is False
    assert node(s)[3] == 2
    s.apply_action(s.legal_actions()[0])
    assert s.phase == "CK" and s.sites_full is True


def test_R049_later_markers_of_the_action_do_nothing_once_full() -> None:
    """R-049/R-034: b (P2, 1 unit) fills the last site; a (P1, 2 units) then has nothing to
    build and yields no node; the Clock marker is next."""
    s = H.load_doc_state(bd_doc(filled=T - 1, M=2, owners=[1, 2], cursor=1, clock=True))
    assert node(s) == ("BD", 1, 2, 1)
    s.apply_action(s.legal_actions()[0])
    assert s.phase == "CK" and s.current_player() == CLOCK_OWNER
    assert s.sites_full is True


def test_R049_buildings_never_exceed_the_number_of_sites() -> None:
    """R-049: whatever is lost, no more than T sites hold buildings and each site holds one."""
    s = H.load_doc_state(bd_doc(filled=T - 1, M=5, owners=[0, 1, 2, 3, 0], cursor=0, clock=True))
    while s.phase == "BD":
        s.apply_action(s.legal_actions()[0])
    assert sum(1 for t in range(T) if s.site_building(t) is not None) == T


@pytest.mark.pending("R-066")
def test_R049_rest_of_the_round_resolves_then_game_ends() -> None:
    """R-049 example: Buildings fills the last site in Phase 2; Clock, Vrrooomm! and Starting
    Player still resolve, then the game ends (end type sites_full) in the same round."""
    s = H.load_doc_state(bd_doc(filled=T - 1, M=1, clock=True, tail=True))
    s.apply_action(s.legal_actions()[0])
    assert s.sites_full is True
    assert s.phase == "CK"
    steps = 0
    while not s.is_terminal():
        assert s.round == 1
        s.apply_action(s.legal_actions()[0])
        steps += 1
        assert steps < 200
    assert "sites_full" in s.end_types


# ---- Buildings strings -----------------------------------------------------------------------


@pytest.mark.parametrize("y", range(3))
def test_R047_action_to_string_build(y: int) -> None:
    """ACTIONS 4: 'P<seat>: build <type> at site <t> (<tier>, n<node>)'."""
    s = H.load_doc_state(bd_doc(filled=4, M=1))
    t = lowest_tier_empty(s)[0]
    node_id = next(n for n in range(len(B.nodes)) if t in B.node_sites[n])
    assert s.action_to_string(1, encode_build(t, y)) == (
        f"P1: build {TYPES[y]} at site {t} ({B.sites[t].tier}, n{node_id})"
    )
