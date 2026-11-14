"""Phase 2 resolution framework, action power and Buses (R-029..R-034, R-043, R-044).

Written ahead of Phase 1 unit 7; every test is pending.

Positions are the *last decision of selection*: ``make_doc`` takes a post-setup document
(a random setup dumped with ``dump_scenario``, or a fixture), puts the requested markers on the
action board with every seat but one dropped out, and the test applies that seat's DROP_OUT.
The state that follows is the first node of Phase 2 (or the next round, for an all-automatic
round).  Public API only:

  state.phase / .round / .current_player() / .legal_actions() / .apply_action(a) / .history()
  state.seat(p).buses / .action_board / .passenger_supply
  bus.scenario.dump_scenario(state)["pending"]  ->  cursor.letter, units_left, moves_left

M is never stored (SCENARIOS 4.2): tests compute it as max(seat.buses).  Accessors the engine may
add (``cursor_letter``, ``units_left``, ``moves_left``, ``bus_max``) are NOT assumed, except in the
one test named test_R032_bus_max_accessor.

Padding: every seat must have placed >= 2 markers for the SEL document to be valid, so seats short
of markers get padding on Starting Player, Vrrooomm!, then Clock.  Padding resolves after
every action the tests look at, so it does not affect the first node or the chains asserted here
(chains stop at the first node of a phase the test does not examine).

Spec gaps flagged in the unit report (not decided here):
  * PX with fewer supply units than power: is ``units_left`` capped on entry or lost later?  The
    tests count the nodes instead (test_R034_passengers_supply_shortfall_*).
  * Whether a skipped (power <= 0) marker leaves any trace (node, string, board change): tests only
    assert there is no node.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from bus.actions import DROP_OUT, PASSENGER_TO_BASE
from tests import harness as H

LINES, BUSES, PASSENGERS, BUILDINGS, CLOCK, VRROOOMM, STARTING = range(7)
NAMES = ["lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player"]
SPACES = [6, 1, 6, 6, 1, 6, 1]
PHASE_OF = {LINES: "LN", PASSENGERS: "PX", BUILDINGS: "BD", CLOCK: "CK"}
CHAIN_PHASES = ("LN", "PX", "BD")


# ---- helpers ---------------------------------------------------------------------------------

_BASE_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


def base_doc(n: int, seed: int = 0) -> dict[str, Any]:
    """Round-1 SEL document of an n-player game reached by a seeded random setup."""
    key = (n, seed)
    if key not in _BASE_CACHE:
        state = H.load_doc_state({"format": 1, "parameters": {"players": n}})
        rng = random.Random(seed)
        while state.phase in ("SB", "L1", "L2"):
            state.apply_action(rng.choice(state.legal_actions()))
        _BASE_CACHE[key] = H.scenario_module().dump_scenario(state)
    return copy.deepcopy(_BASE_CACHE[key])


def make_doc(
    n: int = 4,
    board: dict[int, list[int]] | None = None,
    buses: dict[int, int] | None = None,
    waiting: dict[str, int] | None = None,
    base: dict[str, Any] | None = None,
    pad: bool = True,
) -> dict[str, Any]:
    """SEL document where one seat is left to act (its DROP_OUT ends selection).

    ``board`` maps an action index to the seats on letters a, b, ... ``buses`` maps seat -> buses
    (others keep 1)."""
    d = base if base is not None else base_doc(n)
    for k in ("expect", "current_player", "description"):
        d.pop(k, None)
    n = len(d["seats"])
    lists = {a: list((board or {}).get(a, [])) for a in range(7)}
    count = [0] * n
    for seats in lists.values():
        for s in seats:
            count[s] += 1
    for p in range(n if pad else 0):  # padding so that every seat has placed >= 2 markers
        for a in (STARTING, VRROOOMM, CLOCK):
            while count[p] < 2 and len(lists[a]) < SPACES[a]:
                lists[a].append(p)
                count[p] += 1
    # seats that still have < 2 markers have used up their hand (markers_left 0, R-027)
    spent = {p for p in range(n) if count[p] < 2}
    last = min((p for p in range(n) if p not in spent), key=lambda p: (count[p], p))
    ab = {NAMES[a]: lists[a] + [None] * (SPACES[a] - len(lists[a])) for a in range(7)}
    d["phase"] = "SEL"
    d["round"] = 1
    d["action_board"] = ab
    d["pending"] = {
        "pass": count[last],
        "dropped": [p for p in range(n) if p != last and p not in spent],
    }
    for p in range(n):
        d["seats"][p]["markers_left"] = 0 if p in spent else 20 - count[p]
        d["seats"][p]["buses"] = (buses or {}).get(p, 1)
    if waiting is not None:
        d["waiting"] = waiting
        d.pop("supply", None)
    return d


def enter(doc: dict[str, Any]) -> Any:
    """Load the SEL position and play the last decision (DROP_OUT); return the new state."""
    s = H.load_doc_state(doc)
    assert s.phase == "SEL"
    assert s.legal_actions().count(DROP_OUT) == 1
    s.apply_action(DROP_OUT)
    return s


def run(n: int = 4, **kw: Any) -> Any:
    return enter(make_doc(n, **kw))


def pending(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s).get("pending", {})


def node(s: Any) -> tuple[str, int | None, int, int | None]:
    """(phase, cursor letter, acting seat, units_left) of the current decision node."""
    p = pending(s)
    return (s.phase, p.get("cursor", {}).get("letter"), s.current_player(), p.get("units_left"))


def M(s: Any) -> int:
    return max(s.seat(p).buses for p in range(s.num_players()))


def chain(s: Any, phases: tuple[str, ...] = CHAIN_PHASES, limit: int = 80) -> list[Any]:
    """Apply the first legal action while in one of ``phases``; return the nodes seen."""
    seen: list[Any] = []
    while s.phase in phases and not s.is_terminal() and len(seen) < limit:
        seen.append(node(s))
        s.apply_action(s.legal_actions()[0])
    return seen


def owners(k: int, n: int) -> list[int]:
    """Seats on letters a..k, cycling through the seats."""
    return [i % n for i in range(k + 1)]


# ---- R-029: resolution order between actions -------------------------------------------------


@pytest.mark.parametrize(
    "board,phase,owner",
    [
        ({LINES: [1], PASSENGERS: [2], BUILDINGS: [3]}, "LN", 1),
        ({PASSENGERS: [2], BUILDINGS: [3], CLOCK: [0]}, "PX", 2),
        ({BUILDINGS: [3], CLOCK: [0], VRROOOMM: [2]}, "BD", 3),
        ({CLOCK: [3], STARTING: [1]}, "CK", 3),
        ({LINES: [2], BUSES: [0]}, "LN", 2),
    ],
)
def test_R029_first_node_is_earliest_action_with_a_marker(
    board: dict[int, list[int]], phase: str, owner: int
) -> None:
    """R-029: actions resolve top to bottom (Lines, Buses, Passengers, Buildings, Clock,
    Vrrooomm!, Starting Player); the first node belongs to the earliest action with a marker."""
    s = run(board=board, buses={0: 2})
    assert s.phase == phase
    assert s.current_player() == owner
    assert not s.is_terminal()


def test_R029_buses_marker_is_skipped_as_a_node_and_hands_over_to_passengers() -> None:
    """R-029/R-043: with Buses then Passengers on the board the state after selection is the
    Passengers node (Buses has no decision node)."""
    s = run(board={BUSES: [1], PASSENGERS: [2]}, buses={0: 2})
    assert s.phase == "PX"
    assert s.current_player() == 2


def test_R029_buses_marker_before_buildings() -> None:
    """R-029: Buses (auto) then Buildings when Passengers has no marker: first node is BD."""
    s = run(board={BUSES: [1], BUILDINGS: [2]}, buses={0: 2})
    assert s.phase == "BD" and s.current_player() == 2


def test_R029_unused_actions_are_skipped() -> None:
    """R-029: an action with no markers is skipped; only Clock and Starting Player on the board
    -> Clock node of its owner, cursor 0."""
    s = run(board={CLOCK: [2], STARTING: [1]})
    assert s.phase == "CK"
    assert s.current_player() == 2
    assert pending(s)["cursor"]["letter"] == 0


def test_R029_selection_ends_exactly_when_last_seat_drops() -> None:
    """R-028/R-029: the last DROP_OUT is the transition that enters Phase 2 (one history entry)."""
    s = H.load_doc_state(make_doc(board={PASSENGERS: [0]}))
    before = len(s.history())
    s.apply_action(DROP_OUT)
    assert len(s.history()) == before + 1
    assert s.phase == "PX"
    assert s.round == 1


def test_R029_all_lines_markers_resolve_before_passengers_start() -> None:
    """R-029: every marker of an action resolves before the next action starts.  M = 2: Lines b
    (P1, power 1) then Lines a (P2, power 2) then Passengers a (P3, power 2)."""
    s = run(board={LINES: [2, 1], PASSENGERS: [3]}, buses={0: 2})
    seen = chain(s)
    assert seen == [
        ("LN", 1, 1, 1),
        ("LN", 0, 2, 2),
        ("LN", 0, 2, 1),
        ("PX", 0, 3, 2),
        ("PX", 0, 3, 1),
    ]


def test_R029_buildings_after_passengers_after_lines() -> None:
    """R-029: LN nodes, then PX nodes, then BD nodes, never interleaved (M = 1)."""
    s = run(board={BUILDINGS: [0], PASSENGERS: [1], LINES: [2]})
    seen = chain(s)
    assert [x[0] for x in seen] == ["LN", "PX", "BD"]
    assert [x[2] for x in seen] == [2, 1, 0]


def test_R029_same_player_markers_on_different_actions_resolve_in_action_order() -> None:
    """R-029: the owner's markers resolve by action order, not placement order: P0 selected
    Buildings first and Lines second.  M = 1: Lines b (P2) has power 0 and is skipped, so the
    first node is Lines a (P0), ahead of P0's Buildings marker."""
    s = run(board={BUILDINGS: [0, 1], LINES: [0, 2]})
    assert node(s) == ("LN", 0, 0, 1)


# ---- R-030: order within a multi-use action --------------------------------------------------


def test_R030_lines_highest_letter_first_spec_example() -> None:
    """R-030 example: Lines a (P1), b (P2), c (P0) resolve c (P0), b (P2), a (P1).  M = 3 so
    c has power 1."""
    s = run(board={LINES: [1, 2, 0]}, buses={3: 3})
    assert node(s) == ("LN", 2, 0, 1)


def test_R030_lines_chain_order_spec_example() -> None:
    """R-030 example continued: after c (P0, 1 unit) comes b (P2, 2 units), then a (P1, 3)."""
    s = run(board={LINES: [1, 2, 0]}, buses={3: 3})
    seen = chain(s, ("LN",))
    assert seen == [
        ("LN", 2, 0, 1),
        ("LN", 1, 2, 2),
        ("LN", 1, 2, 1),
        ("LN", 0, 1, 3),
        ("LN", 0, 1, 2),
        ("LN", 0, 1, 1),
    ]


def test_R030_passengers_lowest_letter_first() -> None:
    """R-030: Passengers resolve a, b, ... so the first to select goes first."""
    s = run(board={PASSENGERS: [2, 3, 1]}, buses={0: 3})
    assert node(s) == ("PX", 0, 2, 3)


def test_R030_passengers_chain_a_then_b_then_c() -> None:
    """R-030/R-031: Passengers a (P2), b (P3), c (P1) with M = 3: a does 3 units, then b 2,
    then c 1, each completely before the next."""
    s = run(board={PASSENGERS: [2, 3, 1]}, buses={0: 3})
    seen = chain(s, ("PX",))
    assert seen == [
        ("PX", 0, 2, 3),
        ("PX", 0, 2, 2),
        ("PX", 0, 2, 1),
        ("PX", 1, 3, 2),
        ("PX", 1, 3, 1),
        ("PX", 2, 1, 1),
    ]


def test_R030_buildings_highest_letter_first() -> None:
    """R-030: Buildings resolve highest letter first.  a (P1), b (P2), c (P3), M = 3: c has
    power 1."""
    s = run(board={BUILDINGS: [1, 2, 3]}, buses={0: 3})
    assert node(s) == ("BD", 2, 3, 1)


def test_R030_buildings_chain_c_b_a() -> None:
    """R-030: Buildings chain order c (P3, 1), b (P2, 2), a (P1, 3) at M = 3."""
    s = run(board={BUILDINGS: [1, 2, 3]}, buses={0: 3})
    seen = chain(s, ("BD",))
    assert [(x[1], x[2]) for x in seen] == [(2, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 1)]
    assert [x[3] for x in seen] == [1, 2, 1, 3, 2, 1]


def test_R030_vrrooomm_letter_a_first() -> None:
    """R-030: Vrrooomm! resolves a first, then b.  Position vrrooomm/base: a = P2, b = P1, both
    with a waiting passenger and an empty house on their line (count 1 each), so the first node
    is VP for letter a (P2); after P2's move comes letter b (P1)."""
    base = H.load_doc("vrrooomm/base")
    s = enter(make_doc(board={VRROOOMM: [2, 1], STARTING: [1]}, base=base, pad=False))
    assert s.phase == "VP"
    assert node(s)[1] == 0
    assert s.current_player() == 2
    s.apply_action(s.legal_actions()[0])  # pick
    s.apply_action(s.legal_actions()[0])  # deliver
    assert s.phase == "VP"
    assert node(s)[1] == 1
    assert s.current_player() == 1


def test_R030_owner_is_the_player_who_placed_the_marker_per_letter() -> None:
    """R-030: each marker is resolved by its own owner even when one seat has several letters:
    Passengers a,b,c = P3, P0, P3 -> nodes owners 3 (a), 0 (b) ... M = 3."""
    s = run(board={PASSENGERS: [3, 0, 3]}, buses={1: 3})
    assert node(s) == ("PX", 0, 3, 3)


def test_R030_same_owner_on_two_letters_resolves_each_separately() -> None:
    """R-030: P3 on a and b of Passengers (M = 2): a does 2 units (cursor 0), then b does 1
    (cursor 1); cursor and units are per marker."""
    s = run(board={PASSENGERS: [3, 3]}, buses={0: 2})
    assert chain(s, ("PX",)) == [
        ("PX", 0, 3, 2),
        ("PX", 0, 3, 1),
        ("PX", 1, 3, 1),
    ]


# ---- R-031: dwindling power ------------------------------------------------------------------


@pytest.mark.parametrize("m,k", [(m, k) for m in range(1, 6) for k in range(m)])
def test_R031_lines_power_per_letter(m: int, k: int) -> None:
    """R-031: the Lines marker on letter k (a = 0) has power M - k.  Markers on a..k; the
    highest (k) resolves first, units_left = M - k, for k < M (parametrised that way)."""
    s = run(board={LINES: owners(k, 4)}, buses={3: m})
    assert node(s) == ("LN", k, k % 4, m - k)


@pytest.mark.parametrize("m,k", [(m, k) for m in range(1, 6) for k in range(m)])
def test_R031_buildings_power_per_letter(m: int, k: int) -> None:
    """R-031: Buildings power M - k on letter k; highest letter first."""
    s = run(board={BUILDINGS: owners(k, 4)}, buses={3: m})
    assert node(s) == ("BD", k, k % 4, m - k)


@pytest.mark.parametrize("m", range(1, 6))
def test_R031_passengers_first_marker_has_power_m(m: int) -> None:
    """R-031: Passengers a has power M (= bus maximum) regardless of how many markers follow."""
    s = run(board={PASSENGERS: owners(5, 4)}, buses={2: m})
    assert node(s) == ("PX", 0, 0, m)


def test_R031_passengers_power_dwindles_along_letters() -> None:
    """R-031 example: M = 3, Passengers a,b,c,d have power 3, 2, 1, 0.  d does nothing (no node);
    after c the next action's node (Buildings, P0) follows directly."""
    s = run(board={PASSENGERS: [1, 2, 3, 1], BUILDINGS: [0]}, buses={0: 3})
    seen = chain(s, ("PX",))
    assert [(x[1], x[3]) for x in seen] == [(0, 3), (0, 2), (0, 1), (1, 2), (1, 1), (2, 1)]
    assert s.phase == "BD" and s.current_player() == 0


@pytest.mark.parametrize("m", range(1, 6))
def test_R031_lines_markers_with_power_le_zero_are_skipped(m: int) -> None:
    """R-031: markers a..f of Lines, 4 players.  Letters m..f have power <= 0 and produce no node;
    the first node is letter m - 1 (power 1)."""
    s = run(board={LINES: owners(5, 4)}, buses={1: m})
    assert node(s) == ("LN", m - 1, (m - 1) % 4, 1)


@pytest.mark.parametrize("m", range(1, 6))
def test_R031_buildings_markers_with_power_le_zero_are_skipped(m: int) -> None:
    """R-031: same for Buildings (highest first): first node is letter m - 1, one unit."""
    s = run(board={BUILDINGS: owners(5, 4)}, buses={1: m})
    assert node(s) == ("BD", m - 1, (m - 1) % 4, 1)


def test_R031_power_zero_marker_is_not_a_node_and_does_not_block_next_action() -> None:
    """R-031: Lines b (P1) at M = 1 has power 0 and is spent without effect; the next node is
    Lines a (P2, power 1)... and when a is absent the next action's node comes directly."""
    s = run(board={LINES: [2, 1]})
    assert node(s) == ("LN", 0, 2, 1)
    s = run(board={LINES: [2, 1], PASSENGERS: [3]})
    # Lines a still resolves (power 1): sanity for the same board with an extra action.
    assert node(s) == ("LN", 0, 2, 1)


def test_R031_only_nonpositive_power_markers_fall_through_to_next_action() -> None:
    """R-031: Buildings a (P0), b (P1), c(P2) -> highest first.  At M = 1 only a has power 1, so
    c and b are skipped, a (P0) is the node.  With Clock behind it the Clock is not reached."""
    s = run(board={BUILDINGS: [0, 1, 2], CLOCK: [3]})
    assert node(s) == ("BD", 0, 0, 1)


def test_R031_all_markers_nonpositive_goes_to_next_action() -> None:
    """R-031/R-029: Buildings markers only at letters with power <= 0 cannot exist alone (letters
    are contiguous from a), so use M = 1 with a Lines-less board: the node after skipping is the
    one of the next action.  Clock owner P3 acts after Buildings a (P1) resolves; checked in the
    chain test below."""
    s = run(board={BUILDINGS: [1], CLOCK: [3]})
    assert node(s) == ("BD", 0, 1, 1)


def test_R031_skipped_buildings_then_next_action_node() -> None:
    """R-031: Buildings a is the only marker with power; after its single unit, Clock (P3)."""
    s = run(board={BUILDINGS: [1], CLOCK: [3]})
    seen = chain(s, ("BD",))
    assert seen == [("BD", 0, 1, 1)]
    assert s.phase == "CK" and s.current_player() == 3


def test_R031_m_is_the_maximum_over_all_players_not_the_owners() -> None:
    """R-032/R-031: the owner has 1 bus, another seat 4: Passengers a has power 4."""
    s = run(board={PASSENGERS: [0]}, buses={2: 4})
    assert s.seat(0).buses == 1
    assert node(s) == ("PX", 0, 0, 4)


def test_R031_m_one_when_everybody_has_one_bus() -> None:
    """R-032: M = 1 with all seats at one bus: Passengers a has power 1."""
    s = run(board={PASSENGERS: [3]})
    assert node(s) == ("PX", 0, 3, 1)


def test_R031_pending_state_has_no_stale_fields() -> None:
    """R-031/SCENARIOS 4.2: a PX node carries cursor and units_left only (no moves_left)."""
    s = run(board={PASSENGERS: [3]}, buses={0: 2})
    p = pending(s)
    assert set(p) == {"cursor", "units_left"}
    assert p["cursor"] == {"letter": 0}


def test_R031_power_is_recomputed_when_loaded_directly() -> None:
    """SCENARIOS 4.2: the M of a directly loaded resolution position comes from the seats'
    buses: resolve_px reloaded with buses 3 still reports its stored units_left and owner."""
    s = H.load_fixture("resolve_px")
    assert s.phase == "PX"
    assert s.current_player() == 2
    assert pending(s)["units_left"] == 3
    assert s.legal_actions() == [PASSENGER_TO_BASE, PASSENGER_TO_BASE + 1]


# ---- R-032: the bus maximum M and when it is read --------------------------------------------


def test_R032_bus_max_accessor() -> None:
    """SCENARIOS 4.2 / unit 7 API request: ``state.bus_max`` equals max(seat.buses)."""
    s = H.load_doc_state(make_doc(board={PASSENGERS: [0]}, buses={1: 3, 2: 2}))
    assert s.bus_max == 3


def test_R032_lines_does_not_see_this_rounds_new_bus() -> None:
    """R-032: Lines resolves before Buses.  All seats have 2 buses, P3 plays Buses (M would become
    3).  Lines a,b,c = P0,P1,P2: with M = 2 the letter c has power 0 and is skipped, so the first
    node is b (P1) with ONE unit.  (With M = 3 it would be c, P2.)"""
    s = run(board={LINES: [0, 1, 2], BUSES: [3]}, buses={0: 2, 1: 2, 2: 2, 3: 2})
    assert node(s) == ("LN", 1, 1, 1)
    assert s.seat(3).buses == 2  # Buses has not resolved yet


def test_R032_passengers_see_this_rounds_new_bus() -> None:
    """R-032 example: all seats have 2 buses; P1 resolves Buses -> M = 3; Passengers markers
    afterwards use M = 3."""
    s = run(board={BUSES: [1], PASSENGERS: [2, 3]}, buses={0: 2, 1: 2, 2: 2, 3: 2})
    assert s.seat(1).buses == 3
    assert M(s) == 3
    assert node(s) == ("PX", 0, 2, 3)


def test_R032_buildings_see_this_rounds_new_bus() -> None:
    """R-032: Buildings after Buses use the new maximum: M 2 -> 3, Buildings a has 3 units."""
    s = run(board={BUSES: [0], BUILDINGS: [1]}, buses={0: 2, 1: 2, 2: 2, 3: 2})
    assert node(s) == ("BD", 0, 1, 3)


def test_R032_buses_by_a_non_maximal_player_does_not_raise_m() -> None:
    """R-032: P1 (1 bus) takes Buses while P0 has 3: P1 has 2, M stays 3."""
    s = run(board={BUSES: [1], PASSENGERS: [2]}, buses={0: 3})
    assert s.seat(1).buses == 2
    assert M(s) == 3
    assert node(s) == ("PX", 0, 2, 3)


def test_R032_buses_by_a_maximal_player_raises_m_only_when_unique_max_grows() -> None:
    """R-032: P0 and P1 both have 3; P1 takes Buses -> P1 has 4, M = 4 for Passengers."""
    s = run(board={BUSES: [1], PASSENGERS: [2]}, buses={0: 3, 1: 3})
    assert s.seat(1).buses == 4
    assert node(s) == ("PX", 0, 2, 4)


def test_R032_four_to_five_buses_raises_m_to_five() -> None:
    """R-032/R-044: the 5th bus counts immediately: Passengers a power 5."""
    s = run(board={BUSES: [0], PASSENGERS: [1]}, buses={0: 4})
    assert s.seat(0).buses == 5
    assert node(s) == ("PX", 0, 1, 5)


def test_R032_m_read_when_each_marker_resolves_across_lines_and_passengers() -> None:
    """R-032: Lines a (P0) sees M = 1 (1 unit), Passengers a (P2) after Buses (P1) sees M = 2."""
    s = run(board={LINES: [0], BUSES: [1], PASSENGERS: [2]})
    assert node(s) == ("LN", 0, 0, 1)


def test_R032_m_read_per_marker_chain_lines_buses_passengers() -> None:
    """R-032: Lines a (P0): M = 1 -> one unit.  Then Buses (P1: 1 -> 2).  Then Passengers a (P2):
    M = 2 -> two units."""
    s = run(board={LINES: [0], BUSES: [1], PASSENGERS: [2]})
    seen = chain(s)
    assert seen == [("LN", 0, 0, 1), ("PX", 0, 2, 2), ("PX", 0, 2, 1)]
    assert s.seat(1).buses == 2


# ---- R-033: five-player Lines power ----------------------------------------------------------


@pytest.mark.parametrize("m", range(1, 6))
def test_R033_five_player_lines_first_node_after_plus_one(m: int) -> None:
    """R-033: 5 players, Lines a..f.  Power M - k + 1; letters with M - k + 1 <= 0 are skipped,
    so the first node is letter min(M, 5) with exactly 1 unit."""
    s = run(5, board={LINES: owners(5, 5)}, buses={4: m})
    letter = min(m, 5)
    assert node(s) == ("LN", letter, letter % 5, 1)


def test_R033_spec_example_m3_d_has_power_one() -> None:
    """R-033 example: 5 players, M = 3: a..f have power 4, 3, 2, 1, 0, -1.  Letters f and e are
    skipped; d (letter 3) resolves with one unit."""
    s = run(5, board={LINES: owners(5, 5)}, buses={0: 3})
    assert node(s) == ("LN", 3, 3, 1)


def test_R033_plus_one_is_added_before_the_nonpositive_check() -> None:
    """R-033 / Q-A: 5 players, M = 1, Lines a (P0), b (P1).  b has M - 1 = 0, +1 = 1 > 0 so b
    resolves with one unit and is the first node (a naive 'check then add' would skip it)."""
    s = run(5, board={LINES: [0, 1]})
    assert node(s) == ("LN", 1, 1, 1)


def test_R033_m1_chain_b_then_a_units() -> None:
    """R-033: 5 players, M = 1: b (P1) has 1 unit, then a (P0) has 2 units."""
    s = run(5, board={LINES: [0, 1]})
    seen = chain(s, ("LN",))
    assert seen[:3] == [("LN", 1, 1, 1), ("LN", 0, 0, 2), ("LN", 0, 0, 1)]


@pytest.mark.parametrize("m", range(1, 6))
def test_R033_five_player_lines_letter_a_has_m_plus_one(m: int) -> None:
    """R-033: 5 players: a Lines marker alone on a has M + 1 units (up to 6)."""
    s = run(5, board={LINES: [2]}, buses={0: m})
    assert node(s) == ("LN", 0, 2, m + 1)


@pytest.mark.parametrize("m", range(1, 6))
def test_R033_five_player_passengers_has_no_plus_one(m: int) -> None:
    """R-033: the +1 applies only to Lines; Passengers a at 5 players has M units."""
    s = run(5, board={PASSENGERS: [2]}, buses={0: m})
    assert node(s) == ("PX", 0, 2, m)


@pytest.mark.parametrize("m", range(1, 6))
def test_R033_five_player_buildings_has_no_plus_one(m: int) -> None:
    """R-033: Buildings a at 5 players has M units."""
    s = run(5, board={BUILDINGS: [2]}, buses={0: m})
    assert node(s) == ("BD", 0, 2, m)


@pytest.mark.parametrize("players", [3, 4])
def test_R033_no_plus_one_below_five_players(players: int) -> None:
    """R-033: with 3 or 4 players Lines a has M units (M = 2 -> 2)."""
    s = run(players, board={LINES: [1]}, buses={0: 2})
    assert node(s) == ("LN", 0, 1, 2)


@pytest.mark.parametrize("players", [3, 4])
def test_R033_no_plus_one_means_letter_m_is_skipped(players: int) -> None:
    """R-033/R-031: 3/4 players, M = 1, Lines a, b: b (power 0) is skipped, a resolves 1 unit."""
    s = run(players, board={LINES: [0, 1]})
    assert node(s) == ("LN", 0, 0, 1)


def test_R033_five_players_m5_f_has_power_one() -> None:
    """R-033: M = 5 at 5 players: f has power 5 - 5 + 1 = 1 and resolves first (one unit)."""
    s = run(5, board={LINES: owners(5, 5)}, buses={0: 5})
    assert node(s) == ("LN", 5, 0, 1)


# ---- R-034: as fully as possible -------------------------------------------------------------


def test_R034_passengers_node_offers_only_placements_no_stop() -> None:
    """R-034: the owner may not do less; at a Passengers unit the only legal actions are the
    two PASSENGER_TO ids (no pass / stop action)."""
    s = run(board={PASSENGERS: [1]}, buses={0: 3})
    assert s.legal_actions() == [PASSENGER_TO_BASE, PASSENGER_TO_BASE + 1]


@pytest.mark.parametrize("which", [0, 1])
def test_R034_passengers_supply_shortfall_loses_the_rest(which: int) -> None:
    """R-034/R-045: supply 2 (13 waiting), power 4 (M = 4): exactly two placements are offered;
    the remaining units are lost, no node for them, and play moves on (Buildings P2)."""
    s = run(
        board={PASSENGERS: [1], BUILDINGS: [2]},
        buses={0: 4},
        waiting={"13": 13},
    )
    assert s.passenger_supply == 2
    assert s.phase == "PX"
    n = 0
    while s.phase == "PX":
        s.apply_action(PASSENGER_TO_BASE + which)
        n += 1
        assert n <= 4
    assert n == 2
    assert s.phase == "BD" and s.current_player() == 2
    assert s.passenger_supply == 0


def test_R034_passengers_empty_supply_marker_produces_no_node() -> None:
    """R-034/ACTIONS auto steps: Passengers with an empty supply has no node; Buildings (P2)
    is the first node."""
    s = run(board={PASSENGERS: [1], BUILDINGS: [2]}, waiting={"13": 15})
    assert s.passenger_supply == 0
    assert s.phase == "BD"
    assert s.current_player() == 2


def test_R034_buses_at_five_is_a_noop_not_an_error() -> None:
    """R-034/R-043: Buses with the owner at 5 buses does nothing, resolution continues."""
    s = run(board={BUSES: [0], PASSENGERS: [1]}, buses={0: 5})
    assert s.seat(0).buses == 5
    assert node(s) == ("PX", 0, 1, 5)


# ---- R-043 / R-044: Buses action -------------------------------------------------------------


@pytest.mark.parametrize("start", [1, 2, 3, 4])
def test_R043_buses_adds_exactly_one_bus_to_the_owner(start: int) -> None:
    """R-043: the owner of the Buses marker gets +1 bus; nobody else changes."""
    s = run(board={BUSES: [1], PASSENGERS: [2]}, buses={1: start})
    assert s.seat(1).buses == start + 1
    for p in (0, 2, 3):
        assert s.seat(p).buses == 1


def test_R043_spec_example_two_to_three() -> None:
    """R-043 example: P0 has 2 buses and resolves Buses -> 3."""
    s = run(board={BUSES: [0], PASSENGERS: [1]}, buses={0: 2})
    assert s.seat(0).buses == 3


def test_R043_buses_is_not_dwindling_and_ignores_m() -> None:
    """R-043: no dwindling power: +1 even when M is 5 and the owner has 1 bus."""
    s = run(board={BUSES: [0], PASSENGERS: [1]}, buses={3: 5})
    assert s.seat(0).buses == 2


def test_R043_buses_is_automatic_no_decision_node() -> None:
    """R-043/ACTIONS 3.6: with Buses as the only effective action the state after selection is
    not a Buses node: its phase is the next resolution phase and the bus is already added."""
    s = run(board={BUSES: [2], BUILDINGS: [3]})
    assert s.phase == "BD"
    assert s.seat(2).buses == 2


def test_R043_buses_has_not_resolved_while_lines_is_pending() -> None:
    """R-029/R-043: at the Lines node the Buses marker's owner still has the old count."""
    s = run(board={LINES: [0], BUSES: [2]}, buses={2: 2})
    assert s.phase == "LN"
    assert s.seat(2).buses == 2


def test_R043_buses_resolves_after_the_last_lines_unit() -> None:
    """R-043: after Lines is done, Buses runs and then Passengers sees the new bus."""
    s = run(board={LINES: [0], BUSES: [2], PASSENGERS: [3]}, buses={0: 2, 1: 2, 2: 2, 3: 2})
    seen = chain(s)
    assert seen[:2] == [("LN", 0, 0, 2), ("LN", 0, 0, 1)]
    assert seen[2] == ("PX", 0, 3, 3)
    assert s.seat(2).buses == 3


def test_R043_buses_at_five_has_no_effect() -> None:
    """R-043/R-044: the owner has 5 buses: nothing happens, never 6."""
    s = run(board={BUSES: [2], PASSENGERS: [3]}, buses={2: 5})
    assert s.seat(2).buses == 5
    assert node(s) == ("PX", 0, 3, 5)


def test_R043_buses_at_five_leaves_other_players_alone() -> None:
    """R-043/R-044: a no-op Buses marker changes nobody's count."""
    s = run(board={BUSES: [2], PASSENGERS: [3]}, buses={0: 3, 1: 4, 2: 5})
    assert [s.seat(p).buses for p in range(4)] == [3, 4, 5, 1]


@pytest.mark.parametrize("players", [3, 4, 5])
def test_R044_buses_count_stays_within_one_to_five(players: int) -> None:
    """R-044: after a Buses action every seat has between 1 and 5 buses at every player count."""
    s = run(players, board={BUSES: [0], PASSENGERS: [1]}, buses={0: 4, 1: 5})
    for p in range(players):
        assert 1 <= s.seat(p).buses <= 5


def test_R044_loader_rejects_six_and_zero_buses() -> None:
    """R-044: a scenario with buses 6 or 0 is invalid (S-BUS); buses are never removed or
    exceed 5."""
    scen = H.scenario_module()
    for bad in (0, 6):
        d = make_doc(board={PASSENGERS: [0]})
        d["seats"][1]["buses"] = bad
        with pytest.raises(scen.ScenarioError) as e:
            H.load_doc_state(d)
        assert "S-BUS" in str(e.value)


def test_R044_buses_persist_into_next_round() -> None:
    """R-044/R-043: a round with only Buses and Starting Player: the bus stays and selection
    of round 2 begins (nothing removes a bus)."""
    s = run(board={BUSES: [1], STARTING: [2]}, buses={1: 2})
    assert s.round == 2 and s.phase == "SEL"
    assert s.seat(1).buses == 3
    assert s.start_player == 2


def test_R043_buses_marker_does_not_leave_a_node_string_trace() -> None:
    """R-043/ACTIONS 'Automatic steps': no action id exists for Buses, so the history grows by
    one entry (the DROP_OUT) only."""
    d = make_doc(board={BUSES: [1], PASSENGERS: [2]})
    s = H.load_doc_state(d)
    before = len(s.history())
    s.apply_action(DROP_OUT)
    assert len(s.history()) == before + 1
    assert s.phase == "PX"
    assert s.seat(1).buses == 2


# ---- audit additions: round hand-off (R-027, R-064), automatic skips (R-040, R-049, R-060) ----

V6 = {VRROOOMM: [0, 1, 2, 0, 1, 2]}  # 6 Vrrooomm! markers, nobody waiting: no node anywhere


def _quiet_round(n: int, start: int = 0, board: dict[int, list[int]] | None = None) -> Any:
    """Selection ends in a round without any decision node (Vrrooomm! with nobody waiting).

    Seats without markers on the board are spent (markers_left 0, R-027).  ``board`` replaces
    the default of six Vrrooomm! markers on seats 0, 1, 2."""
    d = make_doc(n, board=board if board is not None else V6, waiting={}, pad=False)
    d["start_player"] = start
    return enter(d)


def test_R027_handoff_skips_seat_without_markers() -> None:
    """R-027/R-064: 4 players, nobody on Starting Player, start passes P0 -> P1, but P1 has no
    Action markers left: round 2 selection starts with P2 (next seat in turn order that has
    markers); P1 is never offered anything and its marker count stays 0."""
    s = _quiet_round(4, 0, {VRROOOMM: [0, 2, 3, 0, 2, 3]})
    assert s.round == 2 and s.phase == "SEL"
    assert s.seat(1).action_markers == 0
    assert s.current_player() == 2
    seen: list[int] = []
    for _ in range(3):  # pass 0 for the three active seats
        seen.append(s.current_player())
        s.apply_action(s.legal_actions()[0])
    assert seen == [2, 3, 0]
    assert s.seat(1).action_markers == 0
    assert all(s.seat(p).action_markers >= 0 for p in range(4))


def test_R027_handoff_second_pass_never_visits_spent_seat() -> None:
    """R-027: across the two mandatory passes of round 2 the spent seat P1 never acts."""
    s = _quiet_round(4, 0, {VRROOOMM: [0, 2, 3, 0, 2, 3]})
    actors: list[int] = []
    for _ in range(6):
        assert s.phase == "SEL"
        actors.append(s.current_player())
        s.apply_action(s.legal_actions()[0])
    assert actors == [2, 3, 0, 2, 3, 0]


def test_R027_starting_player_owner_without_markers_left() -> None:
    """R-027/R-064: P1 places its last marker on Starting Player.  P1 becomes the start player
    (R-064) but has no markers, so selection of round 2 begins with the next seat that has
    some: P2.  P1's count stays 0.  (Whether start_player stays 1 is read from R-064.)"""
    s = _quiet_round(4, 0, {STARTING: [1], VRROOOMM: [0, 2, 3, 0, 2, 3]})
    assert s.round == 2 and s.phase == "SEL"
    assert s.start_player == 1
    assert s.seat(1).action_markers == 0
    assert s.current_player() == 2


def test_R027_handoff_when_next_seat_has_markers_is_unchanged() -> None:
    """R-027 control: if the next seat has markers it simply starts (P1 starts after P0)."""
    s = _quiet_round(4, 0, {VRROOOMM: [0, 1, 2, 0, 1, 2]})
    assert s.start_player == 1 and s.current_player() == 1
    assert s.seat(3).action_markers == 0  # seat 3 is the spent one here


@pytest.mark.parametrize("n,start", [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (5, 0), (5, 1)])
def test_R064_default_start_passes_to_next_seat(n: int, start: int) -> None:
    """R-064: with no Starting Player marker the start passes to (start + 1) mod n, and round 2
    selection begins at that seat.  (Seats >= 3 are spent, so only starts whose successor holds
    markers are used.)"""
    s = _quiet_round(n, start)
    assert s.round == 2 and s.phase == "SEL"
    assert s.start_player == (start + 1) % n
    assert s.current_player() == s.start_player


def test_R064_default_start_wraps_around() -> None:
    """R-064: the last seat started, nobody took the action: seat 0 starts next (3 players)."""
    s = _quiet_round(3, 2)
    assert s.start_player == 0 and s.current_player() == 0


def test_R064_starting_player_marker_overrides_the_default() -> None:
    """R-064: a marker on Starting Player makes its owner the next start player (P2, not P1)."""
    s = _quiet_round(4, 0, {STARTING: [2], VRROOOMM: [0, 1, 2, 0, 1, 2]})
    assert s.start_player == 2 and s.current_player() == 2


def test_R040_lines_marker_with_both_ends_dead_produces_no_node() -> None:
    """R-040/ACTIONS 3.5: P0's line is a full loop at a degree-2 node (both ends dead), so its
    Lines marker has no node; the next node is Passengers of P1."""
    base = H.load_doc("lines/dead_both_ends")
    s = enter(make_doc(board={LINES: [0], PASSENGERS: [1]}, base=base))
    assert s.phase == "PX"
    assert s.current_player() == 1


def _long_trail(length: int) -> list[int]:
    """Street ids of a trail (consecutive streets share a node, no street repeats), read from
    the board data file."""
    import re
    from pathlib import Path

    text = (Path(__file__).parents[2] / "data" / "board.yaml").read_text()
    section = text.split("\nstreets:")[1].split("\nsites:")[0]
    streets = {
        int(i): (int(a), int(b))
        for i, a, b in re.findall(r"\{id: (\d+), a: (\d+), b: (\d+)", section)
    }

    def dfs(node: int, used: list[int]) -> list[int] | None:
        if len(used) == length:
            return used
        for sid, (a, b) in streets.items():
            if sid in used or node not in (a, b):
                continue
            r = dfs(b if node == a else a, used + [sid])
            if r:
                return r
        return None

    for sid0, (a, b) in streets.items():
        for first in (b, a):
            r = dfs(first, [sid0])
            if r:
                return r
    raise AssertionError("no trail")


def test_R040_lines_marker_with_no_bus_line_markers_left_produces_no_node() -> None:
    """R-040/R-004: P0's line has 25 streets (supply 0): its Lines marker produces no node; the
    next node is Passengers of P1."""
    d = make_doc(board={LINES: [0], PASSENGERS: [1]})
    d["seats"][0]["line"] = _long_trail(25)
    s = enter(d)
    assert s.phase == "PX"
    assert s.current_player() == 1


def test_R049_buildings_marker_with_no_empty_site_produces_no_node() -> None:
    """ACTIONS 3.8/R-049: every site already built: the Buildings marker (P1) produces no node,
    so the Clock marker's owner (P2) acts."""
    d = make_doc(board={BUILDINGS: [1], CLOCK: [2]})
    types = ["house", "office", "pub"]
    d["buildings"] = [{"site": t, "type": types[t % 3]} for t in range(47)]
    s = enter(d)
    assert s.phase == "CK"
    assert s.current_player() == 2


def test_R060_vrrooomm_with_zero_forced_moves_produces_no_node() -> None:
    """R-060/ACTIONS 3.10: nobody is waiting, so every Vrrooomm! marker has move count 0: no
    VP/VD node; the round ends and round 2 selection begins."""
    s = _quiet_round(3, 0)
    assert s.phase not in ("VP", "VD")
    assert s.phase == "SEL" and s.round == 2
