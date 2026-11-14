"""Pure game logic (PLAN 5.1/5.3, docs/ACTIONS.md 3). No pyspiel, no I/O, no randomness.

Unit 2 implements only the setup-buildings step (SB); later units extend ``legal_actions`` and
``apply`` by dispatching on ``Core.phase``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

from bus import constants as C
from bus.actions import NUM_ACTION_KINDS, decode_action
from bus.board import Board, load_board

PHASE_SB = "SB"  # setup buildings (R-016)
PHASE_L1 = "L1"  # first setup line markers (R-017); not implemented yet
PHASE_L2 = "L2"
PHASE_SEL = "SEL"
PHASE_LN = "LN"
PHASE_PX = "PX"
PHASE_BD = "BD"
PHASE_CK = "CK"
PHASE_VP = "VP"
PHASE_VD = "VD"
PHASE_END = "END"
PHASES = (
    PHASE_SB,
    PHASE_L1,
    PHASE_L2,
    PHASE_SEL,
    PHASE_LN,
    PHASE_PX,
    PHASE_BD,
    PHASE_CK,
    PHASE_VP,
    PHASE_VD,
    PHASE_END,
)
SETUP_PHASES = (PHASE_SB, PHASE_L1, PHASE_L2)
END_TYPES = ("sites_full", "last_stone", "markers")  # R-066

SETUP_BUILDINGS_PER_SEAT = 2  # R-016
EMPTY = -1  # site_types entry for an empty site


@dataclass(frozen=True)
class Seat:
    """One seat's holdings (R-004, R-014)."""

    points: int = 0
    buses: int = C.INITIAL_BUSES
    action_markers: int = C.ACTION_MARKERS_PER_PLAYER
    line_markers: int = C.LINE_MARKERS_PER_PLAYER
    stones: int = 0
    buildings: int = 0
    line: tuple[int, ...] = ()
    last_score_move: int | None = None  # R-062, R-070


@dataclass(frozen=True)
class Core:
    """Whole game state as plain data."""

    players: int
    seats: tuple[Seat, ...]
    site_types: tuple[int, ...]  # per site: EMPTY or building type index
    waiting: tuple[int, ...]  # per node: passengers standing there
    supply: int
    clock: int  # index into BUILDING_TYPES
    stones_on_clock: int
    action_board: tuple[tuple[int, ...], ...]  # per action kind: seats on the letters in order
    round: int
    phase: str
    current: int
    setup_placed: int  # `done` of SB / L1 / L2 (buildings, first or second markers placed)
    start_player: int = 0  # R-019, R-064
    move_number: int = 0  # moves played so far, counting a loaded scenario's move_offset
    sel_pass: int = 0  # SEL: 0-based pass (R-023)
    dropped: tuple[int, ...] = ()  # SEL: seats that dropped out this round
    cursor: int = 0  # resolution: 0-based letter of the marker being resolved
    units_left: int = 0  # LN / PX / BD: units still to do, pending one included
    moves_left: int = 0  # VP / VD
    picked_node: int = -1  # VD: node of the picked passenger
    occupied: tuple[int, ...] = ()  # sorted sites whose building holds a passenger
    end_types: tuple[str, ...] = ()  # END: R-066 end types


@cache
def board() -> Board:
    return load_board()


def initial_core(players: int) -> Core:
    """Initial state. Implements R-009, R-014, R-015, R-019."""
    if not C.MIN_PLAYERS <= players <= C.MAX_PLAYERS:
        raise ValueError(f"players must be {C.MIN_PLAYERS}..{C.MAX_PLAYERS}, got {players}")
    b = board()
    waiting = tuple(1 if n in b.roundabouts else 0 for n in range(len(b.nodes)))
    return Core(
        players=players,
        seats=(Seat(),) * players,
        site_types=(EMPTY,) * len(b.sites),
        waiting=waiting,
        supply=C.TOTAL_PASSENGERS - sum(waiting),
        clock=0,
        stones_on_clock=C.time_stones(players),
        action_board=((),) * NUM_ACTION_KINDS,
        round=0,
        phase=PHASE_SB,
        current=0,
        setup_placed=0,
    )


def lowest_open_tier_sites(core: Core) -> list[int]:
    """Empty sites of the lowest open tier. Implements R-010."""
    b = board()
    empty = [t for t, ty in enumerate(core.site_types) if ty == EMPTY]
    if not empty:
        return []
    low = min(b.sites[t].tier for t in empty)
    return [t for t in empty if b.sites[t].tier == low]


def turn_order(core: Core) -> tuple[int, ...]:
    """Seats in turn order from the start player. Implements R-016, R-019."""
    return tuple((core.start_player + i) % core.players for i in range(core.players))


def legal_actions(core: Core) -> list[int]:
    """Sorted legal action ids, dispatched by phase."""
    if core.phase == PHASE_SB:
        return _sb_legal(core)
    return []


def apply(core: Core, action: int) -> Core:
    """Apply a legal action, dispatched by phase."""
    if core.phase == PHASE_SB:
        return replace(_sb_apply(core, action), move_number=core.move_number + 1)
    raise ValueError(f"phase {core.phase} not implemented")


def _sb_legal(core: Core) -> list[int]:
    """Implements R-010, R-016: any type on any lowest-open-tier site."""
    from bus.actions import encode_build

    return sorted(
        encode_build(t, y)
        for t in lowest_open_tier_sites(core)
        for y in range(len(C.BUILDING_TYPES))
    )


def _sb_apply(core: Core, action: int) -> Core:
    """Place a setup building; each seat places two in a row, in seat order (R-016)."""
    if action not in _sb_legal(core):
        raise ValueError(f"illegal action {action} in {core.phase}")
    kind, (t, y) = decode_action(action)
    assert kind == "build"
    types = core.site_types[:t] + (y,) + core.site_types[t + 1 :]
    p = core.current
    seats = list(core.seats)
    seats[p] = replace(seats[p], buildings=seats[p].buildings + 1)
    placed = core.setup_placed + 1
    order = turn_order(core)
    nxt = placed // SETUP_BUILDINGS_PER_SEAT
    phase = core.phase
    current = p
    if placed % SETUP_BUILDINGS_PER_SEAT == 0:
        if nxt >= core.players:
            phase = PHASE_L1  # unit 5 continues from here
            current = order[0]
        else:
            current = order[nxt]
    return replace(
        core,
        site_types=types,
        seats=tuple(seats),
        setup_placed=placed,
        phase=phase,
        current=current,
    )
