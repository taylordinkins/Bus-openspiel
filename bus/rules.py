"""Pure game logic (PLAN 5.1/5.3, docs/ACTIONS.md 3). No pyspiel, no I/O, no randomness.

Unit 2 implements only the setup-buildings step (SB); later units extend ``legal_actions`` and
``apply`` by dispatching on ``Core.phase``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

from bus import constants as C
from bus.actions import NUM_ACTION_KINDS, decode_action, encode_extend
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
    if core.phase == PHASE_L1:
        return _l1_legal()
    if core.phase == PHASE_L2:
        return _l2_legal(core)
    if core.phase == PHASE_SEL:
        return _sel_legal(core)
    if core.phase == PHASE_LN:
        return _ln_legal(core)
    if core.phase == PHASE_PX:
        return _px_legal()
    if core.phase == PHASE_BD:
        return _bd_legal(core)
    if core.phase == PHASE_CK:
        return _ck_legal(core)
    return []


def apply(core: Core, action: int) -> Core:
    """Apply a legal action, dispatched by phase."""
    if core.phase == PHASE_SB:
        return replace(_sb_apply(core, action), move_number=core.move_number + 1)
    if core.phase in (PHASE_L1, PHASE_L2):
        return replace(_line_apply(core, action), move_number=core.move_number + 1)
    if core.phase == PHASE_SEL:
        return replace(_sel_apply(core, action), move_number=core.move_number + 1)
    if core.phase == PHASE_LN:
        return replace(_ln_apply(core, action), move_number=core.move_number + 1)
    if core.phase == PHASE_PX:
        return replace(_px_apply(core, action), move_number=core.move_number + 1)
    if core.phase == PHASE_BD:
        return replace(_bd_apply(core, action), move_number=core.move_number + 1)
    if core.phase == PHASE_CK:
        return replace(_ck_apply(core, action), move_number=core.move_number + 1)
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
            phase = PHASE_L1
            current = order[0]
            placed = 0  # `done` restarts for L1
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


def _l1_legal() -> list[int]:
    """Implements R-017: EXTEND(0, s) for every street, occupied or not (ACTIONS 3.2)."""
    from bus.actions import encode_extend

    return [encode_extend(0, s) for s in range(len(board().streets))]


def _l2_legal(core: Core) -> list[int]:
    """Implements R-018, R-038, R-039: encoded legal extensions of the one-street line."""
    from bus.actions import encode_extend
    from bus.lines import legal_extensions

    lines = tuple(s.line for s in core.seats)
    return sorted(encode_extend(e, s) for e, s in legal_extensions(board(), lines, core.current))


def _line_apply(core: Core, action: int) -> Core:
    """Implements R-017, R-018, R-019, R-021: place a setup marker and advance the order."""
    from bus.lines import extend

    legal = _l1_legal() if core.phase == PHASE_L1 else _l2_legal(core)
    if action not in legal:
        raise ValueError(f"illegal action {action} in {core.phase}")
    kind, (e, street) = decode_action(action)
    assert kind == "extend"
    p = core.current
    seats = list(core.seats)
    line = (street,) if core.phase == PHASE_L1 else extend(board(), seats[p].line, e, street)
    seats[p] = replace(seats[p], line=line, line_markers=seats[p].line_markers - 1)
    placed = core.setup_placed + 1
    order = turn_order(core)
    phase = core.phase
    rnd = core.round
    if placed >= core.players:
        placed = 0
        if phase == PHASE_L1:
            phase, current = PHASE_L2, order[-1]
        else:
            phase, current, rnd = PHASE_SEL, order[0], 1
    elif phase == PHASE_L1:
        current = order[placed]
    else:
        current = order[core.players - 1 - placed]
    return replace(
        core, seats=tuple(seats), setup_placed=placed, phase=phase, current=current, round=rnd
    )


def _sel_placed(core: Core, seat: int) -> int:
    return sum(row.count(seat) for row in core.action_board)


def _sel_active(core: Core, seat: int) -> bool:
    return seat not in core.dropped and core.seats[seat].action_markers > 0


def _sel_next(core: Core) -> tuple[int, int] | None:
    """Next (seat, pass) to act, or None when everyone is out. Implements R-023, R-024, R-027."""
    order = turn_order(core)
    pass_ = core.sel_pass
    if not any(_sel_active(core, p) for p in order):
        return None
    while True:
        for p in order:
            if _sel_active(core, p) and _sel_placed(core, p) <= pass_:
                return p, pass_
        pass_ += 1


def _sel_legal(core: Core) -> list[int]:
    """Implements R-023..R-027, ACTIONS 3.4."""
    from bus.actions import DROP_OUT, encode_select

    if core.seats[core.current].action_markers <= 0:  # R-027 guard
        return [DROP_OUT]
    free = [
        encode_select(a) for a, row in enumerate(core.action_board) if len(row) < C.ACTION_SPACES[a]
    ]
    if core.sel_pass >= C.MANDATORY_PASSES or not free:
        return [*free, DROP_OUT]
    return free


def _sel_apply(core: Core, action: int) -> Core:
    """Implements R-023..R-028: place a marker or drop out, then advance to the next seat."""
    from bus.actions import DROP_OUT

    if action not in _sel_legal(core):
        raise ValueError(f"illegal action {action} in {core.phase}")
    p = core.current
    seats = core.seats
    board_ = core.action_board
    dropped = core.dropped
    if action == DROP_OUT:
        dropped = tuple(sorted((*dropped, p)))
    else:
        kind, (a,) = decode_action(action)
        assert kind == "select"
        board_ = board_[:a] + (board_[a] + (p,),) + board_[a + 1 :]
        seats = (
            *seats[:p],
            replace(seats[p], action_markers=seats[p].action_markers - 1),
            *seats[p + 1 :],
        )
    core = replace(core, seats=seats, action_board=board_, dropped=dropped)
    nxt = _sel_next(core)
    if nxt is None:
        return start_resolution(core)
    return replace(core, current=nxt[0], sel_pass=nxt[1])


ACT_LINES, ACT_BUSES, ACT_PASSENGERS, ACT_BUILDINGS, ACT_CLOCK, ACT_VRROOOMM, ACT_STARTING = range(
    NUM_ACTION_KINDS
)
_DESCENDING = (ACT_LINES, ACT_BUILDINGS)  # R-030: highest letter first


def bus_max(core: Core) -> int:
    """M: the largest bus count of any seat, read now. Implements R-032."""
    return max(s.buses for s in core.seats)


def marker_power(core: Core, action: int, letter: int) -> int:
    """Dwindling power M - k, +1 for Lines at 5 players. Implements R-031, R-033."""
    p = bus_max(core) - letter
    if action == ACT_LINES and core.players == C.MAX_PLAYERS:
        p += 1
    return p


def _positions(core: Core) -> list[tuple[int, int]]:
    """Markers to resolve as (action, letter), in resolution order. Implements R-029, R-030.

    Starting Player is not listed: it is handled by ``end_round``.
    """
    out: list[tuple[int, int]] = []
    for a in range(ACT_STARTING):
        letters = range(len(core.action_board[a]))
        out.extend((a, k) for k in (reversed(letters) if a in _DESCENDING else letters))
    return out


def _line_nodes(core: Core, seat: int) -> set[int]:
    from bus.lines import line_nodes

    line = core.seats[seat].line
    return set(line_nodes(board(), line)) if line else set()


def vrrooomm_moves(core: Core, seat: int) -> int:
    """Forced move count of a Vrrooomm! marker. Implements R-060."""
    nodes = _line_nodes(core, seat)
    waiting = sum(core.waiting[n] for n in nodes)
    sites = board().sites
    free = sum(
        1
        for t, ty in enumerate(core.site_types)
        if ty == core.clock and t not in core.occupied and sites[t].node in nodes
    )
    return min(core.seats[seat].buses, waiting, free)


def _enter(core: Core, action: int, letter: int) -> Core | None:
    """Start the marker (action, letter). Returns the node state, or None if it is automatic."""
    owner = core.action_board[action][letter]
    power = marker_power(core, action, letter)
    base = replace(core, cursor=letter, current=owner, units_left=0, moves_left=0, picked_node=-1)
    if action == ACT_LINES:
        entered = replace(base, phase=PHASE_LN, units_left=power)
        if power <= 0 or not _ln_legal(entered):
            return None
        return entered
    if action == ACT_PASSENGERS:
        if power <= 0 or core.supply <= 0:
            return None
        return replace(base, phase=PHASE_PX, units_left=power)
    if action == ACT_BUILDINGS:
        if power <= 0 or not lowest_open_tier_sites(core):
            return None
        return replace(base, phase=PHASE_BD, units_left=power)
    if action == ACT_CLOCK:
        return replace(base, phase=PHASE_CK)
    if action == ACT_VRROOOMM:
        moves = vrrooomm_moves(core, owner)
        return replace(base, phase=PHASE_VP, moves_left=moves) if moves > 0 else None
    raise ValueError(f"action {action} has no decision node")


def _ln_legal(core: Core) -> list[int]:
    """Implements R-040, R-042: maximal-resolution extensions for the current unit."""
    from bus.lines import maximal_extensions

    seat = core.seats[core.current]
    lines = tuple(s.line for s in core.seats)
    found = maximal_extensions(board(), lines, core.current, core.units_left, seat.line_markers)
    return sorted(encode_extend(e, s) for e, s in found)


def _ln_apply(core: Core, action: int) -> Core:
    """Implements R-042, O-2: place one street, then the next unit or the next marker."""
    from bus.lines import extend

    if action not in _ln_legal(core):
        raise ValueError(f"illegal action {action} in {core.phase}")
    _, (e, street) = decode_action(action)
    p = core.current
    seat = core.seats[p]
    seats = list(core.seats)
    seats[p] = replace(
        seat, line=extend(board(), seat.line, e, street), line_markers=seat.line_markers - 1
    )
    nxt = replace(core, seats=tuple(seats), units_left=core.units_left - 1)
    if nxt.units_left > 0 and _ln_legal(nxt):
        return nxt
    return advance(replace(nxt, units_left=0))


def sites_full(core: Core) -> bool:
    """Implements R-049: every site has a building (derived, not stored)."""
    return all(ty != EMPTY for ty in core.site_types)


def _px_legal() -> list[int]:
    """Implements R-045: one PASSENGER_TO per station."""
    from bus.actions import encode_passenger_to

    return sorted(encode_passenger_to(k) for k in range(len(board().stations)))


def _px_apply(core: Core, action: int) -> Core:
    """Implements R-034, R-045: one passenger from supply to the chosen station."""
    if action not in _px_legal():
        raise ValueError(f"illegal action {action} in {core.phase}")
    _, (k,) = decode_action(action)
    node = board().stations[k]
    waiting = core.waiting[:node] + (core.waiting[node] + 1,) + core.waiting[node + 1 :]
    nxt = replace(core, waiting=waiting, supply=core.supply - 1, units_left=core.units_left - 1)
    if nxt.units_left > 0 and nxt.supply > 0:
        return nxt
    return advance(replace(nxt, units_left=0))


def _bd_legal(core: Core) -> list[int]:
    """Implements R-010, R-046: any type on any empty site of the lowest open tier."""
    from bus.actions import encode_build

    return sorted(
        encode_build(t, y)
        for t in lowest_open_tier_sites(core)
        for y in range(len(C.BUILDING_TYPES))
    )


def _bd_apply(core: Core, action: int) -> Core:
    """Implements R-034, R-046, R-047: place one unoccupied building."""
    if action not in _bd_legal(core):
        raise ValueError(f"illegal action {action} in {core.phase}")
    _, (t, y) = decode_action(action)
    types = core.site_types[:t] + (y,) + core.site_types[t + 1 :]
    # Buildings have no owner (SCENARIOS §3.2, §9.6): Seat.buildings counts setup buildings only.
    nxt = replace(core, site_types=types, units_left=core.units_left - 1)
    if nxt.units_left > 0 and lowest_open_tier_sites(nxt):
        return nxt
    return advance(replace(nxt, units_left=0))


def move_passengers(core: Core) -> Core:
    """Step out of every building, then into clock-type buildings. Implements R-054, R-055."""
    sites = board().sites
    waiting = list(core.waiting)
    for t in core.occupied:  # R-054
        waiting[sites[t].node] += 1
    occupied: list[int] = []
    for t, ty in enumerate(core.site_types):  # ascending site id (R-055)
        node = sites[t].node
        if ty == core.clock and waiting[node] > 0:
            waiting[node] -= 1
            occupied.append(t)
    return replace(core, waiting=tuple(waiting), occupied=tuple(occupied))


def _ck_legal(core: Core) -> list[int]:
    """Implements R-050, R-052: advance, or pause (a stone is always on the clock)."""
    from bus.actions import CLOCK_ADVANCE, CLOCK_PAUSE

    assert core.stones_on_clock > 0
    return [CLOCK_ADVANCE, CLOCK_PAUSE]


def _ck_apply(core: Core, action: int) -> Core:
    """Implements R-050..R-055: clock choice, then movement unless the last stone ends the game."""
    from bus.actions import CLOCK_ADVANCE

    if action not in _ck_legal(core):
        raise ValueError(f"illegal action {action} in {core.phase}")
    if action == CLOCK_ADVANCE:  # R-051
        nxt = replace(core, clock=(core.clock + 1) % len(C.BUILDING_TYPES))
    else:  # R-052
        p = core.current
        seat = core.seats[p]
        seats = (*core.seats[:p], replace(seat, stones=seat.stones + 1), *core.seats[p + 1 :])
        nxt = replace(core, seats=seats, stones_on_clock=core.stones_on_clock - 1)
        if nxt.stones_on_clock == 0:  # R-053
            return replace(nxt, phase=PHASE_END, end_types=("last_stone",))
    return advance(move_passengers(nxt))


def _resolve_buses(core: Core, owner: int) -> Core:
    """Implements R-043, R-044: +1 bus below 5, otherwise nothing."""
    seat = core.seats[owner]
    if seat.buses >= C.BUSES_PER_PLAYER:
        return core
    seats = (*core.seats[:owner], replace(seat, buses=seat.buses + 1), *core.seats[owner + 1 :])
    return replace(core, seats=seats)


def _resolve_from(core: Core, index: int) -> Core:
    """Resolve markers from position ``index`` on until a decision node or the round's end."""
    positions = _positions(core)
    for a, k in positions[index:]:
        if a == ACT_BUSES:
            core = _resolve_buses(core, core.action_board[a][k])
            continue
        node = _enter(core, a, k)
        if node is not None:
            return node
    return end_round(core)


def start_resolution(core: Core) -> Core:
    """Selection is over (R-028): resolve from the first marker. Implements R-029."""
    return _resolve_from(replace(core, sel_pass=0, dropped=()), 0)


def advance(core: Core) -> Core:
    """The current marker is finished: go on to the next one. Implements R-029, R-030.

    Units 8-11 call this from their step when ``units_left`` / ``moves_left`` reach zero.
    """
    action = _PHASE_ACTION[core.phase]
    return _resolve_from(core, _positions(core).index((action, core.cursor)) + 1)


_PHASE_ACTION = {
    PHASE_LN: ACT_LINES,
    PHASE_PX: ACT_PASSENGERS,
    PHASE_BD: ACT_BUILDINGS,
    PHASE_CK: ACT_CLOCK,
    PHASE_VP: ACT_VRROOOMM,
    PHASE_VD: ACT_VRROOOMM,
}


def end_round(core: Core) -> Core:
    """Starting Player and round end. Implements R-064 only; unit 12 adds R-065..R-067.

    TODO(unit 12): end checks (R-066) and the END phase.
    """
    row = core.action_board[ACT_STARTING]
    start = row[0] if row else (core.start_player + 1) % core.players
    nxt = replace(
        core,
        phase=PHASE_SEL,
        round=core.round + 1,
        start_player=start,
        current=start,
        action_board=((),) * NUM_ACTION_KINDS,
        sel_pass=0,
        dropped=(),
        cursor=0,
        units_left=0,
        moves_left=0,
        picked_node=-1,
    )
    # R-027: selection opens with the first seat in turn order that still has markers.
    first = _sel_next(nxt)
    return replace(nxt, current=first[0]) if first else nxt
