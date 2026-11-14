"""Invariant checker (PLAN section 6 item 3).

``check_invariants(state, prev)`` returns one string per violated invariant; an empty list means
the state (and the step from ``prev``, if given) is clean. It reads only the public state API.
"""

from __future__ import annotations

from typing import Any

from bus import constants as C
from bus.board import load_board


def _seat_view(state: Any) -> list[Any]:
    n = state.get_game().num_players()
    return [state.seat(p) for p in range(n)]


# Every phase has rules now; a non-terminal state in any of them must have a legal action.
PLAYABLE_PHASES = ("SB", "L1", "L2", "SEL", "LN", "PX", "BD", "CK", "VP", "VD")


def check_invariants(state: Any, prev: Any = None) -> list[str]:
    """Check PLAN 6.3 invariants on ``state`` and on the (prev, state) step."""
    board = load_board()
    out: list[str] = []
    n = state.get_game().num_players()
    seats = _seat_view(state)

    # passengers: waiting + inside buildings + supply = 15; at most one per building
    waiting = sum(state.waiting_at(node.id) for node in board.nodes)
    inside = sum(1 for t in range(len(board.sites)) if state.is_occupied(t))
    if waiting + inside + state.passenger_supply != C.TOTAL_PASSENGERS:
        out.append(
            f"passengers: {waiting} waiting + {inside} inside + {state.passenger_supply} supply "
            f"!= {C.TOTAL_PASSENGERS}"
        )
    for t in range(len(board.sites)):
        if state.is_occupied(t) and state.site_building(t) is None:
            out.append(f"site {t} holds a passenger but has no building")

    # lines: valid trails, line markers + line length = 25
    for p, s in enumerate(seats):
        if (s.last_score_move is None) != (s.points == 0):
            out.append(f"seat {p}: last_score_move {s.last_score_move} vs points {s.points}")
        line = list(s.line)
        if len(set(line)) != len(line):
            out.append(f"seat {p}: line repeats a street")
        if not _is_trail(line):
            out.append(f"seat {p}: line is not a contiguous trail")
        if s.line_markers + len(line) != C.LINE_MARKERS_PER_PLAYER:
            out.append(f"seat {p}: line markers {s.line_markers} + line {len(line)} != 25")
        if not C.INITIAL_BUSES <= s.buses <= C.BUSES_PER_PLAYER:
            out.append(f"seat {p}: buses {s.buses} outside 1..{C.BUSES_PER_PLAYER}")
        if not 0 <= s.action_markers <= C.ACTION_MARKERS_PER_PLAYER:
            out.append(f"seat {p}: action markers {s.action_markers} outside 0..20")

    # tier-by-tier filling (R-010)
    seen_empty = False
    for tier in sorted({s.tier for s in board.sites}):
        sites = board.sites_of_tier(tier)
        if seen_empty and any(state.site_building(t) is not None for t in sites):
            out.append(f"tier {tier} has a building while a lower tier has an empty site")
            break
        if any(state.site_building(t) is None for t in sites):
            seen_empty = True

    # stones (R-006)
    total = state.stones_on_clock + sum(s.stones for s in seats)
    if total != C.time_stones(n):
        out.append(f"stones: {total} on clock and seats, expected {C.time_stones(n)}")

    # legal actions / terminal returns
    if state.is_terminal():
        mode = state.get_game().get_parameters().get("returns", C.DEFAULT_RETURNS)
        rets = list(state.returns())
        if mode == "win_loss" and abs(sum(rets)) > 1e-9:
            out.append(f"win_loss returns sum to {sum(rets)}, not 0")
    elif state.phase in PLAYABLE_PHASES and not state.legal_actions():
        out.append("non-terminal state has no legal action")

    if prev is not None:
        out.extend(_check_step(prev, state, n))
    return out


def _is_trail(line: list[int]) -> bool:
    board = load_board()
    if len(line) < 2:
        return True
    s0, s1 = board.streets[line[0]], board.streets[line[1]]
    shared = {s0.a, s0.b} & {s1.a, s1.b}
    if not shared:
        return False
    cur = s1.other(next(iter(shared)))
    for sid in line[2:]:
        st = board.streets[sid]
        if cur not in (st.a, st.b):
            return False
        cur = st.other(cur)
    return True


def _check_step(prev: Any, state: Any, n: int) -> list[str]:
    out: list[str] = []
    for p in range(n):
        a, b = prev.seat(p), state.seat(p)
        if b.points < a.points:
            out.append(f"seat {p}: points fell from {a.points} to {b.points}")
        if b.last_score_move != a.last_score_move and b.points == a.points:
            out.append(f"seat {p}: last_score_move changed without a point change")
        if b.points > a.points and (
            b.last_score_move is None or b.last_score_move < (a.last_score_move or 0)
        ):
            out.append(f"seat {p}: last_score_move {b.last_score_move} not advanced on scoring")
        if b.action_markers > a.action_markers:
            out.append(
                f"seat {p}: action markers rose from {a.action_markers} to {b.action_markers}"
            )
        if b.line_markers > a.line_markers:
            out.append(f"seat {p}: line markers rose from {a.line_markers} to {b.line_markers}")
    return out
