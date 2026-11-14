"""Line geometry as pure functions (R-035..R-039, R-041; ACTIONS 1.3).

A line is an ordered tuple of street ids. End 0 is the start of the stored path, end 1 the
last street. Extending end 0 prepends, extending end 1 appends. The 25-marker supply limit and
maximal resolution (R-040, R-042) are the caller's job.
"""

from __future__ import annotations

from collections.abc import Sequence

from bus.board import Board


def _walk(board: Board, line: Sequence[int], start: int) -> list[int] | None:
    nodes = [start]
    cur = start
    for sid in line:
        st = board.streets[sid]
        if cur not in (st.a, st.b):
            return None
        cur = st.other(cur)
        nodes.append(cur)
    return nodes


def line_nodes(board: Board, line: Sequence[int]) -> list[int]:
    """Node walk of the line: ``len(line) + 1`` nodes, index 0 is end 0's node."""
    if not line:
        raise ValueError("empty line has no nodes")
    first = board.streets[line[0]]
    if len(line) == 1:
        return [first.a, first.b]
    second = board.streets[line[1]]
    shared = (second.a, second.b)
    starts = [n for n in (first.a, first.b) if n not in shared] or [first.a, first.b]
    for start in starts:
        nodes = _walk(board, line, start)
        if nodes is not None:
            return nodes
    raise ValueError(f"streets {list(line)} do not form a path")


def line_ends(board: Board, line: Sequence[int]) -> tuple[int, int]:
    """Nodes of end 0 and end 1."""
    nodes = line_nodes(board, line)
    return nodes[0], nodes[-1]


def legal_extensions(
    board: Board, lines: Sequence[Sequence[int]], seat: int
) -> frozenset[tuple[int, int]]:
    """Legal ``(end, street)`` pairs for ``seat`` under R-036..R-039 and R-041."""
    mine = lines[seat]
    if not mine:
        return frozenset()
    occupied: set[int] = set()
    # node -> last streets of other seats' lines that end there (R-039)
    shared: dict[int, set[int]] = {}
    for other, line in enumerate(lines):
        occupied.update(line)
        if other == seat or not line:
            continue
        n0, n1 = line_ends(board, line)
        shared.setdefault(n0, set()).add(line[0])
        shared.setdefault(n1, set()).add(line[-1])
    own = set(mine)
    result: set[tuple[int, int]] = set()
    for end, node in enumerate(line_ends(board, mine)):
        cands = [s for s in board.incident[node] if s not in own]
        free = [s for s in cands if s not in occupied]
        if not free:
            result.update((end, s) for s in cands)
            continue
        result.update((end, s) for s in free)
        extra = shared.get(node)
        if extra:
            result.update((end, s) for s in cands if s in extra)
    return frozenset(result)


def extend(board: Board, line: Sequence[int], end: int, street: int) -> tuple[int, ...]:
    """The line with ``street`` added at ``end`` (0 prepends, 1 appends); input is untouched."""
    if end not in (0, 1):
        raise ValueError(f"end must be 0 or 1, got {end}")
    if street in line:
        raise ValueError(f"street {street} is already in the line")
    node = line_ends(board, line)[end]
    st = board.streets[street]
    if node not in (st.a, st.b):
        raise ValueError(f"street {street} is not incident to node {node}")
    return (street, *line) if end == 0 else (*line, street)


def _best_continuation(
    board: Board,
    lines: tuple[tuple[int, ...], ...],
    seat: int,
    depth: int,
    memo: dict[tuple[tuple[int, ...], int], int],
) -> int:
    """Most further placements (at most ``depth``) from ``lines`` along legal continuations."""
    if depth <= 0:
        return 0
    key = (lines[seat], depth)
    cached = memo.get(key)
    if cached is not None:
        return cached
    best = 0
    for end, street in sorted(legal_extensions(board, lines, seat)):
        nxt = (*lines[:seat], extend(board, lines[seat], end, street), *lines[seat + 1 :])
        best = max(best, 1 + _best_continuation(board, nxt, seat, depth - 1, memo))
        if best >= depth:
            break
    memo[key] = best
    return best


def max_placements(board: Board, lines: Sequence[Sequence[int]], seat: int, depth: int) -> int:
    """Largest number (at most ``depth``) of consecutive legal placements for ``seat`` (R-042)."""
    frozen = tuple(tuple(line) for line in lines)
    return _best_continuation(board, frozen, seat, depth, {})


def maximal_extensions(
    board: Board, lines: Sequence[Sequence[int]], seat: int, units: int, supply: int
) -> frozenset[tuple[int, int]]:
    """Legal ``(end, street)`` choices for the next unit under maximal resolution (R-040, R-042).

    ``units`` is the remaining power including this unit, ``supply`` the seat's Bus Line markers
    left. Only choices that keep the maximum achievable number of placements reachable remain.
    """
    reach = min(units, supply)
    if reach <= 0:
        return frozenset()
    frozen = tuple(tuple(line) for line in lines)
    memo: dict[tuple[tuple[int, ...], int], int] = {}
    scored: dict[tuple[int, int], int] = {}
    for end, street in legal_extensions(board, frozen, seat):
        nxt = (*frozen[:seat], extend(board, frozen[seat], end, street), *frozen[seat + 1 :])
        scored[(end, street)] = 1 + _best_continuation(board, nxt, seat, reach - 1, memo)
    if not scored:
        return frozenset()
    top = max(scored.values())
    return frozenset(c for c, v in scored.items() if v == top)
