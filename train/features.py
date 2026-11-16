"""Observation v2: constant-size, ego-rotated float32 vector for 3-5 players (PLAN Phase 4).

Bus has perfect information (R-073), so every feature is read from the public state; nothing is
hidden. Built from the existing observer views (``bus.observer.BusObserver.dict``, see that
module's docstring for scaling) plus extras. Seat-indexed blocks are re-ordered so seat 0 of the
vector is the player to move, seat 1 the next seat in turn order, and so on, then zero-padded to
``MAX_SEATS`` = 5. Action ids are not seat-relative, so the policy output needs no rotation.

Blocks, in vector order (S streets, N nodes, T sites, L action letters; all from the board):
  Observer views, unchanged: phase (11), round (1), setup_placed (1), sel_pass (1),
    end_types (3), node_waiting (N), site_type (T, 3), site_occupied (T), supply (1),
    clock (3), stones_on_clock (1), units_left (1), moves_left (1), cursor (L), picked_node (N).
  Observer views with the seat axis ego-rotated and padded to 5: start_player (5), dropped (5),
    street_occ (S, 5), street_end (S, 5, 2), node_line (N, 5), seat_stones (5), seat_buses (5),
    seat_markers_left (5), seat_line_markers_left (5), seat_points (5), action_board (7, L, 5).
    (The observer's seat_to_act is dropped: it is always seat 0 after rotation.)
  Game shape: n_onehot (3) for 3/4/5 players; present (5) 1 for the first n seats.
  Scores: score (5) = (points - stones) / 10 clipped to [-1, 1] (R-068 final score if the game
    ended now); tiebreak (5): seats that have scored, ranked by earliest last_score_move
    (R-070): 1.0 for the earliest down to 1/m for the latest of m scorers, 0 for none/absent.
  Derived, all from the ego seat's view (the seat to move):
    ego_pickable (N): waiting passengers / 15 at nodes on the ego line.
    ego_free_sites (N): empty sites / 3 at nodes on the ego line (buildings reachable).
    opp_lines_at_node (N): opponent lines through the node / 4.
    street_contested (S): 1 if some opponent's line contains the street.
    frontier (S, 5): 1 if the street is a legal extension for that (rotated) seat's line
      (``bus.lines.legal_extensions``: R-036..R-039, R-041).
    vroom (3): ego waiting on line / 15, ego empty sites on line / 10, and the heuristic bot's
      expected deliveries min(buses, waiting, free) / 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bus import constants as C
from bus.actions import NUM_ACTION_KINDS, N, S, T
from bus.lines import line_nodes
from bus.observer import (
    BUS_SCALE,
    LETTERS,
    LINE_MARKER_SCALE,
    MARKER_SCALE,
    MOVES_SCALE,
    PASS_SCALE,
    POINT_SCALE,
    ROUND_SCALE,
    STONE_SCALE,
    UNITS_SCALE,
    WAITING_SCALE,
    BusObserver,
)
from bus.rules import (
    EMPTY,
    END_TYPES,
    PHASE_BD,
    PHASE_CK,
    PHASE_LN,
    PHASE_PX,
    PHASE_VD,
    PHASE_VP,
    PHASES,
    Core,
    board,
)

# Bump by hand when feature semantics change without a shape change (it is part of the layout
# hash, so older exported models are rejected as stale).
FEATURES_VERSION = 1
MAX_SEATS = C.MAX_PLAYERS
PLAYER_COUNTS = tuple(range(C.MIN_PLAYERS, C.MAX_PLAYERS + 1))
_SITES_PER_NODE = float(max(len(x) for x in board().node_sites))

# Observer views copied as is, in order.
_PLAIN = (
    "phase", "round", "setup_placed", "sel_pass", "end_types", "node_waiting", "site_type",
    "site_occupied", "supply", "clock", "stones_on_clock", "units_left", "moves_left", "cursor",
    "picked_node",
)  # fmt: skip
# Observer views with a seat axis: name -> axis index of that seat axis.
_SEATED = {
    "start_player": 0, "dropped": 0, "street_occ": 1, "street_end": 1, "node_line": 1,
    "seat_stones": 0, "seat_buses": 0, "seat_markers_left": 0, "seat_line_markers_left": 0,
    "seat_points": 0, "action_board": 2,
}  # fmt: skip
_EXTRAS: dict[str, tuple[int, ...]] = {
    "n_onehot": (len(PLAYER_COUNTS),),
    "present": (MAX_SEATS,),
    "score": (MAX_SEATS,),
    "tiebreak": (MAX_SEATS,),
    "ego_pickable": (N,),
    "ego_free_sites": (N,),
    "opp_lines_at_node": (N,),
    "street_contested": (S,),
    "frontier": (S, MAX_SEATS),
    "vroom": (3,),
}

_SHAPES: dict[str, tuple[int, ...]] = {}
_probe = BusObserver("win_loss", C.MAX_PLAYERS)
for _k in _PLAIN:
    _SHAPES[_k] = tuple(_probe.dict[_k].shape)
for _k, _ax in _SEATED.items():
    _sh = list(_probe.dict[_k].shape)
    _sh[_ax] = MAX_SEATS
    _SHAPES[_k] = tuple(_sh)
_SHAPES.update(_EXTRAS)
OBS_SIZE = sum(int(np.prod(sh)) for sh in _SHAPES.values())
assert _probe.dict["cursor"].shape[0] == LETTERS and NUM_ACTION_KINDS == 7 and T > 0
del _probe


def obs_layout() -> dict[str, tuple[int, ...]]:
    """Block name -> shape, in vector order (documentation and tests)."""
    return dict(_SHAPES)


# ---- fast encoder -------------------------------------------------------------------------
# Writes straight into one persistent buffer (no BusObserver pass, no per-block temporaries) and
# caches everything that depends only on the lines / site contents, which change rarely between
# consecutive decisions. Output is identical to the original observer-based implementation
# (tests/train/test_features_equiv.py keeps that implementation as the reference).
# The shared buffer makes ``encode`` non-reentrant across threads (workers are processes).

_OFF: dict[str, int] = {}
_off = 0
for _k, _sh in _SHAPES.items():
    _OFF[_k] = _off
    _off += int(np.prod(_sh))
_BUF = np.zeros(OBS_SIZE, np.float32)
_VIEW = {k: _BUF[_OFF[k] : _OFF[k] + int(np.prod(sh))] for k, sh in _SHAPES.items()}
_PHASE_IDX = {ph: i for i, ph in enumerate(PHASES)}
_ENDT_IDX = {t: i for i, t in enumerate(END_TYPES)}
_RESOLVING = frozenset((PHASE_LN, PHASE_PX, PHASE_BD, PHASE_CK, PHASE_VP, PHASE_VD))
_ONEHOT = {
    n: np.eye(len(PLAYER_COUNTS), dtype=np.float32)[n - C.MIN_PLAYERS] for n in PLAYER_COUNTS
}
_PRESENT = {n: (np.arange(MAX_SEATS) < n).astype(np.float32) for n in PLAYER_COUNTS}
_AB_BASE = _OFF["action_board"]
_BOARD = board()
_CACHE_CAP = 3000


class _LineInfo:
    """Per-line constants (lines recur across many states)."""

    __slots__ = ("streets", "nodes", "ends", "set", "cap_streets")

    def __init__(self, line: tuple[int, ...]) -> None:
        self.streets = list(line)
        self.nodes = line_nodes(_BOARD, line)
        self.ends = (self.nodes[0], self.nodes[-1])
        self.set = frozenset(line)
        self.cap_streets = (line[0], line[-1])


_LINES: dict[tuple[int, ...], _LineInfo] = {}


def _line_info(line: tuple[int, ...]) -> _LineInfo:
    li = _LINES.get(line)
    if li is None:
        if len(_LINES) >= 50 * _CACHE_CAP:
            _LINES.clear()
        li = _LINES[line] = _LineInfo(line)
    return li


def _frontier_streets(infos: list[_LineInfo | None], seat: int) -> set[int]:
    """Streets of ``bus.lines.legal_extensions`` (R-036..R-039, R-041) for ``seat``'s line."""
    mine = infos[seat]
    assert mine is not None
    occupied: set[int] = set()
    shared: dict[int, set[int]] = {}
    for other, li in enumerate(infos):
        if other == seat or li is None:
            continue
        occupied |= li.set
        shared.setdefault(li.ends[0], set()).add(li.cap_streets[0])
        shared.setdefault(li.ends[1], set()).add(li.cap_streets[1])
    result: set[int] = set()
    incident = _BOARD.incident
    for node in mine.ends:
        cands = [x for x in incident[node] if x not in mine.set]
        free = [x for x in cands if x not in occupied]
        if not free:
            result.update(cands)
            continue
        result.update(free)
        extra = shared.get(node)
        if extra:
            result.update(x for x in cands if x in extra)
    return result


class _Geom:
    """Everything derived from (all seats' lines, ego seat); see ``_geometry``."""

    __slots__ = ("occ", "end", "node_line", "contested", "opp_at", "frontier", "mine_idx")

    def __init__(self, lines: tuple[tuple[int, ...], ...], me: int) -> None:
        n = len(lines)
        infos = [_line_info(line) if line else None for line in lines]
        occ_i: list[int] = []
        end_i: list[int] = []
        nl_i: list[int] = []
        con_i: list[int] = []
        opp_i: list[int] = []
        fr_i: list[int] = []
        mine: list[int] = []
        for j in range(n):
            seat = (me + j) % n
            li = infos[seat]
            if li is None:
                continue
            occ_i += [x * MAX_SEATS + j for x in li.streets]
            first, last = li.cap_streets
            end_i += [(first * MAX_SEATS + j) * 2, (last * MAX_SEATS + j) * 2 + 1]
            nl_i += [x * MAX_SEATS + j for x in li.nodes]
            if seat == me:
                mine = sorted(set(li.nodes))
            else:
                con_i += li.streets
                opp_i += li.nodes
            fr_i += [x * MAX_SEATS + j for x in _frontier_streets(infos, seat)]
        self.occ = np.zeros(S * MAX_SEATS, np.float32)
        self.occ[occ_i] = 1.0
        self.end = np.zeros(S * MAX_SEATS * 2, np.float32)
        self.end[end_i] = 1.0
        self.node_line = np.zeros(N * MAX_SEATS, np.float32)
        self.node_line[nl_i] = 1.0
        self.contested = np.zeros(S, np.float32)
        self.contested[con_i] = 1.0
        self.opp_at = np.zeros(N, np.float32)
        for node in opp_i:  # repeated nodes accumulate; exact (multiples of 0.25)
            self.opp_at[node] += 1.0 / (MAX_SEATS - 1)
        self.frontier = np.zeros(S * MAX_SEATS, np.float32)
        self.frontier[fr_i] = 1.0
        self.mine_idx = np.array(mine, np.intp)


_GEOM: dict[tuple[Any, ...], _Geom] = {}


def _geometry(core: Core, me: int) -> _Geom:
    key = (me, *[s.line for s in core.seats])
    g = _GEOM.get(key)
    if g is None:
        if len(_GEOM) >= _CACHE_CAP:
            _GEOM.clear()
        g = _GEOM[key] = _Geom(key[1:], me)
    return g


class _Sites:
    """Per site-contents cache: one-hot indices into the flat buffer and free sites per node."""

    __slots__ = ("onehot_idx", "free", "empty")

    def __init__(self, site_types: tuple[int, ...]) -> None:
        stp = np.array((*site_types, EMPTY + 1), np.int64)  # trailing pad is never empty
        nz = np.flatnonzero(stp[:-1] != EMPTY)
        self.onehot_idx = _OFF["site_type"] + nz * _NTYPES_ + stp[nz]
        empty = (stp[_NODE_SITES] == EMPTY).sum(axis=1)
        self.empty = empty
        self.free = (empty / _SITES_PER_NODE).astype(np.float32)


_NTYPES_ = len(C.BUILDING_TYPES)
_NODE_SITES = np.full((N, int(_SITES_PER_NODE)), T, np.intp)  # pad index T -> never-empty entry
for _i, _ts in enumerate(_BOARD.node_sites):
    _NODE_SITES[_i, : len(_ts)] = _ts
_SITES: dict[tuple[int, ...], _Sites] = {}
_OCC: dict[tuple[int, ...], np.ndarray[Any, Any]] = {}


def encode(core: Core) -> np.ndarray[Any, Any]:
    """Observation v2 for the seat to move in ``core`` (see module docstring)."""
    n = core.players
    me = core.current
    seats = core.seats
    B = _BUF
    V = _VIEW
    O = _OFF  # noqa: E741
    B.fill(0.0)
    B[O["phase"] + _PHASE_IDX[core.phase]] = 1.0
    B[O["round"]] = min(core.round / ROUND_SCALE, 1.0)
    B[O["setup_placed"]] = core.setup_placed / (2 * n)
    B[O["sel_pass"]] = min(core.sel_pass / PASS_SCALE, 1.0)
    for et in core.end_types:
        B[O["end_types"] + _ENDT_IDX[et]] = 1.0
    nw = V["node_waiting"]
    wi = np.array(core.waiting)
    nw[:] = wi
    nw /= WAITING_SCALE
    skey = core.site_types
    sc = _SITES.get(skey)
    if sc is None:
        if len(_SITES) >= _CACHE_CAP:
            _SITES.clear()
        sc = _SITES[skey] = _Sites(skey)
    B[sc.onehot_idx] = 1.0
    if core.occupied:
        oi = _OCC.get(core.occupied)
        if oi is None:
            if len(_OCC) >= _CACHE_CAP:
                _OCC.clear()
            oi = _OCC[core.occupied] = np.array(core.occupied, np.intp) + O["site_occupied"]
        B[oi] = 1.0
    B[O["supply"]] = core.supply / WAITING_SCALE
    B[O["clock"] + core.clock] = 1.0
    B[O["stones_on_clock"]] = core.stones_on_clock / STONE_SCALE
    B[O["units_left"]] = core.units_left / UNITS_SCALE
    B[O["moves_left"]] = core.moves_left / MOVES_SCALE
    if core.phase in _RESOLVING:
        B[O["cursor"] + core.cursor] = 1.0
    if core.picked_node >= 0:
        B[O["picked_node"] + core.picked_node] = 1.0

    B[O["start_player"] + (core.start_player - me) % n] = 1.0
    for p in core.dropped:
        B[O["dropped"] + (p - me) % n] = 1.0
    ab = _AB_BASE
    for a, row in enumerate(core.action_board):
        base = ab + a * LETTERS * MAX_SEATS
        for k, seat_id in enumerate(row):
            B[base + k * MAX_SEATS + (seat_id - me) % n] = 1.0
    ordered = [seats[(me + j) % n] for j in range(n)]
    V["seat_stones"][:n] = [s.stones / STONE_SCALE for s in ordered]
    V["seat_buses"][:n] = [s.buses / BUS_SCALE for s in ordered]
    V["seat_markers_left"][:n] = [s.action_markers / MARKER_SCALE for s in ordered]
    V["seat_line_markers_left"][:n] = [s.line_markers / LINE_MARKER_SCALE for s in ordered]
    V["seat_points"][:n] = [min(s.points / POINT_SCALE, 1.0) for s in ordered]
    V["score"][:n] = [min(1.0, max(-1.0, (s.points - s.stones) / 10.0)) for s in ordered]
    B[O["n_onehot"] : O["n_onehot"] + 3] = _ONEHOT[n]
    B[O["present"] : O["present"] + MAX_SEATS] = _PRESENT[n]
    scored = [(s.last_score_move, i) for i, s in enumerate(seats) if s.last_score_move is not None]
    if scored:
        scored.sort()
        m = len(scored)
        tb = V["tiebreak"]
        for r, (_, seat) in enumerate(scored):
            tb[(seat - me) % n] = (m - r) / m

    g = _geometry(core, me)
    V["street_occ"][:] = g.occ
    V["street_end"][:] = g.end
    V["node_line"][:] = g.node_line
    V["street_contested"][:] = g.contested
    V["opp_lines_at_node"][:] = g.opp_at
    V["frontier"][:] = g.frontier
    mi = g.mine_idx
    waiting_on = 0
    free_on = 0
    if len(mi):
        V["ego_pickable"][mi] = nw[mi]
        V["ego_free_sites"][mi] = sc.free[mi]
        waiting_on = int(wi[mi].sum())
        free_on = int(sc.empty[mi].sum())
    deliveries = min(seats[me].buses, waiting_on, free_on)
    vr = O["vroom"]
    B[vr] = waiting_on / C.TOTAL_PASSENGERS
    B[vr + 1] = free_on / 10.0
    B[vr + 2] = deliveries / float(C.BUSES_PER_PLAYER)
    return B.copy()
