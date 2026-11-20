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

Observation v3 (``FEATURES_V3`` = 2, ``encode_v3``, ``OBS_SIZE_V3``; Phase 4 interaction work):
the v2 vector unchanged (same blocks and offsets), then the blocks below. Seat axes are
ego-rotated and padded to 5 like v2; everything is read from the public state (R-073). Remaining
action and line markers per seat are already in v2 (seat_markers_left, seat_line_markers_left).
  Per-seat versions of v2's ego-only derived features:
    seat_pickable (N, 5): waiting passengers / 15 at nodes on that seat's line (R-035, R-058).
    seat_free_sites (N, 5): empty sites at nodes on that seat's line, divided by the board's
      most sites at one node (as v2's ego_free_sites; R-010, R-011).
    seat_vroom (5, 3): v2's vroom triple for every seat.
    seat_vroom_now (5): the R-060 move count a Vrrooomm! marker of that seat would have now,
      min(buses, waiting on line, unoccupied buildings of the clock's type on line) / 5
      (equals ``bus.rules.vrrooomm_moves``; R-057..R-060).
    seat_targets (5, 3): unoccupied buildings of each type on that seat's line / 10 (the R-059
      destinations under each clock type).
    seat_reach (N, 5): nodes not yet on that seat's line that one legal extension would add
      (far ends of its frontier streets; R-036..R-039, R-041).
  Interaction between seats:
    contested_node (N, 3): at nodes on the ego line that are also on some opponent's line:
      waiting / 15 and unoccupied buildings / most sites per node; and, at nodes on or one
      extension from the ego line, the number of opponents / 4 that one extension would bring
      there.
    pair_overlap (5, 4): per opponent (ego row 0 is zero): shared line nodes / 10, waiting at
      shared nodes / 15, unoccupied buildings at shared nodes / 10, streets both lines hold / 10
      (R-013, R-037).
    frontier_contested (S): on the ego's legal extension streets, the number of opponents / 4
      whose legal extensions include the same street (the race for streets, R-038, R-039).
  Action board and resolution order (R-029, R-030; ``bus.rules.resolution_positions``):
    res_before / res_after (7, 5): per action, that seat's markers resolving before / after the
      ego's first marker on that action, / 6 (zero where the ego has no marker there).
    res_pending (7, 5): that seat's markers not yet resolved this round / 6 (all during
      selection; from the marker being resolved on during resolution; Starting Player counts
      until round end, R-064).
    place_now (7, 2): for a marker placed now (R-025): its power if it resolved with the current
      M (R-031..R-033) / 6 for Lines, Passengers, Buildings, 1 for the other actions, 0 if full;
      and 1 if it would resolve before every marker already there (R-030).
    sel_state (5, 2): still selecting this round (R-024, R-027); markers on the board / 6.
  score_gap (3): ego score (R-068 if the game ended now) minus the best opponent's / 10 and
    minus the opponents' mean / 10 (both clipped to [-1, 1], the R-072 shape), and the pairwise
    rank term 2 (beaten + ties / 2) / (n - 1) - 1.

Versions: ``FEATURES_VERSION`` (the default for new PPO runs, ``encode``, ``OBS_SIZE``) stays v2,
so existing runs, models and tests are unchanged. Nets select their encoder from their stored
layout (``bots.net_forward.NetWeights.features_version`` / ``encoder(version)``).
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
    _DESCENDING,  # pyright: ignore[reportPrivateUsage]
    ACT_BUILDINGS,
    ACT_LINES,
    ACT_PASSENGERS,
    ACT_STARTING,
    EMPTY,
    END_TYPES,
    PHASE_BD,
    PHASE_CK,
    PHASE_LN,
    PHASE_PX,
    PHASE_SEL,
    PHASE_VD,
    PHASE_VP,
    PHASES,
    Core,
    board,
    marker_power,
    resolution_positions,
    resolving_action,
)

# Bump by hand when feature semantics change without a shape change (it is part of the layout
# hash, so older exported models are rejected as stale). This is the default version (v2 layout);
# v3 is FEATURES_V3 below, chosen per net or per run.
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


def obs_layout(version: int | None = None) -> dict[str, tuple[int, ...]]:
    """Block name -> shape, in vector order, of ``version`` (default ``FEATURES_VERSION``, read
    at call time)."""
    v = FEATURES_VERSION if version is None else version
    return dict(_LAYOUTS[_check(v)])


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
    """Observation v2 (``FEATURES_V2``) for the seat to move in ``core`` (module docstring)."""
    _fill_v2(core, _BUF, _VIEW)
    return _BUF.copy()


def _fill_v2(
    core: Core, B: np.ndarray[Any, Any], V: dict[str, np.ndarray[Any, Any]]
) -> tuple[_Geom, _Sites, np.ndarray[Any, Any]]:
    """Write the v2 blocks into ``B`` (zeroed first; v2 offsets, so also the prefix of a v3
    buffer) and return what the v3 blocks reuse: geometry, site cache, waiting counts."""
    n = core.players
    me = core.current
    seats = core.seats
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
    return g, sc, wi


# ---- observation v3 (FEATURES_V3): v2 + per-seat derived and interaction blocks -----------
# The v3 vector is the v2 vector (same blocks, same offsets) followed by the blocks below, so a
# v2 first layer maps onto the first OBS_SIZE inputs (``input_columns``, warm start).

_V3_EXTRAS: dict[str, tuple[int, ...]] = {
    "seat_pickable": (N, MAX_SEATS),
    "seat_free_sites": (N, MAX_SEATS),
    "seat_vroom": (MAX_SEATS, 3),
    "seat_vroom_now": (MAX_SEATS,),
    "seat_targets": (MAX_SEATS, len(C.BUILDING_TYPES)),
    "seat_reach": (N, MAX_SEATS),
    "contested_node": (N, 3),
    "pair_overlap": (MAX_SEATS, 4),
    "frontier_contested": (S,),
    "res_before": (NUM_ACTION_KINDS, MAX_SEATS),
    "res_after": (NUM_ACTION_KINDS, MAX_SEATS),
    "res_pending": (NUM_ACTION_KINDS, MAX_SEATS),
    "place_now": (NUM_ACTION_KINDS, 2),
    "sel_state": (MAX_SEATS, 2),
    "score_gap": (3,),
}
_SHAPES_V3: dict[str, tuple[int, ...]] = {**_SHAPES, **_V3_EXTRAS}
OBS_SIZE_V3 = sum(int(np.prod(sh)) for sh in _SHAPES_V3.values())
_OFF3: dict[str, int] = {}
_off = 0
for _k, _sh in _SHAPES_V3.items():
    _OFF3[_k] = _off
    _off += int(np.prod(_sh))
assert all(_OFF3[k] == _OFF[k] for k in _SHAPES)  # v2 prefix: v2 caches' absolute indices hold
_BUF3 = np.zeros(OBS_SIZE_V3, np.float32)
_VIEW3 = {k: _BUF3[_OFF3[k] : _OFF3[k] + int(np.prod(sh))] for k, sh in _SHAPES_V3.items()}
_NODE_OF_SITE = np.array([st.node for st in _BOARD.sites], np.intp)
_MULTI = (ACT_LINES, ACT_PASSENGERS, ACT_BUILDINGS)  # dwindling power (R-031)
_FIRST_IF_FREE = frozenset(_DESCENDING)  # a new marker there resolves first (R-030)
_LETTER_SCALE = float(max(C.ACTION_SPACES))
_POWER_SCALE = float(C.BUSES_PER_PLAYER + 1)  # M <= 5, +1 for 5-player Lines (R-033)


class _Geom3:
    """Per-(lines, ego) v3 geometry: who can reach what, built on the v2 ``_Geom``."""

    __slots__ = (
        "node_line", "reach", "shared", "shared_nodes", "shared_streets", "shared_mask",
        "frontier_contested", "opp_reach_near",
    )  # fmt: skip

    def __init__(self, lines: tuple[tuple[int, ...], ...], me: int, g: _Geom) -> None:
        n = len(lines)
        infos = [_line_info(line) if line else None for line in lines]
        nl = g.node_line.reshape(N, MAX_SEATS)
        self.node_line = nl
        reach = np.zeros((N, MAX_SEATS), np.float32)
        streets = _BOARD.streets
        for j in range(n):
            seat = (me + j) % n
            li = infos[seat]
            if li is None:
                continue
            on = set(li.nodes)
            for x in _frontier_streets(infos, seat):
                for node in (streets[x].a, streets[x].b):
                    if node not in on:
                        reach[node, j] = 1.0
        self.reach = reach.reshape(-1)
        shared = nl * nl[:, :1]
        shared[:, 0] = 0.0
        self.shared = shared
        self.shared_nodes = shared.sum(axis=0)
        occ = g.occ.reshape(S, MAX_SEATS)
        ss = occ[:, 0] @ occ
        ss[0] = 0.0
        self.shared_streets = ss
        self.shared_mask = nl[:, 0] * (g.opp_at > 0)
        fr = g.frontier.reshape(S, MAX_SEATS)
        self.frontier_contested = fr[:, 1:].sum(axis=1) / (MAX_SEATS - 1) * fr[:, 0]
        near = np.maximum(nl[:, 0], reach[:, 0])
        self.opp_reach_near = reach[:, 1:].sum(axis=1) / (MAX_SEATS - 1) * near


_GEOM3: dict[tuple[Any, ...], _Geom3] = {}


def _geometry3(core: Core, me: int, g: _Geom) -> _Geom3:
    key = (me, *[s.line for s in core.seats])
    g3 = _GEOM3.get(key)
    if g3 is None:
        if len(_GEOM3) >= _CACHE_CAP:
            _GEOM3.clear()
        g3 = _GEOM3[key] = _Geom3(key[1:], me, g)
    return g3


_BLD: dict[tuple[Any, ...], tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}


def _buildings(core: Core) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """(N, 3) unoccupied buildings per node and type, and their per-node total (cached)."""
    key = (core.site_types, core.occupied)
    hit = _BLD.get(key)
    if hit is None:
        if len(_BLD) >= _CACHE_CAP:
            _BLD.clear()
        bl = np.zeros((N, len(C.BUILDING_TYPES)), np.float32)
        occ = set(core.occupied)
        for t, ty in enumerate(core.site_types):
            if ty != EMPTY and t not in occ:
                bl[_NODE_OF_SITE[t], ty] += 1.0
        hit = _BLD[key] = (bl, bl.sum(axis=1))
    return hit


_BOARD_KEYS = (
    "seat_pickable", "seat_free_sites", "seat_vroom", "seat_vroom_now", "seat_targets",
    "seat_reach", "contested_node", "pair_overlap", "frontier_contested",
)  # fmt: skip
_ORDER_KEYS = ("res_before", "res_after", "res_pending", "place_now", "sel_state")


def _span(keys: tuple[str, ...]) -> tuple[int, int, dict[str, slice]]:
    lo = _OFF3[keys[0]]
    hi = _OFF3[keys[-1]] + int(np.prod(_SHAPES_V3[keys[-1]]))
    assert hi - lo == sum(int(np.prod(_SHAPES_V3[k])) for k in keys)  # contiguous
    sl = {k: slice(_OFF3[k] - lo, _OFF3[k] - lo + int(np.prod(_SHAPES_V3[k]))) for k in keys}
    return lo, hi, sl


_BOARD_LO, _BOARD_HI, _BOARD_SL = _span(_BOARD_KEYS)
_ORDER_LO, _ORDER_HI, _ORDER_SL = _span(_ORDER_KEYS)
_BOARD_CACHE: dict[tuple[Any, ...], np.ndarray[Any, Any]] = {}
_ORDER_CACHE: dict[tuple[Any, ...], np.ndarray[Any, Any]] = {}


def _board_blocks(
    core: Core, me: int, g: _Geom, sc: _Sites, wi: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """The v3 blocks that depend on lines, passengers, buildings, buses and the clock."""
    n = core.players
    seats = core.seats
    out = np.zeros(_BOARD_HI - _BOARD_LO, np.float32)
    V = {k: out[sl] for k, sl in _BOARD_SL.items()}
    g3 = _geometry3(core, me, g)
    nl = g3.node_line
    wf = wi.astype(np.float32)
    nw = wf / WAITING_SCALE
    V["seat_pickable"][:] = (nw[:, None] * nl).reshape(-1)
    V["seat_free_sites"][:] = (sc.free[:, None] * nl).reshape(-1)
    bl, bl_any = _buildings(core)
    w_on = wf @ nl
    e_on = sc.empty.astype(np.float32) @ nl
    t_on = bl.T @ nl  # (3, 5)
    buses = np.zeros(MAX_SEATS, np.float32)
    buses[:n] = [seats[(me + j) % n].buses for j in range(n)]
    sv = V["seat_vroom"].reshape(MAX_SEATS, 3)
    sv[:, 0] = w_on / C.TOTAL_PASSENGERS
    sv[:, 1] = e_on / 10.0
    sv[:, 2] = np.minimum(np.minimum(buses, w_on), e_on) / float(C.BUSES_PER_PLAYER)
    vn = np.minimum(np.minimum(buses, w_on), t_on[core.clock])
    V["seat_vroom_now"][:] = vn / float(C.BUSES_PER_PLAYER)
    V["seat_targets"][:] = (t_on.T / 10.0).reshape(-1)
    V["seat_reach"][:] = g3.reach
    cn = V["contested_node"].reshape(N, 3)
    cn[:, 0] = nw * g3.shared_mask
    cn[:, 1] = bl_any / _SITES_PER_NODE * g3.shared_mask
    cn[:, 2] = g3.opp_reach_near
    po = V["pair_overlap"].reshape(MAX_SEATS, 4)
    po[:, 0] = g3.shared_nodes / 10.0
    po[:, 1] = (wf @ g3.shared) / C.TOTAL_PASSENGERS
    po[:, 2] = (bl_any @ g3.shared) / 10.0
    po[:, 3] = g3.shared_streets / 10.0
    V["frontier_contested"][:] = g3.frontier_contested
    return out


def _order_blocks(core: Core, me: int) -> np.ndarray[Any, Any]:
    """The v3 blocks read from the action board: resolution order relative to the ego seat
    (R-029, R-030), markers still to resolve, what a marker placed now would get (R-025,
    R-030..R-033) and who is still selecting (R-024, R-027)."""
    n = core.players
    ab = core.action_board
    seats = core.seats
    phase = core.phase
    out = np.zeros(_ORDER_HI - _ORDER_LO, np.float32)
    rb, ra, rp = (out[_ORDER_SL[k]] for k in ("res_before", "res_after", "res_pending"))
    positions = resolution_positions(core)
    resolving = phase in _RESOLVING
    if resolving:
        act = resolving_action(core)
        assert act is not None
        start = positions.index((act, core.cursor))
    elif phase == PHASE_SEL:
        start = 0
    else:
        start = len(positions)
    first: dict[int, int] = {}
    for i, (a, k) in enumerate(positions):
        if ab[a][k] == me and a not in first:
            first[a] = i
    step = 1.0 / _LETTER_SCALE
    for i, (a, k) in enumerate(positions):
        idx = a * MAX_SEATS + (ab[a][k] - me) % n
        if i >= start:
            rp[idx] += step
        f = first.get(a)
        if f is not None:
            if i < f:
                rb[idx] += step
            elif i > f:
                ra[idx] += step
    if phase == PHASE_SEL or resolving:  # Starting Player resolves at round end (R-064)
        for owner in ab[ACT_STARTING]:
            rp[ACT_STARTING * MAX_SEATS + (owner - me) % n] += step
    pn = out[_ORDER_SL["place_now"]].reshape(NUM_ACTION_KINDS, 2)
    for a, row in enumerate(ab):
        if len(row) >= C.ACTION_SPACES[a]:
            continue  # full: cannot be chosen (R-025)
        if a in _MULTI:
            pn[a, 0] = max(marker_power(core, a, len(row)), 0) / _POWER_SCALE
        else:
            pn[a, 0] = 1.0
        pn[a, 1] = 1.0 if (a in _FIRST_IF_FREE or not row) else 0.0
    ss = out[_ORDER_SL["sel_state"]].reshape(MAX_SEATS, 2)
    for j in range(n):
        seat = (me + j) % n
        if phase == PHASE_SEL and seat not in core.dropped and seats[seat].action_markers > 0:
            ss[j, 0] = 1.0  # still selecting this round (R-024, R-027)
        ss[j, 1] = min(sum(row.count(seat) for row in ab) / _LETTER_SCALE, 1.0)
    return out


def encode_v3(core: Core) -> np.ndarray[Any, Any]:
    """Observation v3 (``FEATURES_V3``) for the seat to move in ``core`` (module docstring)."""
    B = _BUF3
    g, sc, wi = _fill_v2(core, B, _VIEW3)
    me = core.current
    seats = core.seats
    bkey = (
        me, core.waiting, core.site_types, core.occupied, core.clock,
        *[(s.line, s.buses) for s in seats],
    )  # fmt: skip
    blk = _BOARD_CACHE.get(bkey)
    if blk is None:
        if len(_BOARD_CACHE) >= _CACHE_CAP:
            _BOARD_CACHE.clear()
        blk = _BOARD_CACHE[bkey] = _board_blocks(core, me, g, sc, wi)
    B[_BOARD_LO:_BOARD_HI] = blk
    okey = (
        me, core.players, core.action_board, core.phase, core.cursor, core.dropped,
        *[(s.buses, s.action_markers > 0) for s in seats],
    )  # fmt: skip
    blk = _ORDER_CACHE.get(okey)
    if blk is None:
        if len(_ORDER_CACHE) >= _CACHE_CAP:
            _ORDER_CACHE.clear()
        blk = _ORDER_CACHE[okey] = _order_blocks(core, me)
    B[_ORDER_LO:_ORDER_HI] = blk
    # score now (R-068) against the opponents, in the R-072 / leader / rank shapes
    sc_now = [s.points - s.stones for s in seats]
    mine = sc_now[me]
    others = sc_now[:me] + sc_now[me + 1 :]
    sg = _OFF3["score_gap"]
    B[sg] = min(1.0, max(-1.0, (mine - max(others)) / 10.0))
    B[sg + 1] = min(1.0, max(-1.0, (mine - sum(others) / len(others)) / 10.0))
    beaten = sum(1.0 if mine > x else 0.5 if mine == x else 0.0 for x in others)
    B[sg + 2] = 2.0 * beaten / len(others) - 1.0
    return B.copy()


# ---- versions -------------------------------------------------------------------------------

FEATURES_V2 = 1  # observation v2 (this module's original layout); ppo-long, ppo-smoke, runs/az*
FEATURES_V3 = 2  # observation v3: v2 + per-seat derived and interaction blocks
FEATURE_VERSIONS = (FEATURES_V2, FEATURES_V3)
_ENCODERS: dict[int, Any] = {FEATURES_V2: encode, FEATURES_V3: encode_v3}
_LAYOUTS = {FEATURES_V2: _SHAPES, FEATURES_V3: _SHAPES_V3}
_SIZES = {FEATURES_V2: OBS_SIZE, FEATURES_V3: OBS_SIZE_V3}


def _check(version: int) -> int:
    if version not in _ENCODERS:
        raise ValueError(f"unknown features version {version!r} (known: {FEATURE_VERSIONS})")
    return version


def encoder(version: int) -> Any:
    """``encode`` for v2 (1) or ``encode_v3`` for v3 (2)."""
    return _ENCODERS[_check(version)]


def obs_size(version: int) -> int:
    return _SIZES[_check(version)]


def version_for_obs_size(size: int) -> int | None:
    """The features version whose vector has ``size`` entries, or None."""
    return next((v for v, sz in _SIZES.items() if sz == size), None)


def input_columns(src: int, dst: int) -> np.ndarray[Any, Any]:
    """For each input of version ``src``, its column in version ``dst`` (matched by block name);
    the warm start copies first-layer weights along this map and zeroes the other columns."""
    lay_s, lay_d = _LAYOUTS[_check(src)], _LAYOUTS[_check(dst)]
    off_d: dict[str, int] = {}
    o = 0
    for k, sh in lay_d.items():
        off_d[k] = o
        o += int(np.prod(sh))
    cols: list[np.ndarray[Any, Any]] = []
    for k, sh in lay_s.items():
        if k not in lay_d or lay_d[k] != sh:
            raise ValueError(f"block {k!r} of v{src} has no same-shape block in v{dst}")
        cols.append(off_d[k] + np.arange(int(np.prod(sh))))
    return np.concatenate(cols)
