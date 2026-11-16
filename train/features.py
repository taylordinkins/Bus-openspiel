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

from types import SimpleNamespace
from typing import Any

import numpy as np

from bus import constants as C
from bus.actions import NUM_ACTION_KINDS, N, S, T
from bus.lines import legal_extensions, line_nodes
from bus.observer import LETTERS, BusObserver
from bus.rules import EMPTY, Core, board

# Bump by hand when feature semantics change without a shape change (it is part of the layout
# hash, so older exported models are rejected as stale).
FEATURES_VERSION = 1
MAX_SEATS = C.MAX_PLAYERS
PLAYER_COUNTS = tuple(range(C.MIN_PLAYERS, C.MAX_PLAYERS + 1))
_NTYPES = len(C.BUILDING_TYPES)
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

_OBSERVERS: dict[int, BusObserver] = {}


def _observer(n: int) -> BusObserver:
    ob = _OBSERVERS.get(n)
    if ob is None:
        ob = _OBSERVERS[n] = BusObserver("win_loss", n)
    return ob


def obs_layout() -> dict[str, tuple[int, ...]]:
    """Block name -> shape, in vector order (documentation and tests)."""
    return dict(_SHAPES)


def _rotate(view: np.ndarray[Any, Any], axis: int, perm: list[int]) -> np.ndarray[Any, Any]:
    taken = np.take(view, perm, axis=axis)
    shape = list(taken.shape)
    shape[axis] = MAX_SEATS
    out = np.zeros(shape, np.float32)
    idx: list[Any] = [slice(None)] * len(shape)
    idx[axis] = slice(0, len(perm))
    out[tuple(idx)] = taken
    return out


def _tiebreak(core: Core, perm: list[int]) -> np.ndarray[Any, Any]:
    out = np.zeros(MAX_SEATS, np.float32)
    scored = sorted(
        (s.last_score_move, i) for i, s in enumerate(core.seats) if s.last_score_move is not None
    )
    m = len(scored)
    pos = {seat: r for r, (_, seat) in enumerate(scored)}
    for j, seat in enumerate(perm):
        if seat in pos:
            out[j] = (m - pos[seat]) / m
    return out


def encode(core: Core) -> np.ndarray[Any, Any]:
    """Observation v2 for the seat to move in ``core`` (see module docstring)."""
    n = core.players
    me = core.current
    perm = [(me + i) % n for i in range(n)]
    b = board()
    ob = _observer(n)
    ob.set_from(SimpleNamespace(core=core), me)
    d = ob.dict
    parts: dict[str, np.ndarray[Any, Any]] = {}
    for k in _PLAIN:
        parts[k] = d[k]
    for k, ax in _SEATED.items():
        parts[k] = _rotate(d[k], ax, perm)
    onehot = np.zeros(len(PLAYER_COUNTS), np.float32)
    onehot[n - C.MIN_PLAYERS] = 1.0
    parts["n_onehot"] = onehot
    present = np.zeros(MAX_SEATS, np.float32)
    present[:n] = 1.0
    parts["present"] = present
    score = np.zeros(MAX_SEATS, np.float32)
    for j, seat in enumerate(perm):
        s = core.seats[seat]
        score[j] = min(1.0, max(-1.0, (s.points - s.stones) / 10.0))
    parts["score"] = score
    parts["tiebreak"] = _tiebreak(core, perm)

    lines = [s.line for s in core.seats]
    mine: set[int] = set(line_nodes(b, lines[me])) if lines[me] else set()
    pickable = np.zeros(N, np.float32)
    free = np.zeros(N, np.float32)
    opp_at = np.zeros(N, np.float32)
    waiting_on = 0
    free_on = 0
    for node in mine:
        pickable[node] = core.waiting[node] / C.TOTAL_PASSENGERS
        waiting_on += core.waiting[node]
        e = sum(1 for t in b.node_sites[node] if core.site_types[t] == EMPTY)
        free[node] = e / _SITES_PER_NODE
        free_on += e
    contested = np.zeros(S, np.float32)
    frontier = np.zeros((S, MAX_SEATS), np.float32)
    for j, seat in enumerate(perm):
        line = lines[seat]
        if not line:
            continue
        if seat != me:
            contested[list(line)] = 1.0
            for node in line_nodes(b, line):
                opp_at[node] += 1.0 / (MAX_SEATS - 1)
        for _end, street in legal_extensions(b, lines, seat):
            frontier[street, j] = 1.0
    parts["ego_pickable"] = pickable
    parts["ego_free_sites"] = free
    parts["opp_lines_at_node"] = opp_at
    parts["street_contested"] = contested
    parts["frontier"] = frontier
    deliveries = min(core.seats[me].buses, waiting_on, free_on)
    parts["vroom"] = np.array(
        [waiting_on / C.TOTAL_PASSENGERS, free_on / 10.0, deliveries / float(C.BUSES_PER_PLAYER)],
        np.float32,
    )
    out = np.concatenate([parts[k].reshape(-1) for k in _SHAPES]).astype(np.float32, copy=False)
    return out
