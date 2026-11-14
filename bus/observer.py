"""Observer: string and RL tensor (PLAN 5.2, docs/OPENSPIEL_NOTES.md observer pattern).

Bus has perfect information (R-073), so the observation is ABSOLUTE and identical for every
observing player; the seat to act is part of it (one-hot). The tensor is one flat float32 array
of constant size per player count ``n`` with named views in ``dict`` (written through, never
rebound). Counts are scaled to [0, 1] by a stated maximum; ``last_score_move`` and move numbers
are deliberately absent from the tensor (the string shows them).

Views (shape; scaling):
  seat_to_act (n) one-hot, all zero at END; start_player (n) one-hot; phase (11) one-hot;
  round (1) /20; setup_placed (1) / 2n; sel_pass (1) /20; dropped (n) 0/1; end_types (3) 0/1;
  street_occ (S, n) 0/1 line occupancy; street_end (S, n, 2) 0/1: the street holding end e of the
  seat's line (a one-street line marks both ends); node_waiting (N) /15; node_line (N, n) 0/1
  line membership; site_type (T, 3) one-hot (all zero = empty); site_occupied (T) 0/1;
  supply (1) /15; clock (3) one-hot; stones_on_clock (1) /5; seat_stones (n) /5;
  seat_buses (n) /5; seat_markers_left (n) /20; seat_line_markers_left (n) /25;
  seat_points (n) /10 (capped at 1; real games rarely pass 10); action_board (7, 6, n) 0/1 (action x letter x seat);
  units_left (1) /6; moves_left (1) /5; cursor (6) one-hot letter while resolving;
  picked_node (N) one-hot in VD.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from bus import constants as C
from bus.actions import NUM_ACTION_KINDS, N, S, T
from bus.board import load_board
from bus.lines import line_nodes
from bus.rules import (
    EMPTY,
    END_TYPES,
    PHASE_BD,
    PHASE_CK,
    PHASE_END,
    PHASE_LN,
    PHASE_PX,
    PHASE_VD,
    PHASE_VP,
    PHASES,
    Core,
    final_scores,
    winners,
)

LETTERS = max(C.ACTION_SPACES)
ROUND_SCALE = float(C.ACTION_MARKERS_PER_PLAYER)
PASS_SCALE = float(C.ACTION_MARKERS_PER_PLAYER)
WAITING_SCALE = float(C.TOTAL_PASSENGERS)
STONE_SCALE = 5.0  # most stones (R-006)
BUS_SCALE = float(C.BUSES_PER_PLAYER)
MARKER_SCALE = float(C.ACTION_MARKERS_PER_PLAYER)
LINE_MARKER_SCALE = float(C.LINE_MARKERS_PER_PLAYER)
POINT_SCALE = 10.0  # user: points rarely pass 10 in practice; capped at 1
UNITS_SCALE = 6.0  # most Lines units (ACTIONS 6)
MOVES_SCALE = float(C.BUSES_PER_PLAYER)
_NTYPES = len(C.BUILDING_TYPES)
_RESOLVING = (PHASE_LN, PHASE_PX, PHASE_BD, PHASE_CK, PHASE_VP, PHASE_VD)
_BOARD = load_board()


@cache
def _nodes_of(line: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(line_nodes(_BOARD, line)) if line else ()


def observation_string(core: Core, returns_mode: str) -> str:
    """Public state text (perfect information: identical for every seat)."""
    seats = ";".join(
        f"P{i}:pts={s.points},bus={s.buses},am={s.action_markers},lm={s.line_markers},"
        f"st={s.stones},bld={s.buildings},line={list(s.line)},last={s.last_score_move}"
        for i, s in enumerate(core.seats)
    )
    sites = {t: C.BUILDING_TYPES[y] for t, y in enumerate(core.site_types) if y != EMPTY}
    waiting = {n: w for n, w in enumerate(core.waiting) if w}
    text = (
        f"bus n={core.players} returns={returns_mode} round={core.round} phase={core.phase} "
        f"cur={core.current} clock={C.BUILDING_TYPES[core.clock]} "
        f"stones_on_clock={core.stones_on_clock} supply={core.supply} placed={core.setup_placed} "
        f"start={core.start_player} move={core.move_number} pass={core.sel_pass} "
        f"dropped={list(core.dropped)} cursor={core.cursor} units={core.units_left} "
        f"moves_left={core.moves_left} picked={core.picked_node} "
        f"occupied={list(core.occupied)} "
        f"board={[list(x) for x in core.action_board]} sites={sites} waiting={waiting} | {seats}"
    )
    if core.phase == PHASE_END:
        win = " ".join(f"P{p}" for p in winners(core))
        text += (
            f" | end: {','.join(core.end_types)} winners: {win} scores: {list(final_scores(core))}"
        )
    return text


class BusObserver:
    """PyObserver: flat float32 tensor with named views, plus the string."""

    def __init__(self, returns_mode: str, players: int) -> None:
        n = players
        self.returns_mode = returns_mode
        self.players = n
        shapes: dict[str, tuple[int, ...]] = {
            "seat_to_act": (n,),
            "start_player": (n,),
            "phase": (len(PHASES),),
            "round": (1,),
            "setup_placed": (1,),
            "sel_pass": (1,),
            "dropped": (n,),
            "end_types": (len(END_TYPES),),
            "street_occ": (S, n),
            "street_end": (S, n, 2),
            "node_waiting": (N,),
            "node_line": (N, n),
            "site_type": (T, _NTYPES),
            "site_occupied": (T,),
            "supply": (1,),
            "clock": (_NTYPES,),
            "stones_on_clock": (1,),
            "seat_stones": (n,),
            "seat_buses": (n,),
            "seat_markers_left": (n,),
            "seat_line_markers_left": (n,),
            "seat_points": (n,),
            "action_board": (NUM_ACTION_KINDS, LETTERS, n),
            "units_left": (1,),
            "moves_left": (1,),
            "cursor": (LETTERS,),
            "picked_node": (N,),
        }
        total = sum(int(np.prod(sh)) for sh in shapes.values())
        self.tensor = np.zeros(total, np.float32)
        self.dict: dict[str, Any] = {}
        off = 0
        for name, sh in shapes.items():
            size = int(np.prod(sh))
            self.dict[name] = self.tensor[off : off + size].reshape(sh)
            off += size
        self._phase_index = {ph: i for i, ph in enumerate(PHASES)}

    def set_from(self, state: Any, player: int) -> None:
        core: Core = state.core
        d = self.dict
        self.tensor.fill(0)
        n = core.players
        terminal = core.phase == PHASE_END
        if not terminal:
            d["seat_to_act"][core.current] = 1.0
        d["start_player"][core.start_player] = 1.0
        d["phase"][self._phase_index[core.phase]] = 1.0
        d["round"][0] = min(core.round / ROUND_SCALE, 1.0)
        d["setup_placed"][0] = core.setup_placed / (2 * n)
        d["sel_pass"][0] = min(core.sel_pass / PASS_SCALE, 1.0)
        for p in core.dropped:
            d["dropped"][p] = 1.0
        for et in core.end_types:
            d["end_types"][END_TYPES.index(et)] = 1.0
        for p, seat in enumerate(core.seats):
            if seat.line:
                d["street_occ"][list(seat.line), p] = 1.0
                d["street_end"][seat.line[0], p, 0] = 1.0
                d["street_end"][seat.line[-1], p, 1] = 1.0
                d["node_line"][list(_nodes_of(seat.line)), p] = 1.0
            d["seat_stones"][p] = seat.stones / STONE_SCALE
            d["seat_buses"][p] = seat.buses / BUS_SCALE
            d["seat_markers_left"][p] = seat.action_markers / MARKER_SCALE
            d["seat_line_markers_left"][p] = seat.line_markers / LINE_MARKER_SCALE
            d["seat_points"][p] = min(seat.points / POINT_SCALE, 1.0)
        d["node_waiting"][:] = core.waiting
        d["node_waiting"] /= WAITING_SCALE
        sites = core.site_types
        for t, y in enumerate(sites):
            if y != EMPTY:
                d["site_type"][t, y] = 1.0
        if core.occupied:
            d["site_occupied"][list(core.occupied)] = 1.0
        d["supply"][0] = core.supply / WAITING_SCALE
        d["clock"][core.clock] = 1.0
        d["stones_on_clock"][0] = core.stones_on_clock / STONE_SCALE
        for a, row in enumerate(core.action_board):
            for k, seat_id in enumerate(row):
                d["action_board"][a, k, seat_id] = 1.0
        d["units_left"][0] = core.units_left / UNITS_SCALE
        d["moves_left"][0] = core.moves_left / MOVES_SCALE
        if core.phase in _RESOLVING:
            d["cursor"][core.cursor] = 1.0
        if core.picked_node >= 0:
            d["picked_node"][core.picked_node] = 1.0

    def string_from(self, state: Any, player: int) -> str:
        return observation_string(state.core, self.returns_mode)
