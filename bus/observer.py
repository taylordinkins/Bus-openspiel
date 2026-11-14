"""Observation string and (stub) tensor. Full tensor is Phase 1 unit 13."""

from __future__ import annotations

from typing import Any

import numpy as np

from bus import constants as C
from bus.rules import EMPTY, Core

PHASE_CODES = ("SB", "L1", "L2")
TENSOR_SIZE = 8  # stub: replaced in unit 13


def observation_string(core: Core, returns_mode: str) -> str:
    """Public state text (perfect information: identical for every seat)."""
    seats = ";".join(
        f"P{i}:pts={s.points},bus={s.buses},am={s.action_markers},lm={s.line_markers},"
        f"st={s.stones},bld={s.buildings},line={list(s.line)}"
        for i, s in enumerate(core.seats)
    )
    sites = {t: C.BUILDING_TYPES[y] for t, y in enumerate(core.site_types) if y != EMPTY}
    waiting = {n: w for n, w in enumerate(core.waiting) if w}
    return (
        f"bus n={core.players} returns={returns_mode} round={core.round} phase={core.phase} "
        f"cur={core.current} clock={C.BUILDING_TYPES[core.clock]} "
        f"stones_on_clock={core.stones_on_clock} supply={core.supply} placed={core.setup_placed} "
        f"board={[list(x) for x in core.action_board]} sites={sites} waiting={waiting} | {seats}"
    )


class BusObserver:
    """PyObserver: flat float32 tensor plus a string."""

    def __init__(self, returns_mode: str) -> None:
        self.returns_mode = returns_mode
        self.tensor = np.zeros(TENSOR_SIZE, np.float32)
        self.dict: dict[str, Any] = {"summary": self.tensor}

    def set_from(self, state: Any, player: int) -> None:
        core: Core = state.core
        self.tensor.fill(0)
        self.tensor[0] = core.round
        phase = core.phase
        self.tensor[1] = PHASE_CODES.index(phase) if phase in PHASE_CODES else len(PHASE_CODES)
        self.tensor[2] = core.current
        self.tensor[3] = core.clock
        self.tensor[4] = core.stones_on_clock
        self.tensor[5] = core.supply
        self.tensor[6] = core.seats[player].points
        self.tensor[7] = core.setup_placed

    def string_from(self, state: Any, player: int) -> str:
        return observation_string(state.core, self.returns_mode)
