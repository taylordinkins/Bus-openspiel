"""Single-game environment for PPO self-play on python_bus.

The agent only sees decisions with more than one legal action: forced moves (exactly one legal
action, ``bus.rules.forced_action``) are applied automatically and counted. Bus has no chance
nodes (deterministic, R-073), so there is nothing to sample; ``seed`` seeds ``self.rng`` for
convenience (random policies, tie breaks) and the game itself is fully deterministic.
"""

from __future__ import annotations

import random
from typing import Any, cast

import numpy as np
import pyspiel

from bus import constants as C
from bus.actions import NUM_DISTINCT_ACTIONS
from bus.game import BusState
from bus.rules import PHASE_END, final_scores, returns_for
from train.features import OBS_SIZE, encode

DEFAULT_MIX = 0.7


def shaped_return(state: BusState, seat: int, mix: float = DEFAULT_MIX) -> float:
    """``mix * win_loss + (1 - mix) * margin`` for ``seat`` (D7); 0 before the game ends.

    Independent of the game's ``returns`` parameter. Ties share the win exactly as R-071.
    """
    wl = returns_for(state.core, "win_loss")[seat]
    if mix >= 1.0:
        return wl
    return mix * wl + (1.0 - mix) * returns_for(state.core, "margin")[seat]


def potentials(core: Any) -> np.ndarray[Any, Any]:
    """Per-seat potential (own score - mean opponent score) / 10, score = points - stones.

    Same formula as the unclipped R-072 margin (``bus.rules.final_scores``), valid mid-game.
    Training signal only (unit 8e dense reward).
    """
    sc = np.asarray(final_scores(core), np.float64)
    n = len(sc)
    return ((sc - (sc.sum() - sc) / (n - 1)) / 10.0).astype(np.float32)


class BusEnv:
    def __init__(
        self, num_players: int = C.DEFAULT_PLAYERS, returns: str = "win_loss", seed: int = 0
    ) -> None:
        if not C.MIN_PLAYERS <= num_players <= C.MAX_PLAYERS:
            raise ValueError(f"num_players must be {C.MIN_PLAYERS}..{C.MAX_PLAYERS}")
        self.num_players = num_players
        self.game = pyspiel.load_game("python_bus", {"players": num_players, "returns": returns})
        self.num_actions = NUM_DISTINCT_ACTIONS
        self.obs_size = OBS_SIZE
        self.rng = random.Random(seed)
        self.state: BusState = cast(BusState, self.game.new_initial_state())
        self.forced_steps = 0  # forced moves skipped in the current game
        self._legal: list[int] = []

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = cast(BusState, self.game.new_initial_state())
        self.forced_steps = 0
        self._skip_forced()

    def _skip_forced(self) -> None:
        st = self.state
        while not st.is_terminal():
            legal = st.legal_actions()
            if len(legal) != 1:
                self._legal = legal
                return
            st.apply_action(legal[0])
            self.forced_steps += 1
        self._legal = []

    @property
    def done(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_seat(self) -> int:
        return self.state.core.current

    def legal_actions(self) -> list[int]:
        return self._legal

    def obs(self) -> np.ndarray[Any, Any]:
        return encode(self.state.core)

    def legal_mask(self) -> np.ndarray[Any, Any]:
        mask = np.zeros(self.num_actions, np.bool_)
        mask[self._legal] = True
        return mask

    def step(self, action: int) -> None:
        if action not in self._legal:
            raise ValueError(f"action {action} is not legal")
        self.state.apply_action(action)
        self._skip_forced()

    def returns(self) -> list[float]:
        return list(self.state.returns())

    def shaped_returns(self, mix: float = DEFAULT_MIX) -> list[float]:
        return [shaped_return(self.state, p, mix) for p in range(self.num_players)]

    @property
    def phase(self) -> str:
        return self.state.core.phase

    def at_end(self) -> bool:
        return self.state.core.phase == PHASE_END
