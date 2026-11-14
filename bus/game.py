"""OpenSpiel adapter for Bus (PLAN 5.2). Logic lives in ``bus.rules``."""

from __future__ import annotations

from typing import Any

import pyspiel

from bus import constants as C
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS, action_to_string, decode_action
from bus.observer import BusObserver, observation_string
from bus.rules import Core, Seat

MAX_GAME_LENGTH_PER_SEAT = 234  # ACTIONS.md section 6


def _game_type(returns: str) -> pyspiel.GameType:
    return pyspiel.GameType(
        short_name="python_bus",
        long_name="Bus (Splotter)",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=(
            pyspiel.GameType.Utility.ZERO_SUM
            if returns == "win_loss"
            else pyspiel.GameType.Utility.GENERAL_SUM
        ),
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        min_num_players=C.MIN_PLAYERS,
        max_num_players=C.MAX_PLAYERS,
        provides_information_state_string=False,
        provides_information_state_tensor=False,
        provides_observation_string=True,
        provides_observation_tensor=True,
        parameter_specification={"players": C.DEFAULT_PLAYERS, "returns": C.DEFAULT_RETURNS},
    )


GAME_TYPE = _game_type(C.DEFAULT_RETURNS)


class BusGame(pyspiel.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = dict(params or {})
        n = int(p.get("players", C.DEFAULT_PLAYERS))
        returns = p.get("returns", C.DEFAULT_RETURNS)
        if not C.MIN_PLAYERS <= n <= C.MAX_PLAYERS:
            raise ValueError(f"players must be {C.MIN_PLAYERS}..{C.MAX_PLAYERS}, got {n}")
        if returns not in C.RETURNS_MODES:
            raise ValueError(f"returns must be one of {C.RETURNS_MODES}, got {returns!r}")
        self.players = n
        self.returns_mode: str = returns
        low = -1.0 / (n - 1) if returns == "win_loss" else -1.0
        info = pyspiel.GameInfo(
            num_distinct_actions=NUM_DISTINCT_ACTIONS,
            max_chance_outcomes=0,
            num_players=n,
            min_utility=low,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=MAX_GAME_LENGTH_PER_SEAT * n,
        )
        super().__init__(_game_type(returns), info, p)

    def new_initial_state(self, *args: Any) -> BusState:
        return BusState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> BusObserver:
        return BusObserver(self.returns_mode)

    def max_chance_nodes_in_history(self) -> int:
        return 0


class BusState(pyspiel.State):
    def __init__(self, game: BusGame) -> None:
        super().__init__(game)
        self.core: Core = rules.initial_core(game.players)
        self._returns_mode: str = game.returns_mode

    # --- accessors used by tests and tools ---
    @property
    def round(self) -> int:
        return self.core.round

    @property
    def phase(self) -> str:
        return self.core.phase

    @property
    def clock(self) -> str:
        return C.BUILDING_TYPES[self.core.clock]

    @property
    def stones_on_clock(self) -> int:
        return self.core.stones_on_clock

    @property
    def passenger_supply(self) -> int:
        return self.core.supply

    @property
    def action_board(self) -> tuple[tuple[int, ...], ...]:
        return self.core.action_board

    def waiting_at(self, node: int) -> int:
        return self.core.waiting[node]

    def site_building(self, t: int) -> str | None:
        y = self.core.site_types[t]
        return None if y == rules.EMPTY else C.BUILDING_TYPES[y]

    def seat(self, p: int) -> Seat:
        return self.core.seats[p]

    # --- pyspiel API ---
    def current_player(self) -> int:
        return pyspiel.PlayerId.TERMINAL if self.is_terminal() else self.core.current

    def is_terminal(self) -> bool:
        return False  # game end arrives with later units

    def _legal_actions(self, player: int) -> list[int]:
        return rules.legal_actions(self.core)

    def _apply_action(self, action: int) -> None:
        self.core = rules.apply(self.core, action)

    def _action_to_string(self, player: int, action: int) -> str:
        kind, _ = decode_action(action)
        if kind == "extend":
            # Hook: state-dependent EXTEND wording arrives with unit 5 (ACTIONS section 4).
            return action_to_string(player, action)
        return action_to_string(player, action)

    def returns(self) -> list[float]:
        return [0.0] * self.core.players

    def __str__(self) -> str:
        return observation_string(self.core, self._returns_mode)


pyspiel.register_game(GAME_TYPE, BusGame)
