"""OpenSpiel adapter for Bus (PLAN 5.2). Logic lives in ``bus.rules``."""

from __future__ import annotations

from typing import Any

import pyspiel

from bus import constants as C
from bus import rules
from bus.actions import (
    ACTION_NAMES,
    NUM_DISTINCT_ACTIONS,
    action_to_string,
    decode_action,
    street_ref,
)
from bus.lines import line_ends
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
            # ZERO_SUM only for win_loss (R-071); margin is GENERAL_SUM and declares no sum (R-072)
            utility_sum=0.0 if returns == "win_loss" else None,
            max_game_length=MAX_GAME_LENGTH_PER_SEAT * n,
        )
        super().__init__(_game_type(returns), info, p)

    def new_initial_state(self, *args: Any) -> BusState:
        return BusState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> BusObserver:
        return BusObserver(self.returns_mode, self.players)

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

    @property
    def start_player(self) -> int:
        return self.core.start_player

    @property
    def sites_full(self) -> bool:
        return rules.sites_full(self.core)

    @property
    def end_types(self) -> tuple[str, ...]:
        return self.core.end_types

    @property
    def final_scores(self) -> tuple[int, ...]:
        return rules.final_scores(self.core)

    @property
    def winners(self) -> tuple[int, ...]:
        return rules.winners(self.core)

    @property
    def bus_max(self) -> int:
        return rules.bus_max(self.core)

    @property
    def cursor_letter(self) -> int:
        return self.core.cursor

    @property
    def units_left(self) -> int:
        return self.core.units_left

    @property
    def moves_left(self) -> int:
        return self.core.moves_left

    @property
    def returns_mode(self) -> str:
        return self._returns_mode

    def is_occupied(self, t: int) -> bool:
        return t in self.core.occupied

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
        return self.core.phase == rules.PHASE_END

    def _legal_actions(self, player: int) -> list[int]:
        return rules.legal_actions(self.core)

    def _apply_action(self, action: int) -> None:
        self.core = rules.apply(self.core, action)

    def _action_to_string(self, player: int, action: int) -> str:
        kind, params = decode_action(action)
        if kind == "extend":
            e, street = params
            b = rules.board()
            if self.core.phase == rules.PHASE_L1:
                return f"P{player}: start line on {street_ref(street)}"
            line = self.core.seats[player].line
            st = b.streets[street]
            if line and line_ends(b, line)[e] in (st.a, st.b):
                new_end = st.other(line_ends(b, line)[e])
                return f"P{player}: extend end {e} via {street_ref(street)} to n{new_end}"
        return action_to_string(player, action)

    def returns(self) -> list[float]:
        return rules.returns_for(self.core, self._returns_mode)

    def __str__(self) -> str:
        return observation_string(self.core, self._returns_mode)


def action_meta(state: BusState, action_id: int) -> dict[str, Any]:
    """JSON-serialisable description of an action for click-to-act (ACTIONS 5). Pure.

    Any id 0..375 gets a dict; target fields not applicable in this state are None.
    """
    core = state.core
    b = rules.board()
    kind, p = decode_action(action_id)
    player = core.current
    m: dict[str, Any] = {
        "id": action_id,
        "kind": kind,
        "player": player,
        "label": state._action_to_string(player, action_id),  # pyright: ignore[reportPrivateUsage]
    }
    if kind == "select":
        a = p[0]
        spaces = len(core.action_board[a])
        m.update(
            action=a,
            action_name=ACTION_NAMES[a],
            space=spaces if spaces < C.ACTION_SPACES[a] else None,
        )
    elif kind == "extend":
        e, street = p
        first = core.phase == rules.PHASE_L1
        line = core.seats[player].line
        frm: int | None = None
        to: int | None = None
        if line and not first:
            frm = line_ends(b, line)[e]
            st = b.streets[street]
            to = st.other(frm) if frm in (st.a, st.b) else None
        shared = not first and any(
            street in s.line for i, s in enumerate(core.seats) if i != player
        )
        setup = {rules.PHASE_L1: "first", rules.PHASE_L2: "second"}.get(core.phase)
        m.update(
            end=0 if first else e,
            street=street,
            from_node=frm,
            to_node=to,
            setup=setup,
            shared_end=shared,
        )
    elif kind == "build":
        t, y = p
        site = b.sites[t]
        m.update(
            site=t,
            node=site.node,
            tier=site.tier,
            building_type=C.BUILDING_TYPES[y],
            setup=core.phase == rules.PHASE_SB,
        )
    elif kind == "passenger_to":
        m.update(station=p[0], node=b.stations[p[0]])
    elif kind == "clock_advance":
        m["from"] = C.BUILDING_TYPES[core.clock]
        m["to"] = C.BUILDING_TYPES[(core.clock + 1) % len(C.BUILDING_TYPES)]
    elif kind == "clock_pause":
        m.update(
            stones_left_after=core.stones_on_clock - 1, ends_game=core.stones_on_clock - 1 <= 0
        )
    elif kind == "pick_passenger":
        m.update(node=p[0], waiting=core.waiting[p[0]])
    elif kind == "deliver_to":
        site = b.sites[p[0]]
        picked = core.picked_node if core.picked_node >= 0 else None
        m.update(
            site=p[0],
            node=site.node,
            tier=site.tier,
            building_type=C.BUILDING_TYPES[core.clock],
            picked_node=picked,
            same_node=picked == site.node,
        )
    return m


def game_result(state: BusState) -> dict[str, Any] | None:
    """End types, winners, final scores and returns at END; None before (ACTIONS 3.12)."""
    core = state.core
    if core.phase != rules.PHASE_END:
        return None
    return {
        "end_types": list(core.end_types),
        "winners": list(rules.winners(core)),
        "final_scores": list(rules.final_scores(core)),
        "returns": state.returns(),
    }


pyspiel.register_game(GAME_TYPE, BusGame)
