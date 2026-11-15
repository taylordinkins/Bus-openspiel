"""Flat-UCT MCTS seat (Python `open_spiel.python.algorithms.mcts.MCTSBot`, never the C++ bot).

Seat type `mcts[:SIMS[-ROLLOUT]]`: SIMS is the simulation cap (default 100, max 20000) and ROLLOUT
is `random` (default) or `heuristic` (rollouts played by HeuristicBot). The `-` joins the two
fields because the registry allows a single `:`.

A fresh MCTSBot is built per `choose` call from the live state's game, seeded from the caller's
rng, so the same state, rng and simulation cap always give the same action. The search uses
`state.returns()` at terminals (win_loss or margin, whichever the game was built with) and is
sampled with `solve=False`. Rollouts run to the end of the game, uncapped.

Think time: `time_limit_s` (None = unlimited) stops the search once the deadline passes (checked
once per simulation, so one rollout can overrun it). The server sets it through
`MCTSSeatBot.with_time_limit(seconds)`, which returns a copy and leaves the cached registry
instance untouched. At least one simulation always runs.

The tree sees the true state, including hidden information (Phase 6 adds IS-MCTS).
"""

# pyright: basic
from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from bots.heuristic import HeuristicBot

DEFAULT_SIMS = 100
MAX_SIMS = 20000
UCT_C = 2.0
ROLLOUTS = ("random", "heuristic")


class HeuristicRolloutEvaluator(mcts.RandomRolloutEvaluator):
    """Rollouts played by HeuristicBot (random tie-breaks) until the game ends."""

    def __init__(self, random_state: np.random.RandomState) -> None:
        super().__init__(n_rollouts=1, random_state=random_state)
        self._bot = HeuristicBot()
        self._rng = random.Random(int(random_state.randint(2**31)))

    def evaluate(self, state):  # type: ignore[override]
        s = state.clone()
        while not s.is_terminal():
            if s.is_chance_node():
                actions, probs = zip(*s.chance_outcomes(), strict=True)
                s.apply_action(int(self._random_state.choice(actions, p=probs)))
            else:
                s.apply_action(self._bot.choose(s, self._rng))
        return np.array(s.returns())


class _TimedMCTSBot(mcts.MCTSBot):
    deadline: float | None = None

    def mcts_search(self, state):  # type: ignore[override]
        # Same loop as the library's, plus a deadline check before each simulation after the first.
        root = mcts.SearchNode(None, state.current_player(), 1)
        for i in range(self.max_simulations):
            if i > 0 and self.deadline is not None and time.monotonic() >= self.deadline:
                break
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
            else:
                returns = self.evaluator.evaluate(working_state)
            while visit_path:
                idx = -1
                while visit_path[idx].player == pyspiel.PlayerId.CHANCE:
                    idx -= 1
                target = returns[visit_path[idx].player]
                node = visit_path.pop()
                node.total_reward += target
                node.explore_count += 1
        return root


@dataclass(frozen=True)
class MCTSSeatBot:
    sims: int = DEFAULT_SIMS
    rollout: str = "random"
    time_limit_s: float | None = None
    uct_c: float = UCT_C

    def __post_init__(self) -> None:
        if not 1 <= self.sims <= MAX_SIMS:
            raise ValueError(f"mcts simulations must be in 1..{MAX_SIMS}")
        if self.rollout not in ROLLOUTS:
            raise ValueError(f"mcts rollout must be one of {ROLLOUTS}")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")

    def with_time_limit(self, seconds: float | None) -> MCTSSeatBot:
        return replace(self, time_limit_s=seconds)

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        rs = np.random.RandomState(rng.randrange(2**31))
        evaluator = (
            HeuristicRolloutEvaluator(rs)
            if self.rollout == "heuristic"
            else mcts.RandomRolloutEvaluator(1, random_state=rs)
        )
        bot = _TimedMCTSBot(
            state.get_game(), self.uct_c, self.sims, evaluator, solve=False, random_state=rs
        )
        if self.time_limit_s is not None:
            bot.deadline = time.monotonic() + self.time_limit_s
        action = int(bot.step(state))
        return action if action in legal else rng.choice(legal)


def parse_mcts_param(param: str | None) -> MCTSSeatBot:
    if param is None:
        return MCTSSeatBot()
    sims_s, dash, rollout = param.partition("-")
    if dash and not rollout:
        raise ValueError("empty mcts rollout kind")
    if not sims_s.isdigit():
        raise ValueError(f"bad mcts simulations {sims_s!r}")
    return MCTSSeatBot(sims=int(sims_s), rollout=rollout or "random")
