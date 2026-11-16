"""Net-guided MCTS seat: `mcts_net:<SIMS>-<model>` (Phase 4 unit 7).

OpenSpiel's Python `MCTSBot` (never the C++ bot) with PUCT selection and a `NetEvaluator`: the net's
masked policy is the prior and its value head replaces rollouts (one numpy forward per expansion).
The `-` joins the fields because the registry allows a single `:`; SIMS is digits (1..20000) and
the model name is everything after the first `-`, resolved only through `models/manifest.yaml` (see
`bots.net`; names are never paths). Examples: `mcts_net:50-ppo-smoke`.

Values: the net's 5-wide value head is in ego order (index 0 = player to move) and trained on
shaped returns in [-1, 1]. `NetEvaluator.evaluate` maps it back to absolute seats,
`value_abs[(cur + k) % n] = v_ego[k]`, and every player maximises its own entry (max^n). Terminal
nodes use `state.returns()`, i.e. the game's own returns (win_loss by default, R-071: winner +1,
losers -1/(n-1)); that is slightly narrower than the net's [-1, 1], accepted.

Search: Dirichlet noise off, `solve=False`, exploration constant `uct_c` (default 1.5; PUCT adds
`c * prior * sqrt(N) / (n + 1)` to a Q in [-1, 1]). Forced (single-legal-action) root decisions
skip the search; forced nodes inside the tree still cost a simulation. Think time and the
simulation cap behave like `bots.mcts`: `with_time_limit(s)` returns a copy, at least one
simulation runs, the deadline is checked before each later one. Any exception or illegal result
falls back to the plain greedy `NetBot` (D24 spirit), logged.
"""

# pyright: basic
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from bots.mcts import MAX_SIMS, _TimedMCTSBot
from bots.net import NetBot, load_weights, resolve_model

log = logging.getLogger(__name__)

DEFAULT_SIMS = 50
UCT_C = 1.5
_CACHE_CAP = 20000


def _default_c() -> float:
    # BUS_MCTS_NET_C overrides the default exploration constant; a tuning knob for tournaments only.
    return float(os.environ.get("BUS_MCTS_NET_C", UCT_C))


class NetEvaluator(mcts.Evaluator):
    """Prior and value from one cached net forward per state."""

    def __init__(self, name: str) -> None:
        self.weights = load_weights(name)
        self._cache: dict = {}

    def _forward(self, state):
        from train.features import encode

        core = state.core
        hit = self._cache.get(core)
        if hit is not None:
            return hit
        legal = state.legal_actions()
        mask = np.zeros(self.weights.n_actions, np.bool_)
        mask[legal] = True
        probs, v_ego = self.weights.policy(encode(core), mask)
        n = state.num_players()
        cur = state.current_player()
        value = np.zeros(n)
        for k in range(n):
            value[(cur + k) % n] = float(v_ego[k])
        out = ([(a, float(probs[a])) for a in legal], value)
        if len(self._cache) >= _CACHE_CAP:
            self._cache.clear()
        self._cache[core] = out
        return out

    def prior(self, state):
        return list(self._forward(state)[0])

    def evaluate(self, state):
        if state.is_terminal():
            return np.array(state.returns())
        return self._forward(state)[1].copy()


@dataclass(frozen=True)
class NetMCTSBot:
    name: str
    sims: int = DEFAULT_SIMS
    time_limit_s: float | None = None
    uct_c: float = field(default_factory=_default_c)

    def __post_init__(self) -> None:
        if not 1 <= self.sims <= MAX_SIMS:
            raise ValueError(f"mcts_net simulations must be in 1..{MAX_SIMS}")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        resolve_model(self.name)
        load_weights(self.name)  # raises at seat creation if stale

    def with_time_limit(self, seconds: float | None) -> NetMCTSBot:
        return replace(self, time_limit_s=seconds)

    def _search(self, state, rng: random.Random) -> int:
        rs = np.random.RandomState(rng.randrange(2**31))
        bot = _TimedMCTSBot(
            state.get_game(),
            self.uct_c,
            self.sims,
            NetEvaluator(self.name),
            solve=False,
            random_state=rs,
            child_selection_fn=mcts.SearchNode.puct_value,
            dirichlet_noise=None,
        )
        if self.time_limit_s is not None:
            bot.deadline = time.monotonic() + self.time_limit_s
        return int(bot.step(state))

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = list(state.legal_actions())
        if len(legal) == 1:
            return legal[0]
        try:
            a = self._search(state, rng)
            if a in legal:
                return a
            log.warning("mcts_net %r returned illegal action %s; using net", self.name, a)
        except Exception:
            log.exception("mcts_net %r failed; using net", self.name)
        return NetBot(self.name).choose(state, rng)


def split_param(param: str) -> tuple[int, str]:
    """`<sims>-<model>` -> (sims, model); ValueError if malformed."""
    sims_s, dash, name = param.partition("-")
    if not sims_s.isdigit() or not dash or not name:
        raise ValueError("mcts_net takes <sims>-<model>, e.g. mcts_net:50-ppo-smoke")
    return int(sims_s), name


def parse_mcts_net_param(param: str | None) -> NetMCTSBot:
    if param is None:
        raise ValueError("mcts_net needs <sims>-<model>, e.g. mcts_net:50-ppo-smoke")
    sims, name = split_param(param)
    return NetMCTSBot(name, sims)
