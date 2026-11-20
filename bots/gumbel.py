"""Gumbel search seat: `gumbel:<SIMS>-<model>` (Phase 4 unit 8g, design in `docs/GUMBEL_AZ.md`).

Numpy only (never imports torch). The search is Danihelka et al. 2022 ("Policy improvement by
planning with Gumbel", as in `mctx.gumbel_muzero_policy`) on the true simulator, adapted to N
players:

- **Root:** g ~ Gumbel(0, `gumbel_scale`) per legal action (0 for play and evaluation, 1 in AZ
  self-play); sequential halving over the top m = min(`m`, #legal) actions by g + logits with the
  `mctx` visit schedule (`considered_visits`); the action played is the argmax over the actions
  with the most visits of g + logits + σ(completed q̂).
- **Interior nodes:** argmax_a [π'(a) − N(a) / (1 + Σ N)] with π' = softmax(logits + σ(q̂)).
- **Completed Q:** visited actions use their mean backed-up value for the node's mover; unvisited
  ones use v_mix = (v̂ + Σ N · Σ_visited π q / Σ_visited π) / (1 + Σ N); q̂ is min-max rescaled
  over the node's actions; σ(q̂) = (`c_visit` + max N) · `c_scale` · q̂.
- **Values** are per-seat vectors in absolute seat order, backed up unchanged; each node's mover
  reads its own entry (max^n with mean backups). The net's 5-wide ego value head (index 0 = seat to
  move) is rotated to absolute seats on expansion. Terminal nodes use the training quantity
  R = `mix` · win_loss (R-071) + (1 − `mix`) · margin (R-072, relative, stones counted), never
  `state.returns()`. `target` swaps the margin for another relative term (`relative_term`:
  leader-relative or rank); the default is the R-072 margin.
- **Opponent model (optional, off by default):** with `opp_temp` ≠ 1 or `opp_mix` > 0, nodes
  whose mover is not the root's (`opp_scope="opponents"`) or all non-root nodes (`"nonroot"`)
  use the prior (1 − `opp_mix`) · softmax(logits / `opp_temp`) + `opp_mix` / k instead of the
  net's (`opponent_prior`); the interior rule then spreads opponent visits over plausible replies
  instead of the near-one-hot argmax. Design and literature: `docs/GUMBEL_AZ.md` §10.2.
- **Forced moves** (single legal action) are applied while expanding an edge, so one simulation is
  one evaluated decision node or one terminal; forced edges cost nothing.
- Tree nodes hold the immutable engine `Core` (`bus.rules.apply` returns a new one); no pyspiel
  clones, no transposition table, no subtree reuse between moves.

`GumbelSearch` is driven from outside (`want()` returns the core to evaluate, `give()` takes the
net's output), so `run_searches` can batch one forward over many open searches; the AZ trainer
(unit 8h) reuses it across games. `GumbelBot` runs one search per non-forced decision.

The seat grammar is `gumbel:<SIMS>-<model>` like `mcts_net` (model names resolve only through
`models/manifest.yaml`). The search constants are not part of the seat type: they default to
`GumbelConfig` and can be overridden for experiments with `BUS_GUMBEL_C_SCALE`,
`BUS_GUMBEL_C_VISIT`, `BUS_GUMBEL_M`, `BUS_GUMBEL_OPP_TEMP`, `BUS_GUMBEL_OPP_MIX`,
`BUS_GUMBEL_OPP_SCOPE` and `BUS_GUMBEL_TARGET` (tool use only, as `BUS_MCTS_NET_C`).
Tool-only for now: the server's lobby model list does not offer it (Q-AZ7).
"""

# pyright: basic
from __future__ import annotations

import logging
import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyspiel

from bots.heuristic import HeuristicBot
from bots.mcts import MAX_SIMS
from bots.net import load_weights, resolve_model
from bots.net_forward import MASK_FILL, NetWeights
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS as NUM_ACTIONS
from train.features import encode

log = logging.getLogger(__name__)

Array = np.ndarray[Any, Any]
# (obs (B, OBS), mask (B, A)) -> (masked logits (B, A), ego values (B, 5))
Forward = Callable[[Array, Array], tuple[Array, Array]]

DEFAULT_SIMS = 32
DEFAULT_M = 16
C_VISIT = 50.0  # mctx qtransform_completed_by_mix_value defaults
C_SCALE = 0.1
VALUE_MIX = 0.5  # R = 0.5 win_loss + 0.5 margin (user decision 2026-11-17, Q-AZ2)
# Relative term of R (see ``relative_term``): "mean" = R-072 margin (default, unchanged), "leader"
# = own score minus the best opponent score, "rank" = pairwise placement.
VALUE_TARGETS = ("mean", "leader", "rank")
OPP_SCOPES = ("opponents", "nonroot")


@dataclass(frozen=True)
class GumbelConfig:
    m: int = DEFAULT_M  # considered actions at the root
    c_visit: float = C_VISIT
    c_scale: float = C_SCALE
    gumbel_scale: float = 0.0  # 0 = deterministic (play/eval), 1 = AZ self-play
    mix: float = VALUE_MIX  # terminal value R = mix * win_loss + (1 - mix) * relative term
    target: str = "mean"  # relative term of R at terminal nodes (VALUE_TARGETS)
    # Opponent model at non-root nodes (``opponent_prior``); 1.0 / 0.0 = the net prior unchanged.
    opp_temp: float = 1.0  # prior temperature: logits / opp_temp
    opp_mix: float = 0.0  # share of uniform over legal actions mixed into the prior
    opp_scope: str = "opponents"  # "opponents": movers other than the root's; "nonroot": all

    def __post_init__(self) -> None:
        if self.m < 1 or self.c_visit < 0 or self.c_scale < 0 or self.gumbel_scale < 0:
            raise ValueError("gumbel needs m >= 1 and non-negative c_visit, c_scale, scale")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("gumbel value mix must be in [0, 1]")
        if self.target not in VALUE_TARGETS:
            raise ValueError(f"gumbel value target must be one of {VALUE_TARGETS}")
        if self.opp_temp <= 0 or not 0.0 <= self.opp_mix <= 1.0:
            raise ValueError("gumbel needs opp_temp > 0 and opp_mix in [0, 1]")
        if self.opp_scope not in OPP_SCOPES:
            raise ValueError(f"gumbel opp_scope must be one of {OPP_SCOPES}")

    @property
    def models_opponents(self) -> bool:
        """True if non-root priors are changed (otherwise the search is the unit 8g one)."""
        return self.opp_temp != 1.0 or self.opp_mix > 0.0

    @classmethod
    def from_env(cls) -> GumbelConfig:
        """Defaults, overridden by BUS_GUMBEL_C_SCALE / _C_VISIT / _M / _OPP_TEMP / _OPP_MIX /
        _OPP_SCOPE / _TARGET (experiment knobs)."""
        env = os.environ
        return cls(
            m=int(env.get("BUS_GUMBEL_M", DEFAULT_M)),
            c_visit=float(env.get("BUS_GUMBEL_C_VISIT", C_VISIT)),
            c_scale=float(env.get("BUS_GUMBEL_C_SCALE", C_SCALE)),
            target=env.get("BUS_GUMBEL_TARGET", "mean"),
            opp_temp=float(env.get("BUS_GUMBEL_OPP_TEMP", 1.0)),
            opp_mix=float(env.get("BUS_GUMBEL_OPP_MIX", 0.0)),
            opp_scope=env.get("BUS_GUMBEL_OPP_SCOPE", "opponents"),
        )


# ------------------------------------------------------------------ pieces


def considered_visits(m: int, n: int) -> tuple[int, ...]:
    """Sequential-halving root schedule (mctx ``get_sequence_of_considered_visits``).

    Entry i is the visit count an action must have to receive simulation i.
    """
    if m <= 1:
        return tuple(range(n))
    log2max = math.ceil(math.log2(m))
    seq: list[int] = []
    visits = [0] * m
    k = m
    while len(seq) < n:
        extra = max(1, int(n / (log2max * k)))
        for _ in range(extra):
            seq.extend(visits[:k])
            for i in range(k):
                visits[i] += 1
        k = max(2, k // 2)
    return tuple(seq[:n])


def skip_forced(core: Any) -> tuple[Any, list[int]]:
    """Apply single-legal-action moves; return (decision or terminal core, its legal actions)."""
    while core.phase != rules.PHASE_END:
        legal = rules.legal_actions(core)
        if len(legal) != 1:
            return core, legal
        core = rules.apply(core, legal[0])
    return core, []


def relative_term(core: Any, target: str = "mean") -> Array:
    """Per-seat relative score term of R at a finished game, in [-1, 1] (training signal only).

    Final scores s are R-068's (points - stones). All three variants are relative, never
    own-score-only, so a time stone (-1 to its taker) is valued against the other seats:

    - ``mean``: R-072 margin, (s_p - mean of the other seats' s) / 10, clipped to [-1, 1].
    - ``leader``: (s_p - max of the other seats' s) / 10, clipped like R-072. For the leader it
      is the lead over the runner-up; for everyone else the gap to the leader. Seats tied for the
      top score get 0.
    - ``rank``: pairwise placement by final score, 2 * (beaten + ties / 2) / (n - 1) - 1: +1 for
      a strict first, -1 for a strict last, 0 for an all-way tie. Score ties count half (the
      R-069/R-070 tiebreaks are left to the win_loss part of R).
    """
    if target == "mean":
        return np.asarray(rules.returns_for(core, "margin"), float)
    n = core.players
    if core.phase != rules.PHASE_END:
        return np.zeros(n)
    s = np.asarray(rules.final_scores(core), float)
    if target == "leader":
        best_other = np.array([max(s[q] for q in range(n) if q != p) for p in range(n)])
        return np.clip((s - best_other) / 10.0, -1.0, 1.0)
    if target == "rank":
        beaten = (s[:, None] > s[None, :]).sum(axis=1)
        ties = (s[:, None] == s[None, :]).sum(axis=1) - 1
        return 2.0 * (beaten + 0.5 * ties) / (n - 1) - 1.0
    raise ValueError(f"unknown value target {target!r}")


def terminal_values(core: Any, mix: float = VALUE_MIX, target: str = "mean") -> Array:
    """Per-seat R = mix * win_loss + (1 - mix) * relative term at a finished game (R-071, and
    R-072 for the default ``target="mean"``; see ``relative_term``)."""
    wl = np.asarray(rules.returns_for(core, "win_loss"), float)
    if mix >= 1.0:
        return wl
    return mix * wl + (1.0 - mix) * relative_term(core, target)


class Node:
    __slots__ = (
        "core", "mover", "actions", "logits", "prior", "value", "N", "W", "tot", "nmax",
        "children", "terminal",
    )  # fmt: skip

    def __init__(
        self, core: Any, actions: Sequence[int], n: int, mix: float, target: str = "mean"
    ) -> None:
        self.core = core
        self.terminal = core.phase == rules.PHASE_END
        self.mover = -1 if self.terminal else int(core.current)
        self.actions = np.asarray(actions, np.int64)
        k = len(actions)
        self.logits = np.zeros(k)
        self.prior = np.full(k, 1.0 / max(k, 1))
        self.tot = 0  # sum of N
        self.nmax = 0  # max of N
        self.value = terminal_values(core, mix, target) if self.terminal else np.zeros(n)
        self.N = np.zeros(k, np.int64)
        self.W = np.zeros((k, n))  # summed backed-up value vectors, absolute seats
        self.children: list[Node | None] = [None] * k

    def set_eval(
        self, logits_full: Array, v_ego: Array, temp: float = 1.0, eps: float = 0.0
    ) -> None:
        """Store the net prior over this node's actions and its value in absolute seat order.

        ``temp`` / ``eps`` replace the prior by the opponent model ``opponent_prior`` (the
        defaults leave the net prior untouched, bit for bit).
        """
        n = len(self.value)
        lg = np.asarray(logits_full, float)[self.actions]
        if temp != 1.0 or eps > 0.0:
            lg = opponent_prior(lg, temp, eps)
        self.logits = lg - lg.max()
        e = np.exp(self.logits)
        self.prior = e / e.sum()
        # ego index k is seat (mover + k) % n
        self.value = np.asarray([v_ego[(s - self.mover) % n] for s in range(n)], float)


def opponent_prior(logits: Array, temp: float, eps: float) -> Array:
    """Log-probabilities of the opponent model (1 - eps) softmax(logits / temp) + eps / k.

    Used as the prior of non-root nodes when ``GumbelConfig.models_opponents``: the interior rule
    argmax [π'(a) - N(a) / (1 + ΣN)] spreads a node's visits in proportion to π', so the prior
    there is the search's model of how that seat plays. A temperature above 1 or a uniform share
    keeps plausible but non-argmax replies in the tree.
    """
    z = logits / temp
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    if eps > 0.0:
        p = (1.0 - eps) * p + eps / len(p)
    return np.log(np.maximum(p, 1e-300))


def _softmax(x: Array) -> Array:
    e = np.exp(x - x.max())
    return e / e.sum()


def completed_q(node: Node) -> Array:
    """Completed Q for the node's mover, min-max rescaled to [0, 1] (v_mix for unvisited)."""
    p = node.mover
    v = node.value[p]
    vis = node.N > 0
    tot = node.tot
    q = np.empty(len(node.N))
    if tot:
        qv = node.W[vis, p] / node.N[vis]
        pv = node.prior[vis]
        psum = float(pv.sum())
        mean_q = float((pv * qv).sum() / psum) if psum > 0 else float(qv.mean())
        q[:] = (v + tot * mean_q) / (1 + tot)
        q[vis] = qv
    else:
        q[:] = v
    lo, hi = q.min(), q.max()
    return (q - lo) / max(hi - lo, 1e-8)


def sigma(qhat: Array, node: Node, cfg: GumbelConfig) -> Array:
    return (cfg.c_visit + node.nmax) * cfg.c_scale * qhat


def improved_policy(node: Node, cfg: GumbelConfig) -> Array:
    """softmax(logits + σ(completed q̂)) over the node's legal actions (the AZ policy target)."""
    return _softmax(node.logits + sigma(completed_q(node), node, cfg))


# ------------------------------------------------------------------ search


@dataclass
class SearchResult:
    action: int  # action id to play
    policy: Array  # improved policy over `actions`
    actions: Array  # root legal action ids
    value: Array  # root search value vector (absolute seats)
    visits: Array  # root visit counts over `actions`


@dataclass
class GumbelSearch:
    """One search from ``root_core`` (a decision state), driven by ``want`` / ``give``.

    ``want()`` returns the core whose net evaluation is needed next, or None once all ``sims``
    simulations are done; ``give(logits, v_ego)`` supplies that evaluation (masked full logits
    and the 5-wide ego value). The root is evaluated first and is not a simulation.
    """

    root_core: Any
    sims: int
    cfg: GumbelConfig = field(default_factory=GumbelConfig)
    rng: np.random.Generator | None = None
    n: int = 0
    root: Node = field(init=False)
    gumbel: Array = field(init=False)
    schedule: tuple[int, ...] = field(init=False)
    done_sims: int = 0
    _path: list[tuple[Node, int]] = field(default_factory=list)
    _leaf: Node | None = None

    def __post_init__(self) -> None:
        if self.root_core.phase == rules.PHASE_END:
            raise ValueError("cannot search a terminal state")
        if self.sims < 1:
            raise ValueError("gumbel search needs at least one simulation")
        self.n = int(self.root_core.players)
        legal = rules.legal_actions(self.root_core)
        self.root = Node(self.root_core, legal, self.n, self.cfg.mix, self.cfg.target)
        self._leaf = self.root
        k = len(legal)
        if self.cfg.gumbel_scale > 0:
            rng = self.rng if self.rng is not None else np.random.default_rng()
            self.gumbel = self.cfg.gumbel_scale * rng.gumbel(size=k)
        else:
            self.gumbel = np.zeros(k)
        self.schedule = considered_visits(min(self.cfg.m, k), self.sims)

    @property
    def leaf(self) -> Node | None:
        """The node awaiting evaluation (its ``actions`` are the legal mask), if any."""
        return self._leaf

    def _root_scores(self) -> Array:
        root = self.root
        return self.gumbel + root.logits + sigma(completed_q(root), root, self.cfg)

    def _root_action(self, considered: int) -> int:
        score = np.where(considered == self.root.N, self._root_scores(), -np.inf)
        return int(np.argmax(score))

    def _interior_action(self, node: Node) -> int:
        pi = improved_policy(node, self.cfg)
        return int(np.argmax(pi - node.N / (1 + node.tot)))

    def want(self) -> Any:
        if self._leaf is not None:
            return self._leaf.core
        while self.done_sims < self.sims:
            node, path = self.root, []
            a = self._root_action(self.schedule[self.done_sims])
            while True:
                path.append((node, a))
                child = node.children[a]
                if child is None:
                    core, legal = skip_forced(rules.apply(node.core, int(node.actions[a])))
                    child = Node(core, legal, self.n, self.cfg.mix, self.cfg.target)
                    node.children[a] = child
                    if child.terminal:
                        self._backup(path, child.value)
                        break
                    self._path, self._leaf = path, child
                    return child.core
                if child.terminal:
                    self._backup(path, child.value)
                    break
                node = child
                a = self._interior_action(node)
        return None

    def give(self, logits_full: Array, v_ego: Array) -> None:
        leaf = self._leaf
        if leaf is None:
            raise RuntimeError("give() without a pending evaluation")
        cfg = self.cfg
        if (
            leaf is not self.root
            and cfg.models_opponents
            and (cfg.opp_scope == "nonroot" or leaf.mover != self.root.mover)
        ):
            leaf.set_eval(logits_full, v_ego, cfg.opp_temp, cfg.opp_mix)
        else:
            leaf.set_eval(logits_full, v_ego)
        self._leaf = None
        if leaf is not self.root:
            self._backup(self._path, leaf.value)

    def _backup(self, path: list[tuple[Node, int]], v: Array) -> None:
        for node, a in path:
            na = node.N[a] + 1
            node.N[a] = na
            node.tot += 1
            if na > node.nmax:
                node.nmax = int(na)
            node.W[a] += v
        self.done_sims += 1

    def result(self) -> SearchResult:
        root = self.root
        a = self._root_action(int(root.N.max(initial=0)))
        value = (root.value + root.W.sum(axis=0)) / (1 + root.tot)
        return SearchResult(
            int(root.actions[a]), improved_policy(root, self.cfg), root.actions.copy(), value,
            root.N.copy(),
        )  # fmt: skip


def sparse_forward(weights: NetWeights) -> Forward:
    """``weights.forward`` reading only the first-layer rows of non-zero features.

    Observations are ~5% non-zero, so this skips most of the two 2314x512 input matrices, which
    dominate the cost of small batches (numpy at B = 1 is memory-bound). Same outputs as the
    dense forward up to float32 rounding (~1e-6).
    """
    act = np.tanh if weights.activation == "tanh" else (lambda x: np.maximum(x, 0.0))

    def trunk(x: Array, nz: Array, layers: tuple[tuple[Array, Array], ...]) -> Array:
        (w0, b0), *rest = layers
        h = act(x[:, nz] @ w0[nz] + b0)
        for w, b in rest:
            h = act(h @ w + b)
        return h

    def fwd(obs: Array, mask: Array) -> tuple[Array, Array]:
        x = np.asarray(obs, np.float32)
        nz = np.flatnonzero(x.any(axis=0))
        h = trunk(x, nz, weights.layers)
        logits = np.where(
            np.asarray(mask, bool), h @ weights.pi[0] + weights.pi[1], np.float32(MASK_FILL)
        )
        hv = trunk(x, nz, weights.v_layers) if weights.v_layers else h
        return logits, np.tanh(hv @ weights.v[0] + weights.v[1])

    return fwd


def evaluate_batch(
    forward: Forward, searches: Sequence[GumbelSearch], encoder: Callable[[Any], Array] = encode
) -> None:
    """One batched forward for the pending leaf of every search in ``searches``; ``encoder`` is
    the net's observation (``NetWeights.encoder``; default v2)."""
    nodes = [s.leaf for s in searches]
    obs = np.stack([encoder(nd.core) for nd in nodes if nd is not None]).astype(np.float32)
    mask = np.zeros((len(nodes), NUM_ACTIONS), np.bool_)
    for r, nd in enumerate(nodes):
        assert nd is not None
        mask[r, nd.actions] = True
    logits, values = forward(obs, mask)
    for r, s in enumerate(searches):
        s.give(logits[r], values[r])


def run_searches(
    forward: Forward, searches: Sequence[GumbelSearch], encoder: Callable[[Any], Array] = encode
) -> list[SearchResult]:
    """Run all ``searches`` to completion, one batched forward per wave of leaves."""
    while True:
        waiting = [s for s in searches if s.want() is not None]
        if not waiting:
            return [s.result() for s in searches]
        evaluate_batch(forward, waiting, encoder)


# ------------------------------------------------------------------ seat


class GumbelBot:
    """Plays the root action of a Gumbel search with ``sims`` simulations on ``weights``."""

    def __init__(
        self, weights: NetWeights, sims: int = DEFAULT_SIMS, config: GumbelConfig | None = None,
        name: str = "path",
    ) -> None:  # fmt: skip
        if not 1 <= sims <= MAX_SIMS:
            raise ValueError(f"gumbel simulations must be in 1..{MAX_SIMS}")
        self.weights = weights
        self.sims = sims
        self.config = config if config is not None else GumbelConfig.from_env()
        self.name = name
        self._forward = sparse_forward(weights)

    def search(self, core: Any, rng: random.Random) -> SearchResult:
        np_rng = None
        if self.config.gumbel_scale > 0:  # g = 0 consumes nothing from the game rng
            np_rng = np.random.default_rng(rng.randrange(2**63))
        s = GumbelSearch(core, self.sims, self.config, np_rng)
        return run_searches(self._forward, [s], self.weights.encoder)[0]

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = list(state.legal_actions())
        if len(legal) == 1:
            return legal[0]
        try:
            a = self.search(state.core, rng).action  # type: ignore[attr-defined]
            if a in legal:
                return a
            log.warning("gumbel %r returned illegal action %s; using heuristic", self.name, a)
        except Exception:
            log.exception("gumbel %r failed; using heuristic", self.name)
        return HeuristicBot().choose(state, rng)


def split_param(param: str) -> tuple[int, str]:
    """`<sims>-<model>` -> (sims, model); ValueError if malformed."""
    sims_s, dash, name = param.partition("-")
    if not sims_s.isdigit() or not dash or not name:
        raise ValueError("gumbel takes <sims>-<model>, e.g. gumbel:32-ppo-long")
    return int(sims_s), name


def parse_gumbel_param(param: str | None) -> GumbelBot:
    if param is None:
        raise ValueError("gumbel needs <sims>-<model>, e.g. gumbel:32-ppo-long")
    sims, name = split_param(param)
    resolve_model(name)
    return GumbelBot(load_weights(name), sims, name=name)
