"""Model-based planning seat: `planner[:PARAM]` (Phase 4 bot work, 2026-11-19).

Idea: Bus is deterministic with perfect information, and an opponent's play is usually well
described by one of a small set of known policies. The planner
1. **identifies** each opponent from the game history: a posterior over candidate policy models
   (`ppo-long`, `ppo-smoke` greedy nets, the heuristic, uniform random), updated from every
   observed non-forced opponent decision;
2. **plans** each of its own decisions by evaluating every candidate action (pruned by prior) with
   batched lock-step rollouts to the end of the game, opponents played by their identified model
   (greedy in one rollout, sampled in the rest), itself played by a fast base policy;
3. picks the action with the best mean outcome (win share first, score margin as tie-break).

Everything works on `rules.Core`, never on pyspiel states, to keep rollouts cheap.

PARAM (optional) is `R` or `R-K`: R rollouts per candidate (default 4), K max candidates
(default 10).
"""

# pyright: basic
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyspiel

from bots.gumbel import sparse_forward
from bots.heuristic import DEFAULT_WEIGHTS, _Ctx  # pyright: ignore[reportPrivateUsage]
from bots.net import load_weights
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS, decode_action

END = rules.PHASE_END
# Opponent models; a "~" suffix means the net samples from its softmax (T = 1), else greedy.
# name -> (base policy, softmax temperature (0 = greedy), share of uniform-random moves,
#          share of moves sampled from the T = 1 softmax instead of the greedy move)
SPECS: dict[str, tuple[str, float, float, float]] = {
    "ppo-long": ("ppo-long", 0.0, 0.0, 0.0),
    "ppo-long|.1": ("ppo-long", 0.0, 0.0, 0.1),
    "ppo-long|.3": ("ppo-long", 0.0, 0.0, 0.3),
    "ppo-long~.5": ("ppo-long", 0.5, 0.0, 0.0),
    "ppo-long~": ("ppo-long", 1.0, 0.0, 0.0),
    "ppo-long~mix": ("ppo-long", 1.0, 0.5, 0.0),
    "ppo-smoke": ("ppo-smoke", 0.0, 0.0, 0.0),
    "ppo-smoke~": ("ppo-smoke", 1.0, 0.0, 0.0),
    "heuristic": ("heuristic", 0.0, 0.0, 0.0),
    "random": ("random", 0.0, 1.0, 0.0),
}
MODELS = tuple(SPECS)
RANDOM_AS = os.environ.get("PLANNER_RANDOM_AS", "ppo-long~mix")
NET_MODELS = ("ppo-long", "ppo-smoke")
STOCHASTIC = tuple(m for m, (_, t, x, d) in SPECS.items() if t > 0 or x > 0 or d > 0)
# Log prior over MODELS: favour the strong net, hedge towards its sampled versions.
LOG_PRIOR = np.log(np.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1]))


# --------------------------------------------------------------------------------------------
# policies on cores


class _NetPolicy:
    def __init__(self, name: str) -> None:
        self.w = load_weights(name)
        self.fwd = sparse_forward(self.w)
        self.enc = self.w.encoder

    def logits_batch(self, cores: list[Any], legals: list[list[int]]) -> np.ndarray:
        obs = np.stack([self.enc(c) for c in cores]).astype(np.float32)
        mask = np.zeros((len(cores), NUM_DISTINCT_ACTIONS), np.bool_)
        for r, lg in enumerate(legals):
            mask[r, lg] = True
        logits, _ = self.fwd(obs, mask)
        return logits


_STATE_SHIM: Any = None


def _shim(core: Any) -> Any:
    global _STATE_SHIM
    if _STATE_SHIM is None or _STATE_SHIM.core.players != core.players:
        _STATE_SHIM = pyspiel.load_game("python_bus", {"players": core.players}).new_initial_state()
    _STATE_SHIM.core = core
    return _STATE_SHIM


def heuristic_scores(core: Any, legal: list[int]) -> np.ndarray:
    ctx = _Ctx(_shim(core), DEFAULT_WEIGHTS)
    return np.array([ctx.score(a) for a in legal], np.float64)


def _heuristic_pick(core: Any, legal: list[int], rng: random.Random) -> int:
    sc = heuristic_scores(core, legal)
    best = np.flatnonzero(sc >= sc.max() - 1e-9)
    return legal[int(best[0])] if len(best) == 1 else legal[int(rng.choice(list(best)))]


_NETS: dict[str, _NetPolicy] = {}


def net(name: str) -> _NetPolicy:
    if name not in _NETS:
        _NETS[name] = _NetPolicy(name)
    return _NETS[name]


def skip_forced(core: Any) -> tuple[Any, list[int]]:
    while core.phase != END:
        legal = rules.legal_actions(core)
        if len(legal) != 1:
            return core, legal
        core = rules.apply(core, legal[0])
    return core, []


# --------------------------------------------------------------------------------------------
# opponent identification


@dataclass
class _Tracker:
    """Per-game log-likelihoods of each seat under each model; extended incrementally."""

    players: int
    history: list[int] = field(default_factory=list)
    core: Any = None
    loglik: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obs_count: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.core = rules.initial_core(self.players)
        self.loglik = np.tile(LOG_PRIOR, (self.players, 1))
        self.obs_count = np.zeros(self.players)

    def extend(self, history: list[int], me: int) -> None:
        for a in history[len(self.history) :]:
            core = self.core
            legal = rules.legal_actions(core)
            p = core.current
            if len(legal) > 1 and p != me:
                self.loglik[p] += _model_logliks(core, legal, a)
                self.obs_count[p] += 1
            self.core = rules.apply(core, a)
            self.history.append(a)

    def posterior(self, p: int) -> np.ndarray:
        ll = self.loglik[p]
        e = np.exp(ll - ll.max())
        return e / e.sum()


def _model_logliks(core: Any, legal: list[int], a: int) -> np.ndarray:
    out = np.zeros(len(MODELS))
    k = len(legal)
    idx = legal.index(a)
    eps = 0.03
    logits = {n: net(n).logits_batch([core], [legal])[0][legal] for n in NET_MODELS}
    hbest = np.zeros(0)
    for m, name in enumerate(MODELS):
        base, temp, mix, dev = SPECS[name]
        if base in NET_MODELS:
            pr = _probs(logits[base], temp)
            if dev > 0:
                pr = (1 - dev) * pr + dev * _probs(logits[base], 1.0)
        elif base == "heuristic":
            if not len(hbest):
                sc = heuristic_scores(core, legal)
                best = (sc >= sc.max() - 1e-9).astype(np.float64)
                hbest = best / best.sum()
            pr = hbest
        else:
            pr = np.full(k, 1.0 / k)
        mix = max(mix, eps)
        out[m] = math.log((1 - mix) * pr[idx] + mix / k)
    return out


def _probs(lg: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        best = (lg >= lg.max() - 1e-6).astype(np.float64)
        return best / best.sum()
    pr = np.exp((lg - lg.max()) / temp)
    return pr / pr.sum()


# --------------------------------------------------------------------------------------------
# lock-step rollouts


_PHASE_ACT = {
    rules.PHASE_LN: rules.ACT_LINES,
    rules.PHASE_PX: rules.ACT_PASSENGERS,
    rules.PHASE_BD: rules.ACT_BUILDINGS,
    rules.PHASE_VP: rules.ACT_VRROOOMM,
    rules.PHASE_VD: rules.ACT_VRROOOMM,
}
_WASTE_ACTS = (rules.ACT_LINES, rules.ACT_PASSENGERS, rules.ACT_BUILDINGS, rules.ACT_VRROOOMM)


@dataclass
class _Roll:
    core: Any
    legal: list[int]
    sample: bool  # opponents sampled (else greedy)
    tag: int  # candidate index
    me: int = 0
    waste: int = 0  # own markers resolved with no effect (see tools/bot_eval.py)
    board: Any = None  # full action board of the round being resolved
    entered: set[tuple[int, int]] = field(default_factory=set)
    rng: random.Random | None = None  # opponents' sampling (common random numbers)

    def step(self, action: int) -> None:
        """Apply `action`, then forced moves, tracking own wasted markers."""
        pre = self.core
        core = rules.apply(pre, action)
        if pre.phase == rules.PHASE_SEL and core.phase != rules.PHASE_SEL:
            kind, p = decode_action(action)
            board = [list(x) for x in pre.action_board]
            if kind == "select":
                board[p[0]].append(pre.current)
            self.board = board
            self.entered = set()
            buses = board[rules.ACT_BUSES]
            if buses and buses[0] == self.me and pre.seats[self.me].buses >= 5:
                self.waste += 1
            self._observe(core, pre.round)
        else:
            self._observe(core, pre.round)
        while core.phase != END:
            legal = rules.legal_actions(core)
            if len(legal) != 1:
                self.core, self.legal = core, legal
                return
            r0 = core.round
            core = rules.apply(core, legal[0])
            self._observe(core, r0)
        self.core, self.legal = core, []

    def _observe(self, core: Any, prev_round: int) -> None:
        a = _PHASE_ACT.get(core.phase)
        if a is not None and core.current == self.me:
            self.entered.add((a, core.cursor))
        if self.board is not None and (core.round != prev_round or core.phase == END):
            if "last_stone" not in core.end_types:
                for act in _WASTE_ACTS:
                    for k, owner in enumerate(self.board[act]):
                        if owner == self.me and (act, k) not in self.entered:
                            self.waste += 1
            self.board = None


WASTE_PENALTY = float(os.environ.get("PLANNER_WASTE", "0.1"))
MARGIN_W = 0.05


def _outcome(core: Any, me: int) -> float:
    scores = rules.final_scores(core)
    winners = rules.winners(core)
    win = (1.0 / len(winners)) if me in winners else 0.0
    best_opp = max(s for i, s in enumerate(scores) if i != me)
    margin = scores[me] - best_opp
    return win + MARGIN_W * max(-10.0, min(10.0, margin))


def run_rollouts(
    rolls: list[_Roll], me: int, seat_models: list[str], self_model: str, rng: random.Random,
    temp: float = 1.0,
) -> list[float]:  # fmt: skip
    """Play every rollout to the end; returns outcomes for `me`."""
    live = list(range(len(rolls)))
    for r in rolls:
        r.me = me
        if r.legal == []:
            r.core, r.legal = skip_forced(r.core)
    while True:
        live = [i for i in live if rolls[i].core.phase != END]
        if not live:
            break
        groups: dict[str, list[int]] = {}
        for i in live:
            p = rolls[i].core.current
            model = self_model if p == me else seat_models[p]
            groups.setdefault(SPECS[model][0], []).append(i)
        for model, ids in groups.items():
            if model in NET_MODELS:
                lg = net(model).logits_batch(
                    [rolls[i].core for i in ids], [rolls[i].legal for i in ids]
                )
                for row, i in enumerate(ids):
                    r = rolls[i]
                    sub = lg[row][r.legal]
                    p = r.core.current
                    _, t, mix, dev = SPECS[seat_models[p]] if p != me else ("", 0.0, 0.0, 0.0)
                    if r.sample and p != me and (t > 0 or mix > 0 or dev > 0):
                        g = r.rng or rng
                        if dev > 0 and g.random() < dev:
                            t = 1.0
                        if mix > 0 and g.random() < mix:
                            a = g.choice(r.legal)
                        elif t > 0:
                            pr = _probs(sub, t)
                            j = min(int(np.searchsorted(np.cumsum(pr), g.random())), len(pr) - 1)
                            a = r.legal[j]
                        else:
                            a = r.legal[int(np.argmax(sub))]
                    else:
                        a = r.legal[int(np.argmax(sub))]
                    r.step(a)
            else:
                for i in ids:
                    r = rolls[i]
                    if model == "heuristic":
                        a = _heuristic_pick(r.core, r.legal, r.rng or rng)
                    else:
                        a = (r.rng or rng).choice(r.legal)
                    r.step(a)
    return [_outcome(r.core, me) - WASTE_PENALTY * r.waste for r in rolls]


def outcome_vector(r: _Roll) -> np.ndarray:
    """Every seat's outcome at a finished rollout; the tracked seat pays its waste penalty."""
    n = r.core.players
    v = np.array([_outcome(r.core, p) for p in range(n)])
    v[r.me] -= WASTE_PENALTY * r.waste
    return v


# --------------------------------------------------------------------------------------------
# the bot


def surely_wasted(core: Any, action: int) -> bool:
    """True if selecting `action` now places a marker that cannot have any effect: Buses with
    5 buses, or a dwindling action at a letter whose power stays <= 0 even if the round's Buses
    marker raises M (Vrrooomm! is left to the rollouts)."""
    kind, p = decode_action(action)
    if kind != "select":
        return False
    a = p[0]
    me = core.seats[core.current]
    if a == rules.ACT_BUSES:
        return me.buses >= 5
    if a not in (rules.ACT_LINES, rules.ACT_PASSENGERS, rules.ACT_BUILDINGS):
        return False
    letter = len(core.action_board[a])
    m = rules.bus_max(core)
    if a != rules.ACT_LINES and m < 5:
        taken = core.action_board[rules.ACT_BUSES]
        # Buses resolves before Passengers/Buildings: M may still rise by one this round.
        if not taken or core.seats[taken[0]].buses == m:
            m += 1
    power = m - letter + (1 if a == rules.ACT_LINES and core.players == 5 else 0)
    if power <= 0:
        return True
    if a == rules.ACT_PASSENGERS and core.supply <= 0:
        return True
    return a == rules.ACT_BUILDINGS and not rules.lowest_open_tier_sites(core)


class PlannerBot:
    def __init__(
        self, rollouts: int = 4, max_cands: int = 10, self_model: str = "ppo-long",
        early: bool = False,
    ) -> None:  # fmt: skip
        self.early = early
        self.rollouts = rollouts
        self.max_cands = max_cands
        self.self_model = self_model
        self._tracker: _Tracker | None = None
        self.last_info: dict[str, Any] = {}

    def _track(self, state: pyspiel.State, me: int) -> _Tracker:
        hist = list(state.history())
        players = state.get_game().num_players()
        t = self._tracker
        if t is None or t.players != players or t.history != hist[: len(t.history)]:
            t = _Tracker(players)
        t.extend(hist, me)
        self._tracker = t
        return t

    def seat_models(self, t: _Tracker, me: int) -> list[str]:
        out = []
        for p in range(t.players):
            if p == me:
                out.append(self.self_model)
                continue
            m = MODELS[int(np.argmax(t.posterior(p)))]
            # A seat that looks random may be a search bot (e.g. MCTS) whose moves no net
            # predicts; plan against a competent-but-noisy player instead.
            if m == "random" and RANDOM_AS:
                m = RANDOM_AS
            out.append(m)
        return out

    def _batch(
        self, core: Any, legal: list[int], me: int, models: list[str], cands: list[int],
        which: Any, seeds: list[int], tot: np.ndarray, cnt: np.ndarray, rng: random.Random,
    ) -> None:  # fmt: skip
        rolls: list[_Roll] = []
        for ci in which:
            for sd in seeds:
                roll = _Roll(core, legal, sample=True, tag=ci, me=me, rng=random.Random(sd))
                roll.step(cands[ci])
                rolls.append(roll)
        out = run_rollouts(rolls, me, models, self.self_model, rng)
        for r, v in zip(rolls, out, strict=True):
            tot[r.tag] += v
            cnt[r.tag] += 1

    def candidates(self, core: Any, legal: list[int]) -> list[int]:
        if core.phase == rules.PHASE_SEL:
            useful = [a for a in legal if not surely_wasted(core, a)]
            legal = useful or legal
        if len(legal) <= self.max_cands:
            return list(legal)
        lg = net("ppo-long").logits_batch([core], [legal])[0][legal]
        order = list(np.argsort(-lg))
        keep = [legal[i] for i in order[: self.max_cands - 1]]
        h = legal[int(np.argmax(heuristic_scores(core, legal)))]
        if h not in keep:
            keep.append(h)
        else:
            keep.append(legal[order[self.max_cands - 1]])
        return keep

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        core = state.core  # type: ignore[attr-defined]
        me = core.current
        t = self._track(state, me)
        models = self.seat_models(t, me)
        cands = self.candidates(core, legal)
        stochastic = any(m in STOCHASTIC for i, m in enumerate(models) if i != me)
        k = len(cands)
        tot = np.zeros(k)
        cnt = np.zeros(k)
        if not stochastic:
            self._batch(core, legal, me, models, cands, range(k), [0], tot, cnt, rng)
        else:
            # Sequential halving over a budget of `rollouts * k` rollouts, common seeds.
            budget = self.rollouts * k
            alive = list(range(k))
            rounds = max(1, math.ceil(math.log2(k)))
            seed0 = rng.randrange(2**30)
            j = 0
            while True:
                per = max(1, budget // (rounds * len(alive)))
                seeds = [seed0 + j + i for i in range(per)]
                j += per
                self._batch(core, legal, me, models, cands, alive, seeds, tot, cnt, rng)
                if len(alive) <= 1:
                    break
                mean = tot / np.maximum(cnt, 1)
                alive.sort(key=lambda c: -mean[c])
                # early stop: a clear leader (ahead by more than a win's worth of noise)
                if self.early and mean[alive[0]] - mean[alive[1]] > 2.5 / math.sqrt(cnt[alive[0]]):
                    alive = alive[:1]
                    break
                alive = alive[: max(1, len(alive) // 2)]
                if len(alive) == 1:
                    break
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), -1e9)
        # among the most-visited candidates (the halving finalists), pick the best mean
        top = cnt >= cnt.max()
        best = int(np.argmax(np.where(top, mean, -1e9)))
        self.last_info = {"models": models, "cands": cands, "values": mean.tolist()}
        return cands[best]


# --------------------------------------------------------------------------------------------
# tree search variant


# Seats whose identified model is one of these are searched as maximisers (max-n) in the tree;
# the others follow their model (greedy models collapse to one child).
MAXN_MODELS = tuple(
    m for m in os.environ.get("PLANNER_MAXN", "random,ppo-long~mix,ppo-long|.3").split(",") if m
)


@dataclass
class _Node:
    core: Any
    legal: list[int]
    player: int
    mode: str = ""  # "puct" | "det" | "sample" | "" (terminal)
    prior: np.ndarray | None = None
    n: np.ndarray | None = None
    w: np.ndarray | None = None
    children: dict[int, _Node] = field(default_factory=dict)
    visits: int = 0
    terminal_value: np.ndarray | None = None


class TreePlannerBot(PlannerBot):
    """Multi-player tree search over the true state (max-n at our nodes and at unpredictable
    opponents' nodes, model-following elsewhere), leaves valued by one model rollout each."""

    def __init__(self, sims: int = 96, batch: int = 16, c_puct: float = 1.5) -> None:
        super().__init__()
        self.sims, self.batch, self.c = sims, batch, c_puct

    # -- node setup
    def _make(
        self, core: Any, legal: list[int], me: int, models: list[str], raw: list[str]
    ) -> _Node:
        if core.phase == END:
            nd = _Node(core, [], -1)
            r = _Roll(core, [], False, 0, me=me)
            nd.terminal_value = outcome_vector(r)
            return nd
        p = core.current
        nd = _Node(core, legal, p)
        if p == me or raw[p] in MAXN_MODELS:
            nd.mode = "puct"
        elif models[p] in STOCHASTIC:
            nd.mode = "sample"
        else:
            nd.mode = "det"
        return nd

    def _expand(self, nd: _Node, me: int, models: list[str], rng: random.Random) -> None:
        legal = nd.legal
        if nd.player == me and nd.core.phase == rules.PHASE_SEL:
            keep = [a for a in legal if not surely_wasted(nd.core, a)]
            if keep and len(keep) < len(legal):
                nd.legal = legal = keep
        k = len(legal)
        model = self.self_model if nd.player == me else models[nd.player]
        base, temp, mix, dev = SPECS[model]
        if base in NET_MODELS:
            lg = net(base).logits_batch([nd.core], [legal])[0][legal]
            if nd.mode == "puct":
                pr = 0.5 * _probs(lg, 1.0) + 0.5 / k
            elif nd.mode == "det":
                pr = _probs(lg, 0.0)
            else:
                pr = _probs(lg, temp) if temp > 0 else _probs(lg, 0.0)
                if dev > 0:
                    pr = (1 - dev) * pr + dev * _probs(lg, 1.0)
                pr = (1 - mix) * pr + mix / k
        elif base == "heuristic":
            sc = heuristic_scores(nd.core, legal)
            best = (sc >= sc.max() - 1e-9).astype(np.float64)
            pr = best / best.sum()
            if nd.mode == "puct":
                pr = 0.75 * pr + 0.25 / k
        else:
            pr = np.full(k, 1.0 / k)
        nd.prior = pr
        nd.n = np.zeros(k)
        nd.w = np.zeros(k)

    def _select(self, nd: _Node, rng: random.Random) -> int:
        assert nd.prior is not None and nd.n is not None and nd.w is not None
        if nd.mode == "det":
            return int(np.argmax(nd.prior))
        if nd.mode == "sample":
            return min(int(np.searchsorted(np.cumsum(nd.prior), rng.random())), len(nd.prior) - 1)
        tot = nd.n.sum()
        visited = nd.n > 0
        fpu = (nd.w[visited].sum() / nd.n[visited].sum() - 0.2) if visited.any() else 0.5
        q = np.where(visited, nd.w / np.maximum(nd.n, 1), fpu)
        u = self.c * nd.prior * math.sqrt(tot + 1) / (1 + nd.n)
        return int(np.argmax(q + u))

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        core = state.core  # type: ignore[attr-defined]
        me = core.current
        t = self._track(state, me)
        models = self.seat_models(t, me)
        raw = [self.self_model if p == me else MODELS[int(np.argmax(t.posterior(p)))]
               for p in range(t.players)]  # fmt: skip
        cands = self.candidates(core, legal)
        root = self._make(core, cands, me, models, raw)
        k = len(cands)
        root.prior = np.full(k, 1.0 / k)
        root.n = np.zeros(k)
        root.w = np.zeros(k)
        if k == 1:
            return cands[0]
        # Sequential halving at the root (each candidate's sims go through its subtree).
        alive = list(range(k))
        rounds = max(1, math.ceil(math.log2(k)))
        while True:
            per = max(1, self.sims // (rounds * len(alive)))
            forced = [c for c in alive for _ in range(per)]
            for j in range(0, len(forced), self.batch):
                self._simulate(root, forced[j : j + self.batch], me, models, raw, rng)
            if len(alive) == 1:
                break
            q = root.w / np.maximum(root.n, 1)
            alive.sort(key=lambda c: -q[c])
            alive = alive[: max(1, len(alive) // 2)]
            if len(alive) == 1:
                break
        n = root.n
        q = root.w / np.maximum(n, 1)
        top = n >= n.max()
        best = int(np.argmax(np.where(top, q, -1e9)))
        self.last_info = {"models": raw, "cands": cands, "values": q.tolist(), "n": n.tolist()}
        return cands[best]

    def _simulate(
        self, root: _Node, first: list[int], me: int, models: list[str], raw: list[str],
        rng: random.Random,
    ) -> None:  # fmt: skip
        paths: list[list[tuple[_Node, int]]] = []
        leaves: list[_Node] = []
        for i0 in first:
            nd, path = root, []
            while True:
                if nd.terminal_value is not None:
                    break
                if nd is not root and nd.visits == 0:
                    break
                if nd.prior is None:
                    self._expand(nd, me, models, rng)
                i = i0 if nd is root else self._select(nd, rng)
                path.append((nd, i))
                nd.visits += 1
                assert nd.n is not None
                nd.n[i] += 1  # virtual visit; value added at backup
                child = nd.children.get(i)
                if child is None:
                    c2, lg2 = skip_forced(rules.apply(nd.core, nd.legal[i]))
                    child = self._make(c2, lg2, me, models, raw)
                    nd.children[i] = child
                nd = child
            nd.visits += 1
            paths.append(path)
            leaves.append(nd)
        # Replay each path from the root through a waste-tracking roll, then roll out.
        rolls = []
        for j, path in enumerate(paths):
            r = _Roll(root.core, root.legal, True, j, me=me, rng=random.Random(rng.random()))
            for nd, i in path:
                r.step(nd.legal[i])
            rolls.append(r)
        run_rollouts(rolls, me, models, self.self_model, rng)
        vals: list[np.ndarray | None] = [outcome_vector(r) for r in rolls]
        for path, v in zip(paths, vals, strict=True):
            assert v is not None
            for nd, i in path:
                assert nd.w is not None
                nd.w[i] += v[nd.player]


def parse_tree_param(param: str | None) -> TreePlannerBot:
    if param is None:
        return TreePlannerBot()
    sims, _, c = param.partition("-")
    if not sims.isdigit():
        raise ValueError(f"bad tree planner param {param!r}")
    return TreePlannerBot(sims=int(sims), c_puct=float(c) if c else 1.5)


def parse_planner_param(param: str | None) -> PlannerBot:
    if param is None:
        return PlannerBot()
    early = param.endswith("e")
    r, _, k = param.removesuffix("e").partition("-")
    if not r.isdigit() or (k and not k.isdigit()):
        raise ValueError(f"bad planner param {param!r}")
    return PlannerBot(rollouts=int(r), max_cands=int(k) if k else 10, early=early)
