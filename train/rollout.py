# pyright: basic, reportMissingImports=false
"""Rollout actors for PPO self-play (Phase 4 unit 3). Numpy only: never imports torch.

A worker plays full games through ``BusEnv`` with a numpy copy of the network
(``bots.net_forward.NetWeights``), batching the forward pass over several concurrent games.

Seat policies: each game has one policy label per seat (see ``train.league``). ``"current"`` seats
sample from the net being trained and are the only seats that produce training samples (their
samples still get the seat's own terminal shaped return); ``"snap:<id>"`` seats play a frozen net
from ``opponents`` (weights or an npz path, cached per process by label); ``"random"`` plays uniform
random legal actions; any other label is a ``bots.registry`` seat type (``heuristic``, ``mcts:50``).
Net seats are batched per label; bot seats are stepped one at a time on the real state.
``assign_seat_policies`` draws the labels from ``cfg.league`` (pure self-play when it is None).

Rewards and targets. Terminal-only: a seat's reward is 0 at every decision except its last, which
gets the seat's final shaped return (``mix * win_loss + (1 - mix) * margin``). GAE is computed per
seat trajectory (only that seat's own decisions) with the ego value entry (index 0 of the
ego-ordered value vector, i.e. the value for the acting seat) as baseline; the value after the last
decision is 0.
The 5-wide value head is trained on the *undiscounted* final shaped returns of all present seats in
ego order for that trajectory's seat (``vtarget`` / ``vmask``), so entry 0 has the same target as
the seat's own Monte Carlo return, and the other entries are auxiliary.

Dense mode (``cfg.reward == "dense"``, unit 8e). Per-seat potential Phi_p = (score_p - mean opponent
score) / 10 (score = points - stones, ``train.env.potentials``) is recorded for all seats at each of
the seat's own decisions. Rewards are r_t = (1-mix) (Phi(s_t+1) - Phi(s_t)) between own decisions
(opponents' moves count), and the last one is mix * win_loss + (1-mix) (Phi(s_T) - Phi(s_k-1)), so
a seat's rewards sum to mix * win_loss + (1-mix) * unclipped margin. Value targets are
return-to-go per ego entry (clipped to [-1, 1], clipped fraction in ``stats["vclip"]``); ego entry 0
equals the GAE lambda=1 return. Use ``--gae-lambda 0.95`` with it. Samples also carry training-only
aux targets (``ascore`` final score/20, ``arank``, ``awin``), ego-ordered and padded to 5; rollout
workers never evaluate aux heads. ``stats`` counts time-stone takes (``CLOCK_PAUSE``) for current
vs other seats.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bots.net_forward import NetWeights
from bus.actions import CLOCK_PAUSE
from train.env import BusEnv, potentials, shaped_return
from train.features import MAX_SEATS
from train.league import HOF_PREFIX, LeagueSpec, game_type, sample_assignment
from train.shm import make_spill_dir

END_TYPES = ("last_stone", "sites_full", "markers")
CURRENT = "current"
RANDOM = "random"
SNAP_PREFIX = "snap:"
Opponents = dict[str, Any]  # label -> NetWeights | npz path
_OPP_CACHE: dict[str, tuple[Any, NetWeights]] = {}  # per process: label -> (source, weights)
_OWNER_PID = 0  # trainer pid, set in pool workers (names the spill dirs)


@dataclass(frozen=True)
class RolloutConfig:
    players: tuple[int, ...] = (4,)
    gamma: float = 1.0
    lam: float = 0.95
    mix: float = 0.7
    envs_per_worker: int = 8
    greedy: bool = False
    collect: bool = True  # False: evaluation, no training samples
    league: LeagueSpec | None = None  # None: pure self-play
    eval_opponent: str = RANDOM  # label of the non-ego seats when ``egos`` is given
    player_weights: tuple[float, ...] | None = None  # sampling weights over ``players``
    record_scores: bool = False  # keep (labels, final scores) per game in ``stats["scores"]``
    reward: str = "terminal"  # "terminal" or "dense" (potential-based margin, unit 8e)
    ego_greedy: bool = False  # greedy "current" seats only, other nets still sample (AZ eval)


def assign_seat_policies(
    game_index: int, n: int, rng: np.random.Generator, league: LeagueSpec | None = None
) -> list[str]:
    """Policy label per seat: pure self-play, or drawn from the league spec."""
    if league is None:
        return [CURRENT] * n
    return sample_assignment(league, n, rng)


def _opponent_weights(label: str, opponents: Opponents) -> NetWeights:
    """Weights for a ``snap:``/``hof:`` label; npz paths load once per process (cached by label)."""
    src = opponents[label]
    if isinstance(src, NetWeights):
        return src
    hit = _OPP_CACHE.get(label)
    if hit is None or hit[0] != src:
        hit = (src, NetWeights.load(src))
        _OPP_CACHE[label] = hit
    return hit[1]


def _prune_cache(opponents: Opponents) -> None:
    for k in [k for k in _OPP_CACHE if k not in opponents or _OPP_CACHE[k][0] != opponents[k]]:
        del _OPP_CACHE[k]


def gae(
    rewards: np.ndarray[Any, Any], values: np.ndarray[Any, Any], gamma: float, lam: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """GAE for one finished trajectory (value after the last step is 0).

    Returns (advantages, returns = advantages + values).
    """
    n = len(rewards)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values.astype(np.float32)


def _dense_targets(
    phis: np.ndarray[Any, Any],
    phi_t: np.ndarray[Any, Any],
    wlr: list[float],
    seat: int,
    mix: float,
    vclip: list[int],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Dense rewards and return-to-go value targets for one seat's trajectory (unit 8e).

    ``phis`` (k, n): potentials of all seats at the state of each own decision; ``phi_t`` (n,) at
    the terminal state. r_t = (1-mix) (P[t+1] - P[t]) with P[k] = terminal, plus ``mix * win_loss``
    on the last step, so the sum telescopes to ``mix * wl + (1-mix) * (P_T - P_0)``. Value target
    for ego entry j at decision t: ``mix * wl_j + (1-mix) * (Phi_j(T) - Phi_j(t))``, clipped to
    [-1, 1] (tanh head); ``vclip`` accumulates [clipped, total] entries.
    """
    k, n = phis.shape
    own = np.concatenate([phis[:, seat], phi_t[seat : seat + 1]]).astype(np.float64)
    rew = ((1.0 - mix) * np.diff(own)).astype(np.float32)
    rew[-1] += np.float32(mix * wlr[seat])
    vt = np.zeros((k, MAX_SEATS), np.float32)
    for j in range(n):
        q = (seat + j) % n
        vt[:, j] = mix * wlr[q] + (1.0 - mix) * (phi_t[q] - phis[:, q])
    clipped = np.abs(vt[:, :n]) > 1.0
    vclip[0] += int(clipped.sum())
    vclip[1] += clipped.size
    return rew, np.clip(vt, -1.0, 1.0)


def aux_targets(
    core: Any, seat: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Ego-ordered aux targets for ``seat``, padded to MAX_SEATS: final score / 20, rank (0 =
    best, ties share the best rank, -1 padding) and the winner distribution (uniform over the
    R-069/R-070 winner set)."""
    from bus.rules import final_scores, winners

    n = core.players
    sc = final_scores(core)
    win = winners(core)
    a_sc = np.zeros(MAX_SEATS, np.float32)
    a_rk = np.full(MAX_SEATS, -1, np.int64)
    a_wn = np.zeros(MAX_SEATS, np.float32)
    for j in range(n):
        q = (seat + j) % n
        a_sc[j] = sc[q] / 20.0
        a_rk[j] = sum(1 for x in sc if x > sc[q])
        a_wn[j] = 1.0 / len(win) if q in win else 0.0
    return a_sc, a_rk, a_wn


class _Slot:
    __slots__ = ("env", "policies", "steps", "seed", "n", "decisions")

    def __init__(self, env: BusEnv, policies: list[str], seed: int) -> None:
        self.env = env
        self.policies = policies
        self.seed = seed
        self.n = env.num_players
        self.decisions = 0
        # per seat: obs, mask, action, logp, value, potentials of all seats at the decision
        self.steps: list[
            tuple[list[Any], list[Any], list[int], list[float], list[float], list[Any]]
        ] = [([], [], [], [], [], []) for _ in range(self.n)]


def _sample(
    weights: NetWeights, obs: np.ndarray[Any, Any], mask: np.ndarray[Any, Any],
    rng: np.random.Generator, greedy: bool, need_values: bool = True,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], int]:  # fmt: skip
    """(actions, logp, ego values (B,), illegal count) for a batch; values are zeros unless
    ``need_values`` (frozen opponents never use them, and a separate value trunk costs a pass)."""
    if need_values:
        logits, values = weights.forward(obs, mask)
        ego_v = values[:, 0].astype(np.float32)
    else:
        logits = weights.logits(obs, mask)
        ego_v = np.zeros(len(obs), np.float32)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z) * mask
    probs = e / e.sum(axis=-1, keepdims=True)
    if greedy:
        act = logits.argmax(axis=-1)
    else:
        u = rng.random(len(obs))[:, None]
        act = np.minimum((np.cumsum(probs, axis=-1) < u).sum(axis=-1), probs.shape[1] - 1)
    rows = np.arange(len(obs))
    illegal = 0
    bad = ~mask[rows, act]
    if bad.any():  # float edge: u above the cumulative sum; take the last legal action
        illegal = int(bad.sum())
        for i in np.flatnonzero(bad):
            act[i] = np.flatnonzero(mask[i])[-1]
    logp = np.log(np.maximum(probs[rows, act], 1e-30))
    return act, logp.astype(np.float32), ego_v, illegal


def _bot_action(label: str, env: BusEnv) -> int:
    """Action of the registry bot ``label`` (``heuristic``, ``mcts:50`` ...) in ``env``."""
    from bots.registry import make_bot  # lazy: pulls in the MCTS stack

    bot = make_bot(label)
    # MCTS works on clones itself in tournaments; the heuristic only reads the state.
    state = env.state if label == "heuristic" else env.state.clone()
    return int(bot.choose(state, env.rng))


def play_games(
    weights: NetWeights, n_games: int, seed: int, cfg: RolloutConfig,
    egos: list[int] | None = None, first_index: int = 0, opponents: Opponents | None = None,
    seatings: list[list[str]] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Play ``n_games`` full games; return samples (if ``cfg.collect``) and statistics.

    Deterministic for a given (weights, n_games, seed, cfg). ``egos`` (evaluation) pins which seat
    uses the net and gives all other seats ``cfg.eval_opponent``; ``first_index`` offsets the game
    index used for that rotation. ``opponents`` maps ``snap:<id>`` / ``hof:<name>`` labels to
    weights or npz paths. ``seatings`` (rating games) fixes the label of every seat of game i
    (``n_games`` must equal its length). Samples: obs, mask, action, logp, value, adv, vtarget,
    vmask, prog (own-decision index / count within the trajectory, for value EV by game stage).
    """
    t_start = time.perf_counter()
    opponents = opponents or {}
    _prune_cache(opponents)
    rng = np.random.default_rng(seed)
    pool: dict[int, list[BusEnv]] = {}
    out_lists: dict[str, list[Any]] = {
        k: []
        for k in (
            "obs",
            "mask",
            "action",
            "logp",
            "value",
            "adv",
            "vtarget",
            "vmask",
            "prog",
            "ascore",
            "arank",
            "awin",
        )
    }
    stats: dict[str, Any] = {
        "games": 0, "decisions": 0, "forced": 0, "illegal": 0, "lengths": [], "ego_returns": [],
        "end": dict.fromkeys(END_TYPES, 0), "mean_return0": 0.0,
        # game type -> [games, current-seat results, wins, summed shaped return]
        "by_type": {},
        # opponent label -> [games containing it, current seats in them, current wins]
        "by_label": {},
        "scores": [], "vclip": [0, 0],  # [clipped value-target entries, entries]
        "stones_cur": 0, "seats_cur": 0, "stones_oth": 0, "seats_oth": 0,  # time-stone takes
    }  # fmt: skip
    next_game = 0
    pw = None
    if cfg.player_weights is not None:
        pw = np.asarray(cfg.player_weights, float)
        pw = pw / pw.sum()
    slots: list[_Slot] = []

    def start_one() -> None:
        nonlocal next_game
        gi = next_game
        next_game += 1
        n = int(rng.choice(cfg.players, p=pw))
        if seatings is not None:
            n = len(seatings[gi])
        policies = assign_seat_policies(first_index + gi, n, rng, cfg.league)
        if seatings is not None:
            policies = list(seatings[gi])
        if egos is not None:
            ego = egos[gi] % n
            policies = [CURRENT if s == ego else cfg.eval_opponent for s in range(n)]
        free = pool.setdefault(n, [])
        env = free.pop() if free else BusEnv(n)
        gseed = int(rng.integers(0, 2**31 - 1))
        env.reset(seed=gseed)
        slots.append(_Slot(env, policies, gseed))

    def finish(sl: _Slot) -> None:
        env = sl.env
        n = sl.n
        rets = [shaped_return(env.state, p, cfg.mix) for p in range(n)]
        stats["games"] += 1
        stats["decisions"] += sl.decisions
        stats["forced"] += env.forced_steps
        stats["lengths"].append(sl.decisions)
        for t in env.state.core.end_types:
            if t in stats["end"]:
                stats["end"][t] += 1
        wl = env.state.core
        from bus.rules import returns_for  # local: keeps module import light

        wlr = returns_for(wl, "win_loss")
        acc = stats["by_type"].setdefault(game_type(sl.policies), [0, 0, 0, 0.0])
        acc[0] += 1
        for p in range(n):
            stats["seats_cur" if sl.policies[p] == CURRENT else "seats_oth"] += 1
            if sl.policies[p] == CURRENT:
                stats["ego_returns"].append((float(wlr[p]), n))
                acc[1] += 1
                acc[2] += 1 if wlr[p] > 0 else 0
                acc[3] += rets[p]
        cur_seats = [p for p in range(n) if sl.policies[p] == CURRENT]
        if cur_seats:
            cur_wins = sum(1 for p in cur_seats if wlr[p] > 0)
            for lab in {x for x in sl.policies if x != CURRENT}:
                bl = stats["by_label"].setdefault(lab, [0, 0, 0])
                bl[0] += 1
                bl[1] += len(cur_seats)
                bl[2] += cur_wins
        if cfg.record_scores:
            from bus.rules import final_scores

            stats["scores"].append((list(sl.policies), [float(x) for x in final_scores(wl)]))
        if cfg.collect:
            for seat in range(n):
                if sl.policies[seat] != CURRENT:
                    continue
                o, m, a, lp, v, ph = sl.steps[seat]
                if not a:
                    continue
                k = len(a)
                rew = np.zeros(k, np.float32)
                val = np.asarray(v, np.float32)
                tm = np.zeros(MAX_SEATS, np.float32)
                tm[:n] = 1.0
                if cfg.reward == "dense":
                    rew, vt = _dense_targets(
                        np.stack(ph), potentials(wl), wlr, seat, cfg.mix, stats["vclip"]
                    )
                    adv, _ = gae(rew, val, cfg.gamma, cfg.lam)
                else:
                    rew[-1] = rets[seat]
                    adv, _ = gae(rew, val, cfg.gamma, cfg.lam)
                    tv = np.zeros(MAX_SEATS, np.float32)
                    for j in range(n):
                        tv[j] = rets[(seat + j) % n]
                    vt = np.tile(tv, (k, 1))
                a_sc, a_rk, a_wn = aux_targets(wl, seat)
                out_lists["obs"].append(np.stack(o))
                out_lists["mask"].append(np.stack(m))
                out_lists["action"].append(np.asarray(a, np.int64))
                out_lists["logp"].append(np.asarray(lp, np.float32))
                out_lists["value"].append(val)
                out_lists["adv"].append(adv)
                out_lists["vtarget"].append(vt)
                out_lists["ascore"].append(np.tile(a_sc, (k, 1)))
                out_lists["arank"].append(np.tile(a_rk, (k, 1)))
                out_lists["awin"].append(np.tile(a_wn, (k, 1)))
                out_lists["vmask"].append(np.tile(tm, (k, 1)))
                out_lists["prog"].append(((np.arange(k) + 0.5) / k).astype(np.float32))
        pool.setdefault(n, []).append(env)

    while next_game < n_games or slots:
        while next_game < n_games and len(slots) < cfg.envs_per_worker:
            start_one()
        cur = [sl for sl in slots if sl.env.current_seat >= 0]
        actions = [0] * len(cur)
        groups: dict[str, list[int]] = {}
        for i, sl in enumerate(cur):
            label = sl.policies[sl.env.current_seat]
            if label == CURRENT or label.startswith((SNAP_PREFIX, HOF_PREFIX)):
                groups.setdefault(label, []).append(i)
            elif label == RANDOM:
                actions[i] = int(rng.choice(sl.env.legal_actions()))
            else:  # registry bot on the real state (they do not mutate it)
                actions[i] = _bot_action(label, sl.env)
        for label, net_idx in groups.items():
            w = weights if label == CURRENT else _opponent_weights(label, opponents)
            obs = np.stack([cur[i].env.obs() for i in net_idx])
            mask = np.stack([cur[i].env.legal_mask() for i in net_idx])
            act, logp, val, ill = _sample(
                w,
                obs,
                mask,
                rng,
                cfg.greedy or (cfg.ego_greedy and label == CURRENT),
                need_values=cfg.collect and label == CURRENT,
            )
            stats["illegal"] += ill
            for r, i in enumerate(net_idx):
                sl = cur[i]
                actions[i] = int(act[r])
                if cfg.collect and label == CURRENT:
                    o, m, a, lp, v, ph = sl.steps[sl.env.current_seat]
                    if cfg.reward == "dense":
                        ph.append(potentials(sl.env.state.core))
                    o.append(obs[r])
                    m.append(mask[r])
                    a.append(int(act[r]))
                    lp.append(float(logp[r]))
                    v.append(float(val[r]))
        finished: list[_Slot] = []
        for sl, a_ in zip(cur, actions, strict=True):
            if a_ == CLOCK_PAUSE:  # taking a time stone (R-050..R-053)
                stats[
                    "stones_cur" if sl.policies[sl.env.current_seat] == CURRENT else "stones_oth"
                ] += 1
            sl.env.step(a_)
            sl.decisions += 1
            if sl.env.done:
                finished.append(sl)
        for sl in finished:
            slots.remove(sl)
            finish(sl)

    res: dict[str, Any] = {"stats": stats, "seconds": time.perf_counter() - t_start}
    if cfg.collect and out_lists["obs"]:
        res["samples"] = {k: np.concatenate(v) for k, v in out_lists.items()}
    else:
        res["samples"] = None
    return res


Task = tuple[Any, ...]  # (weights, n_games, seed, cfg, egos, first_index, opponents[, seatings])


def run_task(task: Task) -> dict[str, Any]:
    """Picklable pool entry point."""
    weights, n_games, seed, cfg, egos, first_index, opponents = task[:7]
    seatings = task[7] if len(task) > 7 else None
    return play_games(weights, n_games, seed, cfg, egos, first_index, opponents, seatings)


def run_task_spill(task: Task) -> dict[str, Any]:
    """``run_task`` for pool workers: samples go through tmpfs files, not the result pipe.

    Pickling ~1 GB of float32 observations per update through ``Pool``'s pipe took longer than
    the rollout itself and serialised on the parent (unit 3b measurements). The parent reads the
    files back with ``load_spilled``.
    """
    kill_flag = os.environ.get("BUS_TRAIN_KILL_ONCE")  # fault injection for the recovery test
    if kill_flag and not os.path.exists(kill_flag):
        open(kill_flag, "w").close()
        os._exit(9)
    res = run_task(task)
    samples = res["samples"]
    if samples is not None:
        d = make_spill_dir(_OWNER_PID or os.getppid())
        for k, v in samples.items():
            np.save(os.path.join(d, k + ".npy"), v)
        res["samples"] = {"__spill__": d}
    return res


def load_spilled(res: dict[str, Any]) -> dict[str, Any]:
    """Inverse of ``run_task_spill`` (no-op for results that were not spilled); deletes files."""
    samples = res["samples"]
    if isinstance(samples, dict) and "__spill__" in samples:
        d = samples["__spill__"]
        try:
            res["samples"] = {f[:-4]: np.load(os.path.join(d, f)) for f in sorted(os.listdir(d))}
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return res


def worker_init(owner_pid: int = 0) -> None:
    """Pool worker setup: single-threaded BLAS, ignore terminal signals, remember the trainer pid.

    Ctrl-C reaches the whole foreground process group; the trainer alone decides how to stop, so
    workers ignore SIGINT/SIGTERM (the trainer kills them when it needs to).
    """
    global _OWNER_PID
    _OWNER_PID = owner_pid
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
