# pyright: basic, reportMissingImports=false
"""Rollout actors for PPO self-play (Phase 4 unit 3). Numpy only: never imports torch.

A worker plays full games through ``BusEnv`` with a numpy copy of the network
(``bots.net_forward.NetWeights``), batching the forward pass over several concurrent games.

Seat policies: each game has one policy label per seat. ``"current"`` seats sample from the net
being trained and are the only seats that produce training samples; ``"random"`` seats play uniform
random legal actions (used by evaluation). ``assign_seat_policies`` is the hook the opponent league
(unit 4) will replace; for now it returns pure self-play.

Rewards and targets. Terminal-only: a seat's reward is 0 at every decision except its last, which
gets the seat's final shaped return (``mix * win_loss + (1 - mix) * margin``). GAE is computed per
seat trajectory (only that seat's own decisions) with the ego value entry (index 0 of the
ego-ordered value vector, i.e. the value for the acting seat) as baseline; the value after the last
decision is 0.
The 5-wide value head is trained on the *undiscounted* final shaped returns of all present seats in
ego order for that trajectory's seat (``vtarget`` / ``vmask``), so entry 0 has the same target as
the seat's own Monte Carlo return, and the other entries are auxiliary.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bots.net_forward import NetWeights
from train.env import BusEnv, shaped_return
from train.features import MAX_SEATS

END_TYPES = ("last_stone", "sites_full", "markers")
CURRENT = "current"
RANDOM = "random"


@dataclass(frozen=True)
class RolloutConfig:
    players: tuple[int, ...] = (4,)
    gamma: float = 1.0
    lam: float = 0.95
    mix: float = 0.7
    envs_per_worker: int = 8
    greedy: bool = False
    collect: bool = True  # False: evaluation, no training samples


def assign_seat_policies(game_index: int, n: int, rng: np.random.Generator) -> list[str]:
    """Policy label per seat. Pure self-play now; unit 4 mixes in snapshots/heuristic/random."""
    return [CURRENT] * n


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


class _Slot:
    __slots__ = ("env", "policies", "steps", "seed", "n", "decisions")

    def __init__(self, env: BusEnv, policies: list[str], seed: int) -> None:
        self.env = env
        self.policies = policies
        self.seed = seed
        self.n = env.num_players
        self.decisions = 0
        # per seat: obs, mask, action, logp, value
        self.steps: list[tuple[list[Any], list[Any], list[int], list[float], list[float]]] = [
            ([], [], [], [], []) for _ in range(self.n)
        ]


def _sample(
    weights: NetWeights, obs: np.ndarray[Any, Any], mask: np.ndarray[Any, Any],
    rng: np.random.Generator, greedy: bool,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], int]:  # fmt: skip
    """(actions, logp, ego values (B,), illegal count) for a batch."""
    logits, values = weights.forward(obs, mask)
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
    return act, logp.astype(np.float32), values[:, 0].astype(np.float32), illegal


def play_games(
    weights: NetWeights, n_games: int, seed: int, cfg: RolloutConfig,
    egos: list[int] | None = None, first_index: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """Play ``n_games`` full games; return samples (if ``cfg.collect``) and statistics.

    Deterministic for a given (weights, n_games, seed, cfg). ``egos`` (evaluation) pins which seat
    uses the net and makes all other seats random; ``first_index`` offsets the game index used for
    that rotation. Samples: obs, mask, action, logp, value, adv, vtarget, vmask.
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    pool: dict[int, list[BusEnv]] = {}
    out_lists: dict[str, list[Any]] = {
        k: [] for k in ("obs", "mask", "action", "logp", "value", "adv", "vtarget", "vmask")
    }
    stats: dict[str, Any] = {
        "games": 0, "decisions": 0, "forced": 0, "illegal": 0, "lengths": [], "ego_returns": [],
        "end": dict.fromkeys(END_TYPES, 0), "mean_return0": 0.0,
    }  # fmt: skip
    next_game = 0
    slots: list[_Slot] = []

    def start_one() -> None:
        nonlocal next_game
        gi = next_game
        next_game += 1
        n = int(rng.choice(cfg.players))
        policies = assign_seat_policies(first_index + gi, n, rng)
        if egos is not None:
            ego = egos[gi] % n
            policies = [CURRENT if s == ego else RANDOM for s in range(n)]
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
        for p in range(n):
            if sl.policies[p] == CURRENT:
                stats["ego_returns"].append((float(wlr[p]), n))
        if cfg.collect:
            for seat in range(n):
                if sl.policies[seat] != CURRENT:
                    continue
                o, m, a, lp, v = sl.steps[seat]
                if not a:
                    continue
                k = len(a)
                rew = np.zeros(k, np.float32)
                rew[-1] = rets[seat]
                val = np.asarray(v, np.float32)
                adv, _ = gae(rew, val, cfg.gamma, cfg.lam)
                tv = np.zeros(MAX_SEATS, np.float32)
                tm = np.zeros(MAX_SEATS, np.float32)
                for j in range(n):
                    tv[j] = rets[(seat + j) % n]
                    tm[j] = 1.0
                out_lists["obs"].append(np.stack(o))
                out_lists["mask"].append(np.stack(m))
                out_lists["action"].append(np.asarray(a, np.int64))
                out_lists["logp"].append(np.asarray(lp, np.float32))
                out_lists["value"].append(val)
                out_lists["adv"].append(adv)
                out_lists["vtarget"].append(np.tile(tv, (k, 1)))
                out_lists["vmask"].append(np.tile(tm, (k, 1)))
        pool.setdefault(n, []).append(env)

    while next_game < n_games or slots:
        while next_game < n_games and len(slots) < cfg.envs_per_worker:
            start_one()
        cur = [sl for sl in slots if sl.env.current_seat >= 0]
        net_idx = [i for i, sl in enumerate(cur) if sl.policies[sl.env.current_seat] == CURRENT]
        actions = [0] * len(cur)
        if net_idx:
            obs = np.stack([cur[i].env.obs() for i in net_idx])
            mask = np.stack([cur[i].env.legal_mask() for i in net_idx])
            act, logp, val, ill = _sample(weights, obs, mask, rng, cfg.greedy)
            stats["illegal"] += ill
            for r, i in enumerate(net_idx):
                sl = cur[i]
                actions[i] = int(act[r])
                if cfg.collect:
                    o, m, a, lp, v = sl.steps[sl.env.current_seat]
                    o.append(obs[r])
                    m.append(mask[r])
                    a.append(int(act[r]))
                    lp.append(float(logp[r]))
                    v.append(float(val[r]))
        for i, sl in enumerate(cur):
            if sl.policies[sl.env.current_seat] != CURRENT:
                actions[i] = int(rng.choice(sl.env.legal_actions()))
        finished: list[_Slot] = []
        for sl, a_ in zip(cur, actions, strict=True):
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


def run_task(
    task: tuple[NetWeights, int, int, RolloutConfig, list[int] | None, int],
) -> dict[str, Any]:
    """Picklable pool entry point."""
    weights, n_games, seed, cfg, egos, first_index = task
    return play_games(weights, n_games, seed, cfg, egos, first_index)


def _spill_dir() -> str:
    """tmpfs when it has room (samples can be hundreds of MB per update), else the system temp."""
    shm = "/dev/shm"
    if os.path.isdir(shm) and shutil.disk_usage(shm).free > 1_500_000_000:
        return shm
    return tempfile.gettempdir()


def run_task_spill(
    task: tuple[NetWeights, int, int, RolloutConfig, list[int] | None, int],
) -> dict[str, Any]:
    """``run_task`` for pool workers: samples go through tmpfs files, not the result pipe.

    Pickling ~1 GB of float32 observations per update through ``Pool``'s pipe took longer than
    the rollout itself and serialised on the parent (unit 3b measurements). The parent reads the
    files back with ``load_spilled``.
    """
    res = run_task(task)
    samples = res["samples"]
    if samples is not None:
        d = tempfile.mkdtemp(prefix="bus_rollout_", dir=_spill_dir())
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


def worker_init() -> None:
    """Keep each worker single-threaded (set before numpy loads only for spawned children)."""
    import os

    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
