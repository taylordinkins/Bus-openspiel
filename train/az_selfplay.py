# pyright: basic, reportMissingImports=false
"""Gumbel AlphaZero actors (Phase 4 unit 8h). Numpy only: never imports torch.

A task plays ``n_games`` full games in one process with B (``concurrent``) games in flight.
Learner seats ("current") choose every non-forced move with a ``bots.gumbel.GumbelSearch`` on the
current weights (Gumbel noise scale 1, so the root's Gumbel-top-k is the exploration); one
batched forward per wave serves the pending leaf of every open search in the task. Frozen seats
(``hof:<name>`` / ``snap:<gen>`` labels, weights or npz paths in ``opponents``) play their raw
policy sampled at temperature 1, batched per label, and produce no samples.

Tables: with probability ``frozen_share`` a game seats frozen nets in 1..n-1 random seats (one
label drawn uniformly from ``opponents`` per game); otherwise all seats are the learner.

Samples (one per learner decision with >= 2 legal actions), sparse:
- ``obs_off`` / ``obs_idx`` (int16) / ``obs_code`` (uint8, uint16 if needed) / ``obs_tab``
  (float32): non-zero observation entries, values coded through the task's table of distinct
  values (lossless; observations have only a few dozen distinct values);
- ``act_off`` / ``act_id`` (int16) / ``pi`` (float32): root legal actions and the completed-Q
  improved policy over them (the policy target, sums to 1);
- ``z`` (5,): final per-seat R = mix * win_loss + (1 - mix) * margin (R-071, R-072), ego order
  (index 0 = the acting seat), padded with 0; ``vmask`` (5,): present seats;
- ``q`` (5,): the root search value vector in the same ego order (for the optional zq target);
- ``ascore`` / ``arank`` / ``awin``: training-only aux targets (``train.rollout.aux_targets``);
- ``prog``: own-decision index / count within the seat's game (for value EV by quartile);
- ``players``.

Stats: games, decisions (learner, non-forced), sims, wall seconds, R2 diagnostics (how often the
played search action differs from the prior's argmax, mean improved-policy entropy, mean
KL(pi' || prior)), and per-table results for the learner seats with shared wins credited 1/k.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bots.gumbel import (
    GumbelConfig,
    GumbelSearch,
    evaluate_batch,
    skip_forced,
    sparse_forward,
    terminal_values,
)
from bots.net_forward import NetWeights
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS
from train.features import MAX_SEATS, encode
from train.rollout import aux_targets
from train.shm import make_spill_dir

CURRENT = "current"
Array = np.ndarray[Any, Any]
SAMPLE_KEYS = (
    "obs_off", "obs_idx", "obs_code", "obs_tab", "act_off", "act_id", "pi", "z", "vmask", "q",
    "ascore", "arank", "awin", "prog", "players",
)  # fmt: skip
_OPP_CACHE: dict[str, tuple[Any, NetWeights]] = {}


@dataclass(frozen=True)
class ActorConfig:
    players: tuple[int, ...] = (4,)
    player_weights: tuple[float, ...] | None = None
    sims: int = 32
    search: GumbelConfig = GumbelConfig(c_scale=0.3, gumbel_scale=1.0)
    concurrent: int = 32  # games in flight per task
    frozen_share: float = 0.2  # share of tables with frozen seats (needs ``opponents``)


def win_credit(wl: float, n: int) -> float:
    """Win credit of a seat from its R-071 win_loss return: 1/k for a k-way shared win, else 0.

    A winner among k gets (n - k) / (k (n - 1)) and a loser -1/(n - 1); k = n gives 0 to all.
    """
    if wl < -1e-9:
        return 0.0
    return (wl * (n - 1) + 1.0) / n


def _weights(label: str, opponents: dict[str, Any]) -> NetWeights:
    src = opponents[label]
    if isinstance(src, NetWeights):
        return src
    hit = _OPP_CACHE.get(label)
    if hit is None or hit[0] != src:
        hit = (src, NetWeights.load(src))
        _OPP_CACHE[label] = hit
    return hit[1]


def _code_values(vals: Array) -> dict[str, Array]:
    tab, code = np.unique(vals, return_inverse=True)
    dtype = np.uint8 if len(tab) <= 256 else np.uint16
    return {"obs_code": code.astype(dtype), "obs_tab": tab.astype(np.float32)}


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class _Game:
    __slots__ = ("core", "legal", "labels", "search", "steps", "n", "frozen")

    def __init__(self, core: Any, labels: list[str]) -> None:
        self.core, self.legal = skip_forced(core)
        self.labels = labels
        self.n = core.players
        self.frozen = any(lab != CURRENT for lab in labels)
        self.search: GumbelSearch | None = None
        # per seat: list of (obs, actions, pi, q_abs)
        self.steps: list[list[tuple[Array, Array, Array, Array]]] = [[] for _ in range(self.n)]


def play_selfplay(
    weights: NetWeights, n_games: int, seed: int, cfg: ActorConfig,
    opponents: dict[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Play ``n_games`` games; deterministic for given (weights, n_games, seed, cfg, opponents)."""
    t0 = time.perf_counter()
    opponents = opponents or {}
    for k in [k for k in _OPP_CACHE if k not in opponents]:  # evicted snapshots
        del _OPP_CACHE[k]
    rng = np.random.default_rng(seed)
    fwd = sparse_forward(weights)
    pw = None
    if cfg.player_weights is not None:
        pw = np.asarray(cfg.player_weights, float)
        pw = pw / pw.sum()
    labels_all = sorted(opponents)
    out: dict[str, list[Any]] = {k: [] for k in (*SAMPLE_KEYS, "obs_val")}
    obs_n = act_n = 0
    stats: dict[str, Any] = {
        "games": 0, "decisions": 0, "sims": 0, "forwards": 0, "evals": 0, "differs": 0,
        "entropy": 0.0, "kl": 0.0, "lengths": [],
        # table kind ("self" / "frozen") -> [learner seats, win credit, summed R]
        "tables": {}, "seconds": 0.0,
    }  # fmt: skip
    started = 0
    live: list[_Game] = []

    def start() -> None:
        nonlocal started
        started += 1
        n = int(rng.choice(cfg.players, p=pw))
        labels = [CURRENT] * n
        if labels_all and rng.random() < cfg.frozen_share:
            lab = labels_all[int(rng.integers(len(labels_all)))]
            k = int(rng.integers(1, n))
            for s in rng.choice(n, size=k, replace=False):
                labels[int(s)] = lab
        live.append(_Game(rules.initial_core(n), labels))

    def finish(g: _Game) -> None:
        nonlocal obs_n, act_n
        core = g.core
        n = g.n
        r_abs = terminal_values(core, cfg.search.mix)
        wl = rules.returns_for(core, "win_loss")
        stats["games"] += 1
        kind = "frozen" if g.frozen else "self"
        tab = stats["tables"].setdefault(kind, [0, 0.0, 0.0])
        for seat in range(n):
            if g.labels[seat] != CURRENT:
                continue
            tab[0] += 1
            tab[1] += win_credit(wl[seat], n)
            tab[2] += float(r_abs[seat])
            steps = g.steps[seat]
            if not steps:
                continue
            k = len(steps)
            z = np.zeros(MAX_SEATS, np.float32)
            vm = np.zeros(MAX_SEATS, np.float32)
            for j in range(n):
                z[j] = r_abs[(seat + j) % n]
            vm[:n] = 1.0
            a_sc, a_rk, a_wn = aux_targets(core, seat)
            for t, (o, acts, pi, q_abs) in enumerate(steps):
                nz = np.flatnonzero(o)
                out["obs_off"].append(obs_n)
                out["obs_idx"].append(nz.astype(np.int16))
                out["obs_val"].append(o[nz].astype(np.float32))
                obs_n += len(nz)
                out["act_off"].append(act_n)
                out["act_id"].append(acts.astype(np.int16))
                out["pi"].append(pi.astype(np.float32))
                act_n += len(acts)
                q = np.zeros(MAX_SEATS, np.float32)
                for j in range(n):
                    q[j] = q_abs[(seat + j) % n]
                out["z"].append(z)
                out["vmask"].append(vm)
                out["q"].append(q)
                out["ascore"].append(a_sc)
                out["arank"].append(a_rk.astype(np.int8))
                out["awin"].append(a_wn)
                out["prog"].append((t + 0.5) / k)
                out["players"].append(n)

    while started < n_games or live:
        while started < n_games and len(live) < cfg.concurrent:
            start()
        # frozen seats: one sampled move each, batched per label
        groups: dict[str, list[_Game]] = {}
        for g in live:
            if g.legal and g.labels[g.core.current] != CURRENT:
                groups.setdefault(g.labels[g.core.current], []).append(g)
        for lab, gs in groups.items():
            obs = np.stack([encode(g.core) for g in gs]).astype(np.float32)
            mask = np.zeros((len(gs), NUM_DISTINCT_ACTIONS), np.bool_)
            for r, g in enumerate(gs):
                mask[r, g.legal] = True
            logits = _weights(lab, opponents).logits(obs, mask)
            for r, g in enumerate(gs):
                lg = logits[r, g.legal].astype(np.float64)
                p = np.exp(lg - lg.max())
                a = g.legal[int(rng.choice(len(g.legal), p=p / p.sum()))]
                g.core, g.legal = skip_forced(rules.apply(g.core, int(a)))
        # learner seats: one search wave
        pending: list[GumbelSearch] = []
        for g in live:
            while g.legal and g.labels[g.core.current] == CURRENT:
                if g.search is None:
                    g.search = GumbelSearch(g.core, cfg.sims, cfg.search, rng)
                s = g.search
                if s.want() is not None:
                    pending.append(s)
                    break
                res = s.result()
                stats["decisions"] += 1
                stats["sims"] += s.done_sims
                prior = s.root.prior
                stats["differs"] += int(res.action != int(s.root.actions[int(np.argmax(prior))]))
                pi = res.policy
                logpi = np.log(np.maximum(pi, 1e-30))
                stats["entropy"] += float(-(pi * logpi).sum())
                stats["kl"] += float((pi * (logpi - np.log(np.maximum(prior, 1e-30)))).sum())
                o = encode(g.core)
                g.steps[g.core.current].append((o, res.actions, pi, res.value))
                g.core, g.legal = skip_forced(rules.apply(g.core, res.action))
                g.search = None
        if pending:
            evaluate_batch(fwd, pending)
            stats["forwards"] += 1
            stats["evals"] += len(pending)
        done = [g for g in live if not g.legal]
        for g in done:
            live.remove(g)
            finish(g)
    stats["seconds"] = time.perf_counter() - t0
    stats["rss_mb"] = _rss_mb()
    samples = None
    if out["z"]:
        samples = {
            "obs_off": np.asarray(out["obs_off"], np.int64),
            "obs_idx": np.concatenate(out["obs_idx"]),
            **_code_values(np.concatenate(out["obs_val"])),
            "act_off": np.asarray(out["act_off"], np.int64),
            "act_id": np.concatenate(out["act_id"]),
            "pi": np.concatenate(out["pi"]),
            "z": np.stack(out["z"]),
            "vmask": np.stack(out["vmask"]),
            "q": np.stack(out["q"]),
            "ascore": np.stack(out["ascore"]),
            "arank": np.stack(out["arank"]),
            "awin": np.stack(out["awin"]),
            "prog": np.asarray(out["prog"], np.float32),
            "players": np.asarray(out["players"], np.int8),
        }
    return {"stats": stats, "samples": samples}


# ---------------------------------------------------------------- pool entry points

# Task: ("selfplay", weights, n_games, seed, ActorConfig, opponents)
#     | ("eval", rollout task tuple for train.rollout.run_task)


def run_az_task(task: tuple[Any, ...]) -> dict[str, Any]:
    if task[0] == "eval":
        from train.rollout import run_task

        return run_task(task[1])
    _, weights, n_games, seed, cfg, opponents = task
    return play_selfplay(weights, n_games, seed, cfg, opponents)


def run_az_task_spill(task: tuple[Any, ...]) -> dict[str, Any]:
    """``run_az_task`` for pool workers: samples go through tmpfs files (``load_spilled``)."""
    res = run_az_task(task)
    samples = res.get("samples")
    if samples is not None:
        from train import rollout

        d = make_spill_dir(rollout._OWNER_PID or os.getppid())
        try:
            for k, v in samples.items():
                np.save(os.path.join(d, k + ".npy"), v)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        res["samples"] = {"__spill__": d}
    return res
