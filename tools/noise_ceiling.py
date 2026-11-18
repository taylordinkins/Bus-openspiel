# pyright: basic
"""Noise ceiling on value prediction (Phase 4, after unit 8i).

How much of the variance of a single game outcome can any value function explain from a
position, given the policies that play the rest of the game? For a position s and a seat, the
outcome R is random only through policy sampling (Bus is deterministic, R-073), so the
Bayes-optimal value is E[R | s] and the best explained variance against single outcomes is

    EV_max = Var_s(E[R|s]) / Var(R) = 1 - E_s[Var(R|s)] / Var(R).

Protocol:

1. **Positions.** Self-play games of the sampling net (τ = 1, all seats); decision points with
   more than one legal action (as ``train.env``), stratified by the mover's game progress
   ``prog = (t + 0.5) / k`` over its own decisions (the trainer's quartile key), at most
   ``--per-game`` positions per game, in distinct quartiles.
2. **Continuations.** From each position, K independent continuations to the end by the
   continuation policy in all seats (``net``: the net sampled at τ = 1; ``heuristic``: the
   heuristic bot with seeded tie-breaks).
3. **Outcomes** per seat: R = 0.5 win_loss + 0.5 margin (``bots.gumbel.terminal_values``, the AZ
   value target), the variants R_leader and R_rank with the leader-relative and rank terms in
   place of the mean-opponent margin (``bots.gumbel.relative_term``), and win_loss (R-071; shared
   wins split, so the 1/k win credit is an affine map of it and has the same EVs).
4. **Estimator** (one-way ANOVA with K replicates): within = mean_s of the unbiased (K - 1)
   variance; between = Var_s(K-means) - within / K; EV_max = between / (between + within).
   Bootstrap 95% CIs resample sampling games (clusters), so positions from one game stay together.
5. **Nets.** Each net's value at each position (``NetWeights.forward``, tanh head, 5-wide ego
   vector rotated to absolute seats): EV against single outcomes (pooled over the K replicates,
   the quantity the trainers log) and R² against E[R|s] (K-mean, finite-K corrected).

Numpy-only workers (spawn pool, one BLAS thread each); forwards batched over all live games of a
task. Writes ``positions.npz`` (raw), ``summary.json`` and ``summary.md`` to ``--out``.

    PYTHONUNBUFFERED=1 uv run python -m tools.noise_ceiling --positions 400 --k 32 --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

from bots.gumbel import skip_forced, terminal_values  # noqa: E402
from bots.net_forward import NetWeights  # noqa: E402
from bus import rules  # noqa: E402
from bus.actions import NUM_DISTINCT_ACTIONS  # noqa: E402
from train.features import MAX_SEATS, encode  # noqa: E402

Array = np.ndarray[Any, Any]
QUANTITIES = ("R", "R_leader", "R_rank", "wl")
TARGET_OF = {"R": "mean", "R_leader": "leader", "R_rank": "rank"}  # quantity -> relative term
DEFAULT_NETS = "ppo-long=models/ppo-long.npz,az2=runs/az2/best.npz,az1=runs/az1/best.npz"

# ------------------------------------------------------------------ estimators


def ceiling(r: Array) -> dict[str, float]:
    """ANOVA ceiling of ``r`` (units, K): within, finite-K corrected between, EV_max."""
    k = r.shape[1]
    within = float(r.var(axis=1, ddof=1).mean())
    between = float(r.mean(axis=1).var(ddof=1)) - within / k
    total = between + within
    return {
        "within": within,
        "between": between,
        "total": total,
        "ev_max": between / total if total > 1e-12 else float("nan"),
    }


def net_ev(pred: Array, r: Array) -> dict[str, float]:
    """EV of ``pred`` (units,) against single outcomes ``r`` (units, K), and its R² against the
    conditional mean E[R|s] (K-means with the finite-K noise removed). Centered, like
    ``train.ppo.explained_variance``."""
    k = r.shape[1]
    within = float(r.var(axis=1, ddof=1).mean())
    resid = r - pred[:, None]
    var = float(r.var())
    ev_single = 1.0 - float(resid.var()) / var if var > 1e-12 else float("nan")
    m = r.mean(axis=1)
    num = float((m - pred).var(ddof=1)) - within / k
    den = float(m.var(ddof=1)) - within / k
    return {
        "ev_single": ev_single,
        "r2_mean": 1.0 - num / den if den > 1e-12 else float("nan"),
    }


def bootstrap(
    fn: Any, groups: Array, args: Sequence[Array], reps: int, rng: np.random.Generator
) -> dict[str, list[float]]:
    """95% percentile CIs of each key of ``fn(*args)``, resampling whole groups (clusters)."""
    ug = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ug]
    draws: dict[str, list[float]] = {}
    for _ in range(reps):
        pick = rng.integers(0, len(ug), size=len(ug))
        idx = np.concatenate([members[i] for i in pick])
        for key, v in fn(*(a[idx] for a in args)).items():
            draws.setdefault(key, []).append(v)
    with warnings.catch_warnings():  # all-NaN draws (no variance) give NaN bounds
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for key, v in draws.items()
        }


# ------------------------------------------------------------------ play (workers)

_W: dict[str, NetWeights] = {}


def _net(path: str) -> NetWeights:
    if path not in _W:
        _W[path] = NetWeights.load(path)
    return _W[path]


class _Live:
    __slots__ = ("core", "legal", "rng", "trace", "state", "py_rng")

    def __init__(self, core: Any, rng: np.random.Generator, trace: bool) -> None:
        self.core, self.legal = skip_forced(core)
        self.rng = rng
        self.trace: list[tuple[Any, int]] | None = [] if trace else None
        self.state: Any = None
        self.py_rng: random.Random | None = None


def play(
    cores: Sequence[Any], rngs: Sequence[np.random.Generator], policy: str, net_path: str,
    trace: bool = False,
) -> list[_Live]:  # fmt: skip
    """Play every core to the end. ``policy`` "net": sample ``net_path`` at τ = 1, one batched
    forward per step over all live games; "heuristic": the heuristic bot in every seat."""
    games = [_Live(c, g, trace) for c, g in zip(cores, rngs, strict=True)]
    live = [g for g in games if g.legal]
    if policy == "net":
        w = _net(net_path)
        while live:
            if trace:
                for g in live:
                    g.trace.append((g.core, int(g.core.current)))  # type: ignore[union-attr]
            obs = np.stack([encode(g.core) for g in live]).astype(np.float32)
            mask = np.zeros((len(live), NUM_DISTINCT_ACTIONS), np.bool_)
            for i, g in enumerate(live):
                mask[i, g.legal] = True
            logits = w.logits(obs, mask)
            for i, g in enumerate(live):
                lg = logits[i, g.legal].astype(np.float64)
                p = np.exp(lg - lg.max())
                a = g.legal[int(g.rng.choice(len(g.legal), p=p / p.sum()))]
                g.core, g.legal = skip_forced(rules.apply(g.core, int(a)))
            live = [g for g in live if g.legal]
    elif policy == "heuristic":
        import pyspiel

        import bus.game  # noqa: F401  (registers python_bus)
        from bots.registry import make_bot

        bot = make_bot("heuristic")
        loaded: dict[int, Any] = {}
        for g in live:
            n = int(g.core.players)
            if n not in loaded:
                loaded[n] = pyspiel.load_game("python_bus", {"players": n})
            g.state = loaded[n].new_initial_state()
            g.py_rng = random.Random(int(g.rng.integers(2**62)))
            while g.legal:
                if trace:
                    g.trace.append((g.core, int(g.core.current)))  # type: ignore[union-attr]
                g.state.core = g.core
                a = int(bot.choose(g.state, g.py_rng))
                g.core, g.legal = skip_forced(rules.apply(g.core, a))
            g.state = None
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return games


def sample_task(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Play sampling games; pick one position per requested quartile (random decision of any seat
    whose prog falls in it)."""
    seed, game_ids, quartiles, net_path = task
    rngs = [np.random.default_rng([seed, 0, gid]) for gid in game_ids]
    games = play([rules.initial_core(4) for _ in game_ids], rngs, "net", net_path, trace=True)
    out: list[dict[str, Any]] = []
    for gid, g, qs in zip(game_ids, games, quartiles, strict=True):
        trace = g.trace or []
        n = int(g.core.players)
        counts = [sum(1 for _, s in trace if s == p) for p in range(n)]
        seen = [0] * n
        progs = []
        for _, s in trace:
            progs.append((seen[s] + 0.5) / counts[s])
            seen[s] += 1
        progs_a = np.asarray(progs)
        pick_rng = np.random.default_rng([seed, 2, gid])
        for q in qs:
            cand = np.flatnonzero((progs_a >= q / 4) & (progs_a < (q + 1) / 4))
            i = int(cand[pick_rng.integers(len(cand))])
            core, mover = trace[i]
            out.append({"game": gid, "quartile": q, "prog": float(progs_a[i]), "core": core,
                        "mover": mover, "decision": i, "length": len(trace)})  # fmt: skip
    return out


def continue_task(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    """K continuations of each position; per-seat R variants and win_loss (absolute seats, padded
    NaN)."""
    seed, items, k, policy, net_path, mix = task
    cores, rngs, owner = [], [], []
    for pid, core in items:
        for j in range(k):
            cores.append(core)
            rngs.append(np.random.default_rng([seed, 1, pid, j]))
            owner.append(pid)
    games = play(cores, rngs, policy, net_path)
    res: dict[int, dict[str, Any]] = {}
    for pid, _core in items:
        res[pid] = {"pid": pid, "_j": 0, **{q: np.full((MAX_SEATS, k), np.nan) for q in QUANTITIES}}
    for pid, g in zip(owner, games, strict=True):
        d = res[pid]
        n = int(g.core.players)
        j = d["_j"]
        for q, target in TARGET_OF.items():
            d[q][:n, j] = terminal_values(g.core, mix, target)
        d["wl"][:n, j] = rules.returns_for(g.core, "win_loss")
        d["_j"] = j + 1
    for d in res.values():
        del d["_j"]
    return list(res.values())


# ------------------------------------------------------------------ analysis


def net_values(path: str, cores: Sequence[Any], movers: Array) -> Array:
    """(P, MAX_SEATS) net values in absolute seat order (NaN padding), batched dense forward."""
    w = NetWeights.load(path)
    out = np.full((len(cores), MAX_SEATS), np.nan)
    for a in range(0, len(cores), 256):
        chunk = cores[a : a + 256]
        obs = np.stack([encode(c) for c in chunk]).astype(np.float32)
        mask = np.zeros((len(chunk), NUM_DISTINCT_ACTIONS), np.bool_)
        for i, c in enumerate(chunk):
            mask[i, rules.legal_actions(c)] = True
        _, v = w.forward(obs, mask)
        for i, c in enumerate(chunk):
            n = int(c.players)
            m = int(movers[a + i])
            for j in range(n):
                out[a + i, (m + j) % n] = v[i, j]
    return out


def analyse(data: dict[str, Array], nets: dict[str, Array], reps: int, seed: int) -> dict[str, Any]:
    """Ceilings and net EVs by quartile (1..4, "all") for ego and all-seat units."""
    rng = np.random.default_rng(seed)
    quart, game, mover, players = data["quartile"], data["game"], data["mover"], data["players"]
    P = len(quart)
    summary: dict[str, Any] = {}
    for unit in ("ego", "pooled"):
        if unit == "ego":
            rows, seats = np.arange(P), mover.astype(int)
        else:
            rows = np.concatenate([np.full(int(players[i]), i) for i in range(P)])
            seats = np.concatenate([np.arange(int(players[i])) for i in range(P)])
        for qty in QUANTITIES:
            r = data[qty][rows, seats, :]
            for qname in ("q1", "q2", "q3", "q4", "all"):
                sel = (
                    np.ones(len(rows), bool) if qname == "all" else quart[rows] == int(qname[1]) - 1
                )
                rr, gg = r[sel], game[rows][sel]
                ent: dict[str, Any] = {"units": int(sel.sum()), **ceiling(rr)}
                ent["ci"] = bootstrap(ceiling, gg, [rr], reps, rng)
                for name, pv in nets.items():
                    p = pv[rows, seats][sel]
                    ne: dict[str, Any] = {**net_ev(p, rr)}
                    ne["ci"] = bootstrap(net_ev, gg, [p, rr], reps, rng)
                    ent[name] = ne
                summary.setdefault(unit, {}).setdefault(qty, {})[qname] = ent
    return summary


def _fmt(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:+.3f}" if x < 0 else f"{x:.3f}"


def markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    head_meta = json.dumps({k: v for k, v in meta.items() if k != "nets"})
    lines = [f"# Noise ceiling ({meta['policy']} continuations)", "", "```", head_meta, "```", ""]
    nets = meta["nets"]
    for unit in ("ego", "pooled"):
        for qty in QUANTITIES:
            lines += [f"## {unit} seat{'s' if unit == 'pooled' else ''}, {qty}", ""]
            head = "| quartile | units | EV_max [95% CI] | within | between |"
            sep = "|---|---|---|---|---|"
            for n in nets:
                head += f" {n} EV single [CI] | {n} R² vs E[R|s] [CI] |"
                sep += "---|---|"
            lines += [head, sep]
            for qname, e in summary[unit][qty].items():
                ci = e["ci"]["ev_max"]
                row = (f"| {qname} | {e['units']} | {_fmt(e['ev_max'])} [{_fmt(ci[0])}, "
                       f"{_fmt(ci[1])}] | {e['within']:.4f} | {e['between']:.4f} |")  # fmt: skip
                for n in nets:
                    ne = e[n]
                    a, b = ne["ci"]["ev_single"], ne["ci"]["r2_mean"]
                    row += (f" {_fmt(ne['ev_single'])} [{_fmt(a[0])}, {_fmt(a[1])}] |"
                            f" {_fmt(ne['r2_mean'])} [{_fmt(b[0])}, {_fmt(b[1])}] |")  # fmt: skip
                lines.append(row)
            lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main


def _chunks(xs: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(xs[i : i + size]) for i in range(0, len(xs), size)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_game = args.per_game
    if per_game not in (1, 2) or args.positions % (4 * per_game):
        raise SystemExit("--per-game must be 1 or 2 and --positions a multiple of 4 * per-game")
    n_games = args.positions // per_game
    quarts = [[i % 4] if per_game == 1 else [i % 4, (i + 2) % 4] for i in range(n_games)]
    nets = dict(x.split("=", 1) for x in args.nets.split(",") if x)
    nets = {k: v for k, v in nets.items() if Path(v).exists() or print(f"skip {k}: no {v}")}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        tasks = [(args.seed, ids, [quarts[i] for i in ids], args.sample_net)
                 for ids in _chunks(range(n_games), args.games_per_task)]  # fmt: skip
        pos: list[dict[str, Any]] = []
        for res in pool.imap_unordered(sample_task, tasks):
            pos.extend(res)
        pos.sort(key=lambda d: (d["game"], d["quartile"]))
        t1 = time.perf_counter()
        print(f"sampled {len(pos)} positions from {n_games} games in {t1 - t0:.0f}s", flush=True)
        items = [(i, d["core"]) for i, d in enumerate(pos)]
        ctasks = [(args.seed, ch, args.k, args.policy, args.cont_net, args.mix)
                  for ch in _chunks(items, args.positions_per_task)]  # fmt: skip
        cont: dict[int, dict[str, Any]] = {}
        for res in pool.imap_unordered(continue_task, ctasks):
            for d in res:
                cont[d["pid"]] = d
            if len(cont) % 40 < args.positions_per_task:
                el = time.perf_counter() - t1
                print(f"  continued {len(cont)}/{len(pos)} positions, {el:.0f}s", flush=True)
    t2 = time.perf_counter()
    data = {
        "game": np.asarray([d["game"] for d in pos]),
        "quartile": np.asarray([d["quartile"] for d in pos]),
        "prog": np.asarray([d["prog"] for d in pos]),
        "mover": np.asarray([d["mover"] for d in pos]),
        "decision": np.asarray([d["decision"] for d in pos]),
        "length": np.asarray([d["length"] for d in pos]),
        "players": np.asarray([int(d["core"].players) for d in pos]),
        **{q: np.stack([cont[i][q] for i in range(len(pos))]) for q in QUANTITIES},
    }
    cores = [d["core"] for d in pos]
    preds = {name: net_values(path, cores, data["mover"]) for name, path in nets.items()}
    summary = analyse(data, preds, args.bootstrap, args.seed)
    t3 = time.perf_counter()
    meta = {
        "policy": args.policy, "positions": len(pos), "k": args.k, "per_game": per_game,
        "games": n_games, "players": 4, "seed": args.seed, "sample_net": args.sample_net,
        "cont_net": args.cont_net if args.policy == "net" else None, "mix": args.mix,
        "workers": args.workers, "bootstrap": args.bootstrap, "nets": nets,
        "seconds": {"sample": round(t1 - t0, 1), "continue": round(t2 - t1, 1),
                    "analyse": round(t3 - t2, 1), "total": round(t3 - t0, 1)},
    }  # fmt: skip
    arrays: dict[str, Any] = {**data, **{f"pred_{k}": v for k, v in preds.items()}}
    np.savez_compressed(out / "positions.npz", **arrays)
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    (out / "summary.md").write_text(markdown(summary, meta))
    print(f"done in {t3 - t0:.0f}s -> {out}", flush=True)
    return {"meta": meta, "summary": summary}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--positions", type=int, default=400, help="total, equal per quartile")
    ap.add_argument("--k", type=int, default=32, help="continuations per position")
    ap.add_argument("--per-game", type=int, default=2, help="positions per sampling game (1|2)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=("net", "heuristic"), default="net",
                    help="continuation policy in all seats")  # fmt: skip
    ap.add_argument("--sample-net", default="models/ppo-long.npz", help="positions' self-play net")
    ap.add_argument("--cont-net", default="models/ppo-long.npz", help="continuation net (τ = 1)")
    ap.add_argument("--nets", default=DEFAULT_NETS, help="name=npz,... value nets to score")
    ap.add_argument("--mix", type=float, default=0.5, help="R = mix wl + (1 - mix) margin")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--games-per-task", type=int, default=8)
    ap.add_argument("--positions-per-task", type=int, default=4)
    ap.add_argument("--out", default="runs/noise_ceiling/net")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
