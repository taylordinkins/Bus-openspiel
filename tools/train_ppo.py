# pyright: basic, reportMissingImports=false
"""PPO self-play trainer for Bus (Phase 4 unit 3).

    uv run --group train python -m tools.train_ppo --name smoke --players 4 --updates 40 \\
        --games-per-update 128 --workers 11

One shared ego-rotated policy plays every seat (pure self-play; the opponent league is unit 4).
Rollouts run in a spawn ``multiprocessing`` pool on CPU with a numpy copy of the weights
(``train.rollout``); the learner (``train.ppo``) runs on the GPU when available. Reward is the
shaped return ``mix * win_loss + (1 - mix) * margin`` with ``mix`` annealed linearly to 1.0;
the periodic evaluation is always pure win_loss against 3 random bots.

Outputs in ``runs/<name>/``: ``log.csv`` (one row per update), ``ckpt_<update>.pt`` (model +
optimizer + config) and ``ckpt_<update>.npz`` (numpy export for ``bots.net_forward``).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Children inherit these at spawn (before numpy loads there): one BLAS thread per worker.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train.export import export_npz  # noqa: E402
from train.features import FEATURES_VERSION  # noqa: E402
from train.model import BusNet  # noqa: E402
from train.ppo import (  # noqa: E402
    PPOConfig,
    explained_variance,
    linear,
    mix_at,
    net_to_weights,
    ppo_update,
)
from train.rollout import END_TYPES, RolloutConfig, run_task, worker_init  # noqa: E402

from .tournament import wilson_interval  # noqa: E402

COLUMNS = [
    "update", "games", "decisions", "samples", "decisions_per_s", "games_per_s", "env_s",
    "learn_s", "worker_busy", "entropy", "approx_kl", "clip_frac", "pg_loss", "value_loss",
    "explained_var", "mean_len", "illegal", "mix", "lr", "ent_coef",
    *[f"end_{t}" for t in END_TYPES],
    "eval_games", "eval_win", "eval_lo", "eval_hi", "elapsed_s",
]  # fmt: skip


def parse_players(text: str) -> tuple[int, ...]:
    ps = tuple(int(x) for x in text.split(",") if x)
    if not ps or any(p not in (3, 4, 5) for p in ps):
        raise argparse.ArgumentTypeError("players must be a list from 3,4,5")
    return ps


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PPO self-play trainer for Bus.")
    ap.add_argument("--name", default="run", help="output dir is <out-dir>/<name>")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--players", type=parse_players, default=(4,), help="e.g. 4 or 3,4,5")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--games-per-update", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop after this (0 = off)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--envs-per-worker", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="512,512")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-decay", action="store_true")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-coef-final", type=float, default=None)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--mix", type=float, default=0.7, help="initial win_loss share of the reward")
    ap.add_argument("--mix-anneal-updates", type=int, default=300, help="0 = keep --mix")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=10, help="0 = off")
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-greedy", action="store_true")
    return ap


def _chunks(total: int, parts: int) -> list[int]:
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


class Runner:
    """Runs rollout tasks in a spawn pool (or inline with 0 workers)."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.pool = (
            mp.get_context("spawn").Pool(workers, initializer=worker_init) if workers > 0 else None
        )

    def map(self, tasks: list[Any]) -> list[dict[str, Any]]:
        if self.pool is None:
            return [run_task(t) for t in tasks]
        return self.pool.map(run_task, tasks, chunksize=1)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def evaluate(
    runner: Runner, weights: Any, games: int, players: tuple[int, ...], greedy: bool, seed: int,
    envs: int,
) -> tuple[int, float, float, float]:  # fmt: skip
    """Win rate (pure win_loss, ego = net, 3 random opponents) with Wilson 95% CI."""
    cfg = RolloutConfig(
        players=players, mix=1.0, envs_per_worker=envs, greedy=greedy, collect=False
    )
    sizes = _chunks(games, max(runner.workers, 1))
    tasks, start = [], 0
    for i, sz in enumerate(sizes):
        tasks.append((weights, sz, seed + i, cfg, list(range(start, start + sz)), start))
        start += sz
    wins = n = 0.0
    for r in runner.map(tasks):
        for wl, _ in r["stats"]["ego_returns"]:
            n += 1
            wins += 1.0 if wl > 0 else 0.0
    lo, hi = wilson_interval(wins, int(n))
    return int(n), wins / max(n, 1), lo, hi


def save_checkpoint(
    out: Path, update: int, model: BusNet, opt: torch.optim.Optimizer, args: argparse.Namespace,
    mix: float, games: int,
) -> None:  # fmt: skip
    meta = {
        "update": update, "mix": mix, "games": games, "hidden": list(model.hidden),
        "features_version": FEATURES_VERSION, "players": list(args.players), "seed": args.seed,
    }  # fmt: skip
    torch.save(
        {
            "model": model.state_dict(), "opt": opt.state_dict(), "update": update,
            "meta": meta, "args": {k: v for k, v in vars(args).items()},
        },
        out / f"ckpt_{update}.pt",
    )  # fmt: skip
    export_npz(model, out / f"ckpt_{update}.npz", meta)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_name)
    model = BusNet(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    pcfg = PPOConfig(
        lr=args.lr, lr_decay=args.lr_decay, clip=args.clip, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, ent_coef_final=args.ent_coef_final,
        max_grad_norm=args.max_grad_norm, epochs=args.epochs, minibatches=args.minibatches,
        target_kl=args.target_kl,
    )  # fmt: skip
    gen = torch.Generator().manual_seed(args.seed)
    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1, default=list))
    runner = Runner(args.workers)
    t0 = time.perf_counter()
    total_games = 0
    last_ckpt = 0
    print(f"device={device} workers={args.workers} hidden={hidden} out={out}", flush=True)
    try:
        with open(out / "log.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, COLUMNS)
            writer.writeheader()
            for update in range(1, args.updates + 1):
                mix = mix_at(update - 1, args.mix, args.mix_anneal_updates)
                lr = (
                    linear(args.lr, args.lr * 0.1, update - 1, args.updates)
                    if args.lr_decay
                    else args.lr
                )
                for g in opt.param_groups:
                    g["lr"] = lr
                ent_coef = (
                    args.ent_coef
                    if args.ent_coef_final is None
                    else linear(args.ent_coef, args.ent_coef_final, update - 1, args.updates)
                )
                rcfg = RolloutConfig(
                    players=args.players, gamma=args.gamma, lam=args.lam, mix=mix,
                    envs_per_worker=args.envs_per_worker,
                )  # fmt: skip
                weights = net_to_weights(model)
                te = time.perf_counter()
                sizes = _chunks(args.games_per_update, max(args.workers, 1) * 2)
                base = args.seed * 1_000_003 + update * 1009
                tasks = [(weights, sz, base + i, rcfg, None, 0) for i, sz in enumerate(sizes)]
                results = runner.map(tasks)
                env_s = time.perf_counter() - te

                tl = time.perf_counter()
                parts = [r["samples"] for r in results if r["samples"] is not None]
                cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                n_samples = len(cat["action"])
                ev = explained_variance(cat["value"], cat["vtarget"][:, 0])
                data = {
                    "obs": torch.from_numpy(cat["obs"]), "mask": torch.from_numpy(cat["mask"]),
                    "action": torch.from_numpy(cat["action"]),
                    "logp": torch.from_numpy(cat["logp"]), "adv": torch.from_numpy(cat["adv"]),
                    "vtarget": torch.from_numpy(cat["vtarget"]),
                    "vmask": torch.from_numpy(cat["vmask"]),
                }  # fmt: skip
                data = {k: v.to(device) for k, v in data.items()}
                diag = ppo_update(model, opt, data, pcfg, ent_coef, gen)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                learn_s = time.perf_counter() - tl

                games = sum(r["stats"]["games"] for r in results)
                decisions = sum(r["stats"]["decisions"] for r in results)
                lengths = [x for r in results for x in r["stats"]["lengths"]]
                total_games += games
                row: dict[str, Any] = {
                    "update": update, "games": total_games, "decisions": decisions,
                    "samples": n_samples, "decisions_per_s": decisions / max(env_s, 1e-9),
                    "games_per_s": games / max(env_s, 1e-9), "env_s": env_s, "learn_s": learn_s,
                    "worker_busy": sum(r["seconds"] for r in results)
                    / max(env_s * max(args.workers, 1), 1e-9),
                    **diag, "explained_var": ev, "mean_len": float(np.mean(lengths)),
                    "illegal": sum(r["stats"]["illegal"] for r in results), "mix": mix, "lr": lr,
                    "ent_coef": ent_coef,
                }  # fmt: skip
                for t in END_TYPES:
                    row[f"end_{t}"] = sum(r["stats"]["end"][t] for r in results) / max(games, 1)
                if args.eval_every and update % args.eval_every == 0:
                    n, w, lo, hi = evaluate(
                        runner, weights, args.eval_games, args.players, args.eval_greedy,
                        args.seed + 10**6 + update * 1000, args.envs_per_worker,
                    )  # fmt: skip
                    row.update(eval_games=n, eval_win=w, eval_lo=lo, eval_hi=hi)
                row["elapsed_s"] = time.perf_counter() - t0
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
                fh.flush()
                print(
                    f"u{update} games={total_games} dec/s={row['decisions_per_s']:.0f} "
                    f"env={env_s:.1f}s learn={learn_s:.1f}s ent={row['entropy']:.3f} "
                    f"kl={row['approx_kl']:.4f} vl={row['value_loss']:.4f} ev={ev:.2f} "
                    f"mix={mix:.2f}"
                    + (f" eval={row['eval_win']:.2f}[{row['eval_lo']:.2f},{row['eval_hi']:.2f}]"
                       if "eval_win" in row else ""),
                    flush=True,
                )  # fmt: skip
                stop = bool(args.max_minutes) and (time.perf_counter() - t0) > args.max_minutes * 60
                if update % args.ckpt_every == 0 or update == args.updates or stop:
                    save_checkpoint(out, update, model, opt, args, mix, total_games)
                    last_ckpt = update
                if stop:
                    break
    finally:
        runner.close()
    print(f"done: last checkpoint ckpt_{last_ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
