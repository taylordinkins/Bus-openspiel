# pyright: basic, reportMissingImports=false
"""Gumbel AlphaZero trainer for Bus (Phase 4 unit 8h; design in ``docs/GUMBEL_AZ.md``); the CLI is
``tools.train_az``.

    uv run --group train python -m tools.train_az --name az1 \\
        --init-from runs/long/best_1320.pt --max-minutes 90

Synchronous generations: every generation the actors (``--workers`` spawn processes, numpy only,
``train.az_selfplay``) play ``--games-per-gen`` games under Gumbel search with the current
weights, ``--concurrent`` games in flight per task and one batched forward per search wave; the
samples come back through tmpfs spill files; the learner (``train.az_learn``, GPU when available)
then runs ``ceil(--reuse * new samples / --batch)`` minibatch steps on a FIFO replay of the last
``--replay-size`` samples. Generation 0 plays with the initial (warm-started) weights.

Targets: the completed-Q improved policy at the root (policy), the final per-seat
R = ``--mix`` win_loss + (1 - ``--mix``) relative term (value, ego order like the PPO head;
``--value-quantity`` picks the mean-opponent margin (default, R-072), the leader-relative margin
or the rank term; ``--value-target zq`` averages in the root search value), and optional aux
heads (``--aux-coef``, training only, never exported). The first ``--warmup-gens`` generations
train the value side only (policy loss off).

Search opponent model: ``--opp-temp`` / ``--opp-mix`` / ``--opp-scope`` flatten the prior at
non-root opponent nodes (``bots.gumbel.opponent_prior``; defaults leave the search unchanged).

Tables: ``--frozen-share`` of games seat a frozen net (``--frozen-nets`` manifest names and, every
``--snapshot-every`` generations, the learner's own snapshot; the last ``--snapshot-keep`` kept)
in 1..n-1 random seats; ``--heuristic-share`` of games (disjoint from the frozen ones) seat
``bots.registry`` heuristic bots in 1..n-1 random seats; the rest is pure self-play. Only learner
seats produce samples. Both shares are run-defining (a change on resume is rejected).

Evaluation (``--eval-every`` generations, generation 0 first, before any training): the raw
policy (greedy unless ``--no-eval-greedy``; no search) as one seat vs 3 sampled copies of each
``--eval-nets`` net and vs 3 heuristic seats, seats rotated, shared wins credited 1/k, Wilson 95%
intervals, mean score margin. ``best.*`` follows the best win rate vs the first ``--eval-nets``
entry.

Outputs in ``runs/<name>/``: ``log.csv`` (one row per generation: losses, value EV overall and by
game quartile on the fresh samples before training, policy-target entropy, KL(pi' || prior),
share of moves where search overrode the prior's argmax, sims/s, games/h, RSS, and per table
kind self / frozen / heuristic the share of games and the learner seats' mean win credit,
1/k for a k-way shared win), ``eval.csv``,
``ckpt_<gen>.pt/.npz``, ``latest.json``, ``best.*``, ``pool/snap_<gen>.npz``, ``args.json``,
``resume_log.json``. ``--resume <run dir>`` continues from the latest checkpoint (model,
optimizer, generation, frozen pool; the replay is not saved and refills from the resume point).
SIGINT/SIGTERM or ``--max-minutes`` finish the current generation, save and exit 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bots.gumbel import GumbelConfig  # noqa: E402
from bots.net_forward import layout_hash  # noqa: E402
from tools.tournament import wilson_interval  # noqa: E402
from tools.train_ppo import (  # noqa: E402
    AbortRollout,
    ForcedExit,
    Runner,
    StopSignals,
    WorkerFailure,
    _chunks,
    append_rows,
    load_init_from,
    parse_players,
    resolve_player_weights,
)
from train.atomic import atomic_write_json  # noqa: E402
from train.az_learn import LearnConfig, diagnostics, train_steps  # noqa: E402
from train.az_replay import Replay  # noqa: E402
from train.az_selfplay import (  # noqa: E402
    TABLE_KINDS,
    ActorConfig,
    run_az_task,
    run_az_task_spill,
    win_credit,
)
from train.export import export_npz  # noqa: E402
from train.features import FEATURES_VERSION  # noqa: E402
from train.model import BusNet  # noqa: E402
from train.ppo import cosine, net_to_weights  # noqa: E402
from train.rollout import RolloutConfig, load_spilled  # noqa: E402
from train.runstate import (  # noqa: E402
    ResumeError,
    append_resume_log,
    apply_retention,
    explicit_options,
    find_latest_checkpoint,
    merge_resume_args,
    now_iso,
    remove_stray_tmp,
    rss_mb,
    save_pt,
    truncate_csv,
    write_latest,
)
from train.shm import cleanup_own, cleanup_stale  # noqa: E402

EVAL_SEED_BIT = 1 << 61
COLUMNS = [
    "gen", "games", "samples", "replay", "steps", "value_only", "lr", "loss_pi", "loss_v",
    "loss_aux", "ev", "ev_q1", "ev_q2", "ev_q3", "ev_q4", "policy_ce", "target_entropy",
    "kl_prior", "search_differs", "win_self", "win_frozen", "frozen_seats", "sims_per_s",
    "games_per_h", "decisions_per_s", "rss_mb", "worker_rss_mb", "replay_mb", "gen_s",
    "learn_s", "eval_s", "elapsed_s", "tables_self", "tables_frozen", "tables_heuristic",
    "win_heuristic", "heuristic_seats",
]  # fmt: skip
EVAL_COLUMNS = ["gen", "opponent", "players", "games", "win_rate", "lo", "hi", "mean_margin"]
RESUME_WHITELIST = frozenset({
    "max_minutes", "workers", "eval_every", "eval_games", "ckpt_every", "keep_every",
    "stop_after", "task_timeout", "max_retries", "device", "generations", "games_per_gen",
    "concurrent", "lr", "lr_final", "reuse",
})  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Gumbel AlphaZero trainer for Bus.")
    ap.add_argument("--name", default="az", help="output dir is <out-dir>/<name>")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--players", type=parse_players, default=(4,), help="e.g. 4 or 3,4,5")
    ap.add_argument("--player-weights", default=None, help="sampling weights over --players")
    ap.add_argument("--generations", type=int, default=10_000, help="schedule horizon")
    ap.add_argument("--games-per-gen", type=int, default=256)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop after this (0 = off)")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N generations (session)")
    ap.add_argument("--workers", type=int, default=6, help="actor processes (0 = inline)")
    ap.add_argument("--concurrent", type=int, default=32, help="games in flight per actor task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="512,512")
    ap.add_argument("--value-trunk", choices=("separate", "shared"), default="separate")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--init-from", default=None, metavar="CKPT.pt", help="warm start (fresh run)")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR")
    # search
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--m", type=int, default=16, help="actions considered at the root")
    ap.add_argument("--c-scale", type=float, default=0.3)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--mix", type=float, default=0.5, help="win_loss share of R")
    ap.add_argument(
        "--value-quantity", choices=("mean", "leader", "rank"), default="mean",
        help="relative term of R: mean-opponent margin (R-072), leader-relative margin or rank "
        "(bots.gumbel.relative_term); used for z and for terminal search values",
    )  # fmt: skip
    ap.add_argument("--opp-temp", type=float, default=1.0, help="opponent-node prior temperature")
    ap.add_argument("--opp-mix", type=float, default=0.0, help="uniform share in opponent priors")
    ap.add_argument("--opp-scope", choices=("opponents", "nonroot"), default="opponents")
    # tables
    ap.add_argument("--frozen-share", type=float, default=0.2, help="share of tables w/ frozen")
    ap.add_argument(
        "--heuristic-share", type=float, default=0.0,
        help="share of tables w/ heuristic seats (disjoint from --frozen-share)",
    )  # fmt: skip
    ap.add_argument("--frozen-nets", default="ppo-long", help="manifest names (comma list)")
    ap.add_argument("--snapshot-every", type=int, default=10, help="gens; 0 = no own snapshots")
    ap.add_argument("--snapshot-keep", type=int, default=4)
    # learner
    ap.add_argument("--replay-size", type=int, default=500_000)
    ap.add_argument("--min-replay", type=int, default=0, help="no training below this many")
    ap.add_argument("--reuse", type=float, default=4.0, help="samples trained per new sample")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=1e-4, help="cosine end over --generations")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--aux-coef", type=float, default=0.25, help="0 = no aux heads")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--value-target", choices=("z", "zq"), default="z")
    ap.add_argument("--warmup-gens", type=int, default=2, help="value-only generations")
    # eval / checkpoints
    ap.add_argument("--eval-every", type=int, default=10, help="generations; 0 = off")
    ap.add_argument("--eval-games", type=int, default=200, help="per opponent")
    ap.add_argument(
        "--eval-nets", default="ppo-long=models/ppo-long.npz,exploiter=runs/exploit_long/best.npz",
        help="name=npz list; each is evaluated as 3 sampled seats (missing files are skipped)",
    )  # fmt: skip
    ap.add_argument("--eval-heuristic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-greedy", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--keep-every", type=int, default=50, help="milestone ckpt every N gens")
    ap.add_argument("--task-timeout", type=float, default=1800.0, help="s per batch; 0 = off")
    ap.add_argument("--max-retries", type=int, default=3)
    return ap


# ---------------------------------------------------------------- helpers


def actor_config(args: argparse.Namespace) -> ActorConfig:
    return ActorConfig(
        players=tuple(args.players),
        player_weights=resolve_player_weights(tuple(args.players), args.player_weights),
        sims=args.sims,
        search=GumbelConfig(
            m=args.m, c_visit=args.c_visit, c_scale=args.c_scale, gumbel_scale=args.gumbel_scale,
            mix=args.mix, target=args.value_quantity, opp_temp=args.opp_temp,
            opp_mix=args.opp_mix, opp_scope=args.opp_scope,
        ),  # fmt: skip
        concurrent=args.concurrent, frozen_share=args.frozen_share,
        heuristic_share=args.heuristic_share,
    )  # fmt: skip


def frozen_base(args: argparse.Namespace) -> dict[str, str]:
    from bots.net import resolve_model

    out: dict[str, str] = {}
    for name in (x.strip() for x in args.frozen_nets.split(",") if x.strip()):
        out[f"hof:{name}"] = str(resolve_model(name).path)
    return out


def eval_nets(args: argparse.Namespace) -> list[tuple[str, str]]:
    out = []
    for item in (x.strip() for x in args.eval_nets.split(",") if x.strip()):
        name, eq, path = item.partition("=")
        if not eq:
            raise ResumeError(f"--eval-nets entries are name=path, got {item!r}")
        if not Path(path).exists():
            print(f"WARNING: eval net {name} skipped: {path} not found", flush=True)
            continue
        out.append((name, path))
    return out


def evaluate(
    runner: Runner, weights: Any, args: argparse.Namespace, gen: int,
    abort: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """Raw policy vs 3 copies of each eval opponent per player count; 1/k shared-win credit."""
    specs = [(name, f"hof:eval_{name}") for name, _ in eval_nets(args)]
    opponents = {f"hof:eval_{name}": path for name, path in eval_nets(args)}
    if args.eval_heuristic:
        specs.append(("heuristic", "heuristic"))
    cells = [(k, n) for k in range(len(specs)) for n in sorted(set(args.players))]
    tasks: list[Any] = []
    owner: list[int] = []
    for ci, (k, n) in enumerate(cells):
        cfg = RolloutConfig(
            players=(n,), mix=1.0, envs_per_worker=16, collect=False,
            eval_opponent=specs[k][1], record_scores=True, ego_greedy=args.eval_greedy,
        )  # fmt: skip
        start = 0
        for i, sz in enumerate(_chunks(args.eval_games, max(runner.workers, 1))):
            seed = EVAL_SEED_BIT + ((args.seed * 100_003 + gen) * 64 + k * 3 + (n - 3)) * 4096 + i
            rt = (weights, sz, seed, cfg, list(range(start, start + sz)), start, opponents)
            tasks.append(("eval", rt))
            owner.append(ci)
            start += sz
    credit = [0.0] * len(cells)
    margin = [0.0] * len(cells)
    games = [0] * len(cells)
    for ci, r in zip(owner, runner.map(tasks, abort), strict=True):
        n = cells[ci][1]
        for wl, _n in r["stats"]["ego_returns"]:
            games[ci] += 1
            credit[ci] += win_credit(wl, n)
        for labels, sc in r["stats"]["scores"]:
            ego = [x for x, lab in zip(sc, labels, strict=True) if lab == "current"]
            oth = [x for x, lab in zip(sc, labels, strict=True) if lab != "current"]
            margin[ci] += sum(ego) / len(ego) - sum(oth) / len(oth)
    rows = []
    for ci, (k, n) in enumerate(cells):
        g = max(games[ci], 1)
        lo, hi = wilson_interval(credit[ci], games[ci])
        rows.append({
            "gen": gen, "opponent": specs[k][0], "players": n, "games": games[ci],
            "win_rate": credit[ci] / g, "lo": lo, "hi": hi, "mean_margin": margin[ci] / g,
        })  # fmt: skip
    return rows


def save_checkpoint(
    out: Path, gen: int, model: BusNet, opt: torch.optim.Optimizer, args: argparse.Namespace,
    state: dict[str, Any],
) -> None:  # fmt: skip
    meta = {
        "gen": gen, "hidden": list(model.hidden), "features_version": FEATURES_VERSION,
        "layout_hash": layout_hash(), "obs_size": model.obs_size, "seed": args.seed,
        "value_trunk": "separate" if model.separate_value else "shared", "trainer": "az",
    }  # fmt: skip
    save_pt(
        out / f"ckpt_{gen}.pt",
        {"model": model.state_dict(), "opt": opt.state_dict(), "update": gen, "meta": meta,
         "args": vars(args), "state": state},
    )  # fmt: skip
    export_npz(model, out / f"ckpt_{gen}.npz", meta)
    write_latest(out, gen)
    apply_retention(out, keep_last=3, keep_every=args.keep_every)


def prepare(argv: list[str]) -> tuple[argparse.Namespace, Path, dict[str, Any] | None]:
    ap = build_parser()
    args = ap.parse_args(argv)
    for k in ("frozen_share", "heuristic_share"):
        if not 0.0 <= getattr(args, k) <= 1.0:
            raise ResumeError(f"--{k.replace('_', '-')} must be in [0, 1]")
    if args.frozen_share + args.heuristic_share > 1.0:
        raise ResumeError("--frozen-share + --heuristic-share must be <= 1")
    if args.resume and args.init_from:
        raise ResumeError("--init-from cannot be combined with --resume")
    if not args.resume:
        out = Path(args.out_dir) / args.name
        if out.exists() and ((out / "latest.json").exists() or any(out.glob("ckpt_*.pt"))):
            raise ResumeError(f"{out} already holds a run; continue it with --resume {out}")
        return args, out, None
    run = Path(args.resume)
    if not (run / "args.json").exists():
        raise ResumeError(f"{run} is not a run directory (no args.json)")
    saved = json.loads((run / "args.json").read_text())
    merged, changes = merge_resume_args(
        saved, explicit_options(ap, argv), vars(ap.parse_args([])), RESUME_WHITELIST
    )
    merged["resume"] = None
    merged["players"] = tuple(merged["players"])
    found = find_latest_checkpoint(run)
    if found is None:
        raise ResumeError(f"no loadable checkpoint in {run}")
    gen, ck = found
    if ck.get("meta", {}).get("trainer") != "az":
        raise ResumeError(f"{run} is not an AZ run")
    return argparse.Namespace(**merged), run, {"gen": gen, "ckpt": ck, "changes": changes}


# ---------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args, out, res = prepare(argv)
        base_frozen = frozen_base(args)
    except (ResumeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    model = BusNet(
        hidden=hidden, separate_value=args.value_trunk == "separate", aux=args.aux_coef > 0
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lcfg = LearnConfig(
        batch=args.batch, vf_coef=args.vf_coef, aux_coef=args.aux_coef,
        max_grad_norm=args.max_grad_norm, value_target=args.value_target,
    )  # fmt: skip
    acfg = actor_config(args)
    start = 0
    elapsed_base = 0.0
    snaps: list[int] = []
    best: dict[str, Any] | None = None
    if res is None and args.init_from:
        try:
            load_init_from(model, args)
        except ResumeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        model.to(device)
        print(f"warm-started weights from {args.init_from}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pool").mkdir(exist_ok=True)
    if res is not None:
        ck = res["ckpt"]
        start = int(res["gen"])
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck.get("state") or {}
        elapsed_base = float(st.get("elapsed_s", 0.0))
        snaps = list(st.get("snaps", []))
        best = st.get("best")
        if not (out / "args.original.json").exists():
            atomic_write_json(
                out / "args.original.json", json.loads((out / "args.json").read_text())
            )
        append_resume_log(
            out, {"time": now_iso(), "from_gen": start, "overrides": res["changes"], "argv": argv,
                  "note": "replay not saved: refills from this generation"},
        )  # fmt: skip
    atomic_write_json(out / "args.json", vars(args))
    remove_stray_tmp(out)
    cleanup_stale()
    truncate_csv(out / "log.csv", COLUMNS, start)
    truncate_csv(out / "eval.csv", EVAL_COLUMNS, start - 1)
    if res is not None:
        print(f"resumed {out} at generation {start}; overrides: {res['changes'] or 'none'}")
    replay = Replay(args.replay_size)
    sig = StopSignals()
    sig.install()
    runner = Runner(
        args.workers, args.task_timeout, args.max_retries, run_az_task, run_az_task_spill
    )
    t0 = time.perf_counter()
    completed = start
    last_ckpt = start
    exit_code = 0
    print(f"device={device} workers={args.workers} sims={args.sims} c_scale={args.c_scale} "
          f"out={out}", flush=True)  # fmt: skip

    def elapsed() -> float:
        return elapsed_base + time.perf_counter() - t0

    def checkpoint(gen: int) -> None:
        nonlocal last_ckpt
        save_checkpoint(
            out, gen, model, opt, args,
            {"elapsed_s": elapsed(), "snaps": snaps, "best": best, "replay_size": replay.size},
        )  # fmt: skip
        last_ckpt = gen

    def stop_now() -> bool:
        if sig.requested:
            return True
        if args.max_minutes > 0 and elapsed() >= args.max_minutes * 60:
            return True
        return args.stop_after > 0 and completed - start >= args.stop_after

    try:
        for gen in range(start, args.generations):
            if stop_now():
                break
            g_t0 = time.perf_counter()
            eval_s = 0.0
            if args.eval_every > 0 and gen % args.eval_every == 0:
                te = time.perf_counter()
                rows = evaluate(runner, net_to_weights(model), args, gen, sig.overdue)
                append_rows(out / "eval.csv", EVAL_COLUMNS, rows)
                eval_s = time.perf_counter() - te
                for r in rows:
                    print(
                        f"  eval g{gen} vs {r['opponent']} {r['players']}p: "
                        f"{r['win_rate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] "
                        f"margin {r['mean_margin']:+.2f} ({r['games']} games)", flush=True,
                    )  # fmt: skip
                key = [r for r in rows if r["opponent"] == rows[0]["opponent"]] if rows else []
                if key:
                    wr = sum(r["win_rate"] * r["games"] for r in key) / sum(r["games"] for r in key)
                    if best is None or wr > best["win_rate"]:
                        best = {"gen": gen, "opponent": key[0]["opponent"], "win_rate": wr}
                        meta = {"gen": gen, "best": best, "features_version": FEATURES_VERSION}
                        save_pt(out / "best.pt", {"model": model.state_dict(), "meta": meta})
                        export_npz(model, out / "best.npz", meta)
                        atomic_write_json(out / "best.json", best)
            weights = net_to_weights(model)
            opponents = dict(base_frozen)
            for s in snaps:
                opponents[f"snap:{s:06d}"] = str(out / "pool" / f"snap_{s}.npz")
            tasks = []
            for i, sz in enumerate(_chunks(args.games_per_gen, max(args.workers, 1))):
                seed = (args.seed * 1_000_003 + gen) * 4096 + i
                tasks.append(("selfplay", weights, sz, seed, acfg, opponents))
            results = [load_spilled(r) for r in runner.map(tasks, sig.overdue)]
            parts = [r["samples"] for r in results if r["samples"] is not None]
            n_new = sum(len(p["z"]) for p in parts)
            diag = diagnostics(model, parts, device, rng=np.random.default_rng([args.seed, gen, 1]))
            for p in parts:
                replay.add(p)
            lt = time.perf_counter()
            steps = 0
            if replay.size >= args.min_replay and n_new:
                steps = math.ceil(args.reuse * n_new / args.batch)
            lr = cosine(args.lr, args.lr_final, gen, args.generations)
            for group in opt.param_groups:
                group["lr"] = lr
            value_only = gen < args.warmup_gens
            lrn = train_steps(
                model, opt, replay, steps, np.random.default_rng([args.seed, gen, 2]), lcfg,
                device, value_only,
            )  # fmt: skip
            learn_s = time.perf_counter() - lt
            completed = gen + 1
            if args.snapshot_every > 0 and completed % args.snapshot_every == 0:
                export_npz(model, out / "pool" / f"snap_{completed}.npz", {"gen": completed})
                snaps = [*snaps, completed][-args.snapshot_keep :] if args.snapshot_keep else []
                for p in (out / "pool").glob("snap_*.npz"):
                    if int(p.stem[5:]) not in snaps:
                        p.unlink(missing_ok=True)
            st = [r["stats"] for r in results]
            dec = sum(s["decisions"] for s in st)
            sims = sum(s["sims"] for s in st)
            gen_s = time.perf_counter() - g_t0
            play_s = max(gen_s - learn_s - eval_s, 1e-9)

            def tab(kind: str, i: int, st: list[dict[str, Any]] = st) -> float:
                return float(sum(s["tables"].get(kind, [0, 0.0, 0.0, 0])[i] for s in st))

            n_games = sum(s["games"] for s in st)

            row = {
                "gen": completed, "games": sum(s["games"] for s in st), "samples": n_new,
                "replay": replay.size, "steps": steps, "value_only": int(value_only),
                "lr": lr, **lrn,
                **{k: diag.get(k, float("nan")) for k in
                   ("ev", "ev_q1", "ev_q2", "ev_q3", "ev_q4", "policy_ce")},
                "target_entropy": sum(s["entropy"] for s in st) / max(dec, 1),
                "kl_prior": sum(s["kl"] for s in st) / max(dec, 1),
                "search_differs": sum(s["differs"] for s in st) / max(dec, 1),
                "win_self": tab("self", 1) / max(tab("self", 0), 1),
                "win_frozen": tab("frozen", 1) / max(tab("frozen", 0), 1),
                "frozen_seats": int(tab("frozen", 0)),
                "sims_per_s": sims / play_s, "games_per_h": sum(s["games"] for s in st) / play_s
                * 3600, "decisions_per_s": dec / play_s, "rss_mb": rss_mb(),
                "worker_rss_mb": max(s.get("rss_mb", 0.0) for s in st),
                "replay_mb": replay.nbytes() / 2**20, "gen_s": gen_s, "learn_s": learn_s,
                "eval_s": eval_s, "elapsed_s": elapsed(),
                **{f"tables_{k}": tab(k, 3) / max(n_games, 1) for k in TABLE_KINDS},
                "win_heuristic": tab("heuristic", 1) / max(tab("heuristic", 0), 1),
                "heuristic_seats": int(tab("heuristic", 0)),
            }  # fmt: skip
            append_rows(out / "log.csv", COLUMNS, [row])
            print(
                f"g{completed} games={row['games']} samples={n_new} steps={steps} "
                f"{'(value-only) ' if value_only else ''}pi={row['loss_pi']:.3f} "
                f"v={row['loss_v']:.4f} ev={row['ev']:.3f} q1={row['ev_q1']:.3f} "
                f"H={row['target_entropy']:.3f} differs={row['search_differs']:.3f} "
                f"{row['sims_per_s']:.0f} sims/s {row['games_per_h']:.0f} games/h "
                f"rss={row['rss_mb']:.0f}+{row['worker_rss_mb']:.0f}MB {gen_s:.0f}s "
                f"tables s/f/h={row['tables_self']:.2f}/{row['tables_frozen']:.2f}/"
                f"{row['tables_heuristic']:.2f} win s/f/h={row['win_self']:.3f}/"
                f"{row['win_frozen']:.3f}/{row['win_heuristic']:.3f}", flush=True,
            )  # fmt: skip
            if args.ckpt_every > 0 and completed % args.ckpt_every == 0:
                checkpoint(completed)
    except AbortRollout:
        print("stop requested: in-flight generation abandoned", flush=True)
    except ForcedExit:
        print("second signal: aborting without a final checkpoint", flush=True)
        runner.kill()
        exit_code = 130
    except WorkerFailure as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if exit_code == 0 and completed > last_ckpt:
            checkpoint(completed)
        if exit_code == 0:
            runner.close()
        else:
            runner.kill()
        cleanup_own(os.getpid())
        sig.restore()
    print(f"stopped at generation {completed} after {elapsed() / 60:.1f} min", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
