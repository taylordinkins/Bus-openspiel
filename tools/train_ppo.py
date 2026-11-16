# pyright: basic, reportMissingImports=false
"""PPO self-play trainer for Bus (Phase 4 unit 3).

    uv run --group train python -m tools.train_ppo --name smoke --players 4 --updates 40 \\
        --games-per-update 128 --workers 11

One shared ego-rotated policy plays the "current" seats; by default ~50% of games are pure
self-play, ~30% put frozen past snapshots in some seats and ~20% put heuristic or random bots in
some seats (``--league 0.5,0.3,0.2``; ``--league 1,0,0`` is pure self-play). Only "current" seats
are trained on. Rollouts run in a spawn ``multiprocessing`` pool on CPU with a numpy copy of the
weights (``train.rollout``); the learner (``train.ppo``) runs on the GPU when available. Reward
is the shaped return ``mix * win_loss + (1 - mix) * margin`` with ``mix`` annealed linearly to 1.0;
the periodic evaluation is always pure win_loss against 3 random, 3 heuristic (optionally 3
``mcts:50``) bots and 3 copies of the oldest/latest pool snapshot, run on the worker pool.

Outputs in ``runs/<name>/``: ``log.csv`` (one row per update), ``eval.csv`` (one row per
evaluated opponent), ``pool/`` (league snapshots ``snap_<id>.npz``), ``ckpt_<update>.pt`` (model +
optimizer + config + RNG/run state) and ``ckpt_<update>.npz`` (numpy export for
``bots.net_forward``), ``latest.json``, ``best.pt/.npz/.json``, ``args.json``, ``resume_log.json``.
``--resume <run dir>`` continues a stopped run exactly (see ``docs/TRAINING.md``).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import random
import shutil
import signal
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

# Children inherit these at spawn (before numpy loads there): one BLAS thread per worker.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train.rollout as rollout_mod  # noqa: E402
from bots.net_forward import layout_hash  # noqa: E402
from train.atomic import atomic_write_json  # noqa: E402
from train.export import export_npz  # noqa: E402
from train.features import FEATURES_VERSION  # noqa: E402
from train.league import GAME_TYPES, League, parse_league  # noqa: E402
from train.model import BusNet  # noqa: E402
from train.ppo import (  # noqa: E402
    PPOConfig,
    ent_at,
    ev_by_quartile,
    explained_variance,
    lr_at,
    mix_at,
    net_to_weights,
    ppo_update,
)
from train.ratings import elo_ratings  # noqa: E402
from train.rollout import (  # noqa: E402
    END_TYPES,
    RolloutConfig,
    load_spilled,
    run_task,
    run_task_spill,
    worker_init,
)
from train.runstate import (  # noqa: E402
    ResumeError,
    append_resume_log,
    apply_retention,
    beats,
    explicit_options,
    find_latest_checkpoint,
    merge_resume_args,
    now_iso,
    read_eval_rows,
    remove_stray_tmp,
    rss_mb,
    save_pt,
    shortlist_insert,
    sync_best_files,
    truncate_csv,
    write_latest,
)
from train.shm import cleanup_own, cleanup_stale  # noqa: E402

from .tournament import wilson_interval  # noqa: E402

# Evaluation seeds live above this bit; training seeds (seed * 1_000_003 + update * 1009 + ...)
# stay far below it for any realistic seed and update count, so the two spaces never overlap.
EVAL_SEED_BIT = 1 << 62
CURRENT_LABEL = "current"  # the net being trained, in rating games
SIGNAL_DEBOUNCE_S = 2.0  # signals closer together than this count as one
SIGNAL_GRACE_S = 60.0  # after the first signal, an in-flight rollout is aborted after this long

COLUMNS = [
    "update", "games", "decisions", "samples", "decisions_per_s", "games_per_s", "env_s",
    "learn_s", "worker_busy", "entropy", "approx_kl", "clip_frac", "pg_loss", "value_loss",
    "explained_var", "ev_q1", "ev_q2", "ev_q3", "ev_q4", "mean_len", "illegal", "mix", "lr",
    "ent_coef",
    *[f"end_{t}" for t in END_TYPES],
    *[f"frac_{t}" for t in GAME_TYPES], *[f"win_{t}" for t in GAME_TYPES[1:]],
    *[f"ret_{t}" for t in GAME_TYPES[1:]], "pool_size", "hof_size",
    "eval_games", "eval_win", "eval_lo", "eval_hi", "eval_composite", "eval_s", "ratings_s",
    "rating_latest", "elapsed_s", "rss_mb",
]  # fmt: skip
EVAL_COLUMNS = ["update", "opponent", "players", "games", "win_rate", "lo", "hi", "mean_return"]
RATINGS_COLUMNS = ["update", "entrant", "kind", "rating", "games", "score_rate"]
DEFAULT_PLAYER_WEIGHTS: dict[tuple[int, ...], tuple[float, ...]] = {(3, 4, 5): (0.3, 0.4, 0.3)}
COMPOSITE_WEIGHTS = (0.8, 0.2)  # (vs 3 heuristic, vs the hall-of-fame reference net)


def parse_players(text: str) -> tuple[int, ...]:
    ps = tuple(int(x) for x in text.split(",") if x)
    if not ps or any(p not in (3, 4, 5) for p in ps):
        raise argparse.ArgumentTypeError("players must be a list from 3,4,5")
    return ps


def resolve_player_weights(players: tuple[int, ...], text: str | None) -> tuple[float, ...] | None:
    """Sampling weights over ``players``: the text (``0.3,0.4,0.3``), the default 0.3/0.4/0.3 for
    exactly 3,4,5, else None (uniform)."""
    if text:
        ws = tuple(float(x) for x in text.split(",") if x)
        if len(ws) != len(players) or any(w < 0 for w in ws) or sum(ws) <= 0:
            raise ValueError("--player-weights needs one non-negative weight per --players entry")
        return ws
    return DEFAULT_PLAYER_WEIGHTS.get(players)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PPO self-play trainer for Bus.")
    ap.add_argument("--name", default="run", help="output dir is <out-dir>/<name>")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--players", type=parse_players, default=(4,), help="e.g. 4 or 3,4,5")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--games-per-update", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop after this (0 = off)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--envs-per-worker", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="512,512")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lr-schedule", choices=("linear", "cosine"), default="cosine")
    ap.add_argument("--lr-final-frac", type=float, default=0.1, help="decay target, x --lr")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-coef-final", type=float, default=None, help="constant floor")
    ap.add_argument(
        "--ent-floor-at", type=float, default=0.5,
        help="fraction of --updates at which the entropy coefficient reaches its floor",
    )  # fmt: skip
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae-lambda", "--lam", dest="lam", type=float, default=0.99)
    ap.add_argument(
        "--value-trunk", choices=("separate", "shared"), default="separate",
        help="separate value network (export format 2) or the original shared trunk",
    )  # fmt: skip
    ap.add_argument(
        "--player-weights", default=None,
        help="sampling weights over --players (default 0.3,0.4,0.3 for 3,4,5; else uniform)",
    )  # fmt: skip
    ap.add_argument("--mix", type=float, default=0.7, help="initial win_loss share of the reward")
    ap.add_argument("--mix-anneal-updates", type=int, default=300, help="0 = keep --mix")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=10, help="0 = off")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-greedy", action="store_true", help="greedy instead of sampled eval")
    ap.add_argument(
        "--eval-mcts", action=argparse.BooleanOptionalAction, default=False,
        help="also evaluate vs 3 mcts:50 (slow)",
    )  # fmt: skip
    ap.add_argument(
        "--league", default="0.4,0.42,0.18",
        help="game mix self,snapshots,bots (1,0,0 = self-play); bots = ~5/6 heuristic, 1/6 random",
    )  # fmt: skip
    ap.add_argument("--snapshot-every", type=int, default=10, help="updates; 0 = no snapshots")
    ap.add_argument("--pool-size", type=int, default=10, help="snapshots kept (oldest evicted)")
    ap.add_argument("--snapshot-sampling", choices=("uniform", "recency"), default="uniform")
    ap.add_argument("--hof-every", type=int, default=100, help="milestone to hall of fame; 0 = off")
    ap.add_argument("--hof-cap", type=int, default=20, help="hall-of-fame size before thinning")
    ap.add_argument("--hof-share", type=float, default=0.5, help="snapshot seats from the hof")
    ap.add_argument(
        "--hof-seed", default="ppo-smoke",
        help="comma list of models/manifest.yaml nets added as permanent hall-of-fame entries",
    )  # fmt: skip
    ap.add_argument("--pfsp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pfsp-floor", type=float, default=0.05)
    ap.add_argument("--pfsp-ema", type=float, default=0.2, help="win-rate EMA step per update")
    ap.add_argument("--ratings-every", type=int, default=50, help="Elo round-robin; 0 = off")
    ap.add_argument("--ratings-games", type=int, default=300, help="games per rating round")
    ap.add_argument("--league-seed", type=int, default=0, help="extra seed for league sampling")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue a stopped run")
    ap.add_argument("--keep-every", type=int, default=250, help="milestone ckpt every N (0 = off)")
    ap.add_argument("--best-keep", type=int, default=5, help="top-K best checkpoints kept")
    ap.add_argument(
        "--allow-horizon-change", action="store_true",
        help="with --resume: accept a changed --updates (stretches the lr/entropy schedules)",
    )  # fmt: skip
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N updates this session")
    ap.add_argument("--task-timeout", type=float, default=1800.0, help="s per batch; 0 = off")
    ap.add_argument("--max-retries", type=int, default=3, help="pool rebuilds per batch")
    return ap


def _chunks(total: int, parts: int) -> list[int]:
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


class AbortRollout(Exception):
    """The in-flight batch was abandoned because a stop was requested and it ran too long."""


class WorkerFailure(Exception):
    """Workers kept dying or hanging after all retries."""


class ForcedExit(BaseException):
    """Second stop signal: unwind now (cleanup still runs)."""


class Runner:
    """Runs rollout tasks in a spawn process pool (or inline with 0 workers).

    A dead worker (``BrokenProcessPool``), a task error or a batch past ``timeout`` seconds is
    handled by killing the pool, removing this run's spill dirs, building a new pool and redoing
    the whole batch (tasks are pure functions of their seeds); after ``max_retries`` rebuilds it
    raises ``WorkerFailure``.
    """

    def __init__(self, workers: int, timeout: float = 0.0, max_retries: int = 3) -> None:
        self.workers = workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.pid = os.getpid()
        self.rebuilds = 0
        self.ex: ProcessPoolExecutor | None = None
        self._build()

    def _build(self) -> None:
        if self.workers > 0:
            self.ex = ProcessPoolExecutor(
                self.workers, mp_context=mp.get_context("spawn"),
                initializer=worker_init, initargs=(self.pid,),
            )  # fmt: skip

    def _kill(self) -> None:
        ex, self.ex = self.ex, None
        if ex is None:
            return
        for p in list((getattr(ex, "_processes", None) or {}).values()):
            with contextlib.suppress(Exception):
                p.kill()
        with contextlib.suppress(Exception):
            ex.shutdown(wait=False, cancel_futures=True)

    def _map_once(self, tasks: list[Any], abort: Callable[[], bool] | None) -> list[dict[str, Any]]:
        if self.ex is None:
            return [run_task(t) for t in tasks]
        futs = [self.ex.submit(run_task_spill, t) for t in tasks]
        deadline = time.monotonic() + self.timeout if self.timeout > 0 else None
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when="FIRST_COMPLETED")
            for f in done:
                f.result()  # raises BrokenProcessPool / the task's exception
            if abort is not None and abort():
                self._kill()
                cleanup_own(self.pid)
                raise AbortRollout
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"batch of {len(tasks)} tasks exceeded {self.timeout:.0f}s")
        return [load_spilled(f.result()) for f in futs]

    def map(
        self, tasks: list[Any], abort: Callable[[], bool] | None = None
    ) -> list[dict[str, Any]]:
        attempt = 0
        while True:
            try:
                return self._map_once(tasks, abort)
            except (BrokenProcessPool, TimeoutError, OSError, RuntimeError, ValueError) as e:
                attempt += 1
                print(
                    f"WARNING: rollout batch failed ({type(e).__name__}: {e}); "
                    f"rebuilding the worker pool (retry {attempt}/{self.max_retries})",
                    flush=True,
                )
                self._kill()
                cleanup_own(self.pid)
                if attempt > self.max_retries:
                    raise WorkerFailure(f"workers failed {attempt} times in a row") from e
                self.rebuilds += 1
                self._build()

    def close(self) -> None:
        if self.ex is not None:
            self.ex.shutdown(wait=True)
            self.ex = None

    def kill(self) -> None:
        self._kill()


def evaluate_all(
    runner: Runner, weights: Any, league: League | None, args: argparse.Namespace, update: int,
    abort: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """Pure win_loss evaluation of ``weights`` as ego (seat rotation) vs 3 copies of each opponent,
    separately for every player count in ``args.players`` (``--eval-games`` games per opponent and
    count).

    Opponents: random, heuristic, optionally mcts:50, the oldest/latest recent snapshot, and the
    hall-of-fame reference (its oldest entry, ``ppo-smoke`` by default). All run as one batch of
    tasks on the worker pool. Rows: opponent, players, games, win_rate (wl > 0), Wilson 95%
    interval, mean win_loss return.
    """
    specs: list[tuple[str, str]] = [("random", "random"), ("heuristic", "heuristic")]
    if args.eval_mcts:
        specs.append(("mcts:50", "mcts:50"))
    labels: list[str] = []
    if league is not None and len(league):
        ids = sorted({league.ids[0], league.ids[-1]})
        for i in ids:
            tag = "oldest" if i == league.ids[0] else "latest"
            specs.append((f"snap_{tag}:{i}", league.label(i)))
            if len(ids) == 1:  # a single snapshot is both
                specs[-1] = (f"snap_latest:{i}", league.label(i))
        labels += [league.label(i) for i in ids]
    if league is not None and league.hof_names:
        ref = league.hof_names[0]
        specs.append((f"hof_ref:{ref}", league.hof_label(ref)))
        labels.append(league.hof_label(ref))
    opp = league.opponents(labels) if league is not None and labels else {}
    counts = sorted(set(args.players))
    cells = [(k, n) for k in range(len(specs)) for n in counts]
    tasks: list[Any] = []
    owner: list[int] = []
    for ci, (k, n) in enumerate(cells):
        cfg = RolloutConfig(
            players=(n,), mix=1.0, envs_per_worker=args.envs_per_worker,
            greedy=args.eval_greedy, collect=False, eval_opponent=specs[k][1],
        )  # fmt: skip
        sizes = _chunks(args.eval_games, max(runner.workers, 1))
        start = 0
        for i, sz in enumerate(sizes):
            seed = (
                EVAL_SEED_BIT + ((args.seed * 100_003 + update) * 64 + k * 3 + (n - 3)) * 4096 + i
            )
            tasks.append((weights, sz, seed, cfg, list(range(start, start + sz)), start, opp))
            owner.append(ci)
            start += sz
    wins = [0.0] * len(cells)
    rets = [0.0] * len(cells)
    ns = [0] * len(cells)
    for ci, r in zip(owner, runner.map(tasks, abort), strict=True):
        for wl, _ in r["stats"]["ego_returns"]:
            ns[ci] += 1
            wins[ci] += 1.0 if wl > 0 else 0.0
            rets[ci] += wl
    rows = []
    for ci, (k, n) in enumerate(cells):
        lo, hi = wilson_interval(wins[ci], ns[ci])
        rows.append({
            "update": update, "opponent": specs[k][0], "players": n, "games": ns[ci],
            "win_rate": wins[ci] / max(ns[ci], 1), "lo": lo, "hi": hi,
            "mean_return": rets[ci] / max(ns[ci], 1),
        })  # fmt: skip
    return rows


def pooled(rows: list[dict[str, Any]], opponent: str) -> dict[str, Any] | None:
    """Rows of ``opponent`` pooled over player counts (games-weighted win rate, Wilson interval)."""
    rs = [r for r in rows if r["opponent"] == opponent]
    if not rs:
        return None
    n = sum(r["games"] for r in rs)
    w = sum(r["win_rate"] * r["games"] for r in rs)
    lo, hi = wilson_interval(w, n)
    return {"games": n, "win_rate": w / max(n, 1), "lo": lo, "hi": hi}


def composite_score(rows: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    """Fixed best-model criterion computed from one evaluation (see docs/TRAINING.md).

    ``0.8 * H + 0.2 * R`` where ``H`` is the mean over the evaluated player counts of the win rate
    against 3 heuristic seats and ``R`` the same against 3 copies of the hall-of-fame reference
    net; without a reference ``H`` alone is used. Each player count counts equally.
    """

    def mean_wr(match: Callable[[str], bool]) -> float | None:
        v = [r["win_rate"] for r in rows if match(r["opponent"])]
        return sum(v) / len(v) if v else None

    h = mean_wr(lambda o: o == "heuristic")
    ref = mean_wr(lambda o: o.startswith("hof_ref:"))
    if h is None:
        return -1.0, {}
    detail: dict[str, Any] = {"heuristic": h, "formula": "h-only"}
    if ref is None:
        return h, detail
    detail["hof_ref"] = ref
    detail["formula"] = "h+hof"
    return COMPOSITE_WEIGHTS[0] * h + COMPOSITE_WEIGHTS[1] * ref, detail


def rating_round(
    runner: Runner, weights: Any, league: League | None, args: argparse.Namespace, update: int,
    abort: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """Round-robin ratings: latest net, every hall-of-fame entry and the fixed bots.

    ``--ratings-games`` games; each draws a player count from ``--players`` and distinct entrants
    for the seats uniformly at random (repeats only if there are fewer entrants than seats). Final
    placements become Bradley-Terry / Elo ratings (``train.ratings``), heuristic = 1000. Rows for
    ``ratings.csv``.
    """
    hof = [league.hof_label(n) for n in league.hof_names] if league is not None else []
    entrants = [CURRENT_LABEL, *hof, "heuristic", "random"]
    rng = np.random.default_rng([args.seed, update, 77])
    seatings: list[list[str]] = []
    for _ in range(args.ratings_games):
        n = int(rng.choice(list(args.players)))
        pick = rng.choice(len(entrants), size=n, replace=len(entrants) < n)
        seatings.append([entrants[int(j)] for j in pick])
    opp = league.opponents(hof) if league is not None and hof else {}
    cfg = RolloutConfig(
        players=(4,), mix=1.0, envs_per_worker=args.envs_per_worker, collect=False,
        greedy=args.eval_greedy, record_scores=True,
    )  # fmt: skip
    tasks: list[Any] = []
    start = 0
    for i, sz in enumerate(_chunks(len(seatings), max(runner.workers, 1))):
        seed = EVAL_SEED_BIT + ((args.seed * 100_003 + update) * 64 + 63) * 4096 + i
        tasks.append((weights, sz, seed, cfg, None, 0, opp, seatings[start : start + sz]))
        start += sz
    games = [g for r in runner.map(tasks, abort) for g in r["stats"]["scores"]]
    res = elo_ratings(games, entrants, "heuristic")
    rows = []
    for e in entrants:
        kind = "latest" if e == CURRENT_LABEL else "hof" if e in hof else "bot"
        name = "latest" if e == CURRENT_LABEL else e
        rows.append({
            "update": update, "entrant": name, "kind": kind, "rating": res[e]["rating"],
            "games": int(res[e]["games"]), "score_rate": res[e]["score_rate"],
        })  # fmt: skip
    return rows


def format_eval_table(rows: list[dict[str, Any]]) -> str:
    """Compact pivot: one line per evaluated (update, player count), one column per opponent."""
    kinds: list[str] = []

    def kind_of(name: str) -> str:
        return name.split(":")[0] if name.startswith(("snap_", "hof_ref")) else name

    for r in rows:
        if kind_of(r["opponent"]) not in kinds:
            kinds.append(kind_of(r["opponent"]))
    lines = ["eval win rate [95% CI] (3 copies of the opponent, seats rotated):"]
    lines.append(f"{'update':>7}{'n':>3}" + "".join(f"{k:>24}" for k in kinds))
    for u, n in sorted({(r["update"], r["players"]) for r in rows}):
        cells = {
            kind_of(r["opponent"]): f"{r['win_rate']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]"
            for r in rows
            if r["update"] == u and r["players"] == n
        }
        lines.append(f"{u:>7}{n:>3}" + "".join(f"{cells.get(k, '-'):>24}" for k in kinds))
    return "\n".join(lines)


class StopSignals:
    """SIGINT/SIGTERM: the first sets a flag, the second unwinds immediately (``ForcedExit``)."""

    def __init__(self) -> None:
        self.count = 0
        self.first_at = 0.0
        self._old: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        now = time.monotonic()
        if self.count > 0 and now - self.first_at < SIGNAL_DEBOUNCE_S:
            return  # one Ctrl-C can arrive twice (terminal group + `uv run` forwarding it)
        self.count += 1
        if self.count == 1:
            self.first_at = now
            print(
                f"\nsignal {signum}: finishing the current update, then saving and exiting "
                "(signal again to abort now)",
                flush=True,
            )
        else:
            raise ForcedExit

    @property
    def requested(self) -> bool:
        return self.count > 0

    def overdue(self) -> bool:
        return self.count > 0 and time.monotonic() - self.first_at > SIGNAL_GRACE_S

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError):  # not the main thread
                self._old[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, old in self._old.items():
            signal.signal(sig, old)
        self._old.clear()


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(), "python": random.getstate(),
    }  # fmt: skip


def _restore_rng(st: dict[str, Any]) -> None:
    torch.set_rng_state(st["torch"].cpu())
    cuda = st.get("cuda")
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([x.cpu() for x in cuda])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def save_checkpoint(
    out: Path, update: int, model: BusNet, opt: torch.optim.Optimizer, args: argparse.Namespace,
    mix: float, games: int, pool_ids: list[int], state: dict[str, Any] | None = None,
) -> None:  # fmt: skip
    """Write ``ckpt_<update>.pt`` and ``.npz`` atomically, then ``latest.json``, then retention."""
    meta = {
        "update": update, "mix": mix, "games": games, "hidden": list(model.hidden),
        "features_version": FEATURES_VERSION, "players": list(args.players), "seed": args.seed,
        "pool_ids": pool_ids, "layout_hash": layout_hash(), "obs_size": model.obs_size,
        "value_trunk": "separate" if model.separate_value else "shared",
    }  # fmt: skip
    save_pt(
        out / f"ckpt_{update}.pt",
        {
            "model": model.state_dict(), "opt": opt.state_dict(), "update": update,
            "meta": meta, "args": {k: v for k, v in vars(args).items()}, "state": state,
        },
    )  # fmt: skip
    export_npz(model, out / f"ckpt_{update}.npz", meta)
    write_latest(out, update)
    apply_retention(out, keep_last=3, keep_every=args.keep_every)


def save_best(
    out: Path, update: int, model: BusNet, args: argparse.Namespace, rec: dict[str, Any]
) -> None:
    """Write ``best_<update>.pt/.npz`` (a shortlist member; retention never touches them)."""
    meta = {
        "update": update, "hidden": list(model.hidden), "features_version": FEATURES_VERSION,
        "layout_hash": layout_hash(), "obs_size": model.obs_size, "best": rec,
    }  # fmt: skip
    save_pt(
        out / f"best_{update}.pt", {"model": model.state_dict(), "update": update, "meta": meta}
    )
    export_npz(model, out / f"best_{update}.npz", meta)


def is_better(rec: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Best-model criterion: strictly higher ``composite`` within one formula; a record with a
    hall-of-fame reference (``h+hof``) outranks any without (``h-only``). See ``beats``."""
    return best is None or beats(rec, best)


def update_shortlist(
    out: Path, short: list[dict[str, Any]], rec: dict[str, Any], model: BusNet,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:  # fmt: skip
    """Admit ``rec`` to the top-``--best-keep`` shortlist if it ranks, saving its files, evicting
    the lowest and refreshing ``best.*`` / ``best_list.json``."""
    new, admitted, _ = shortlist_insert(short, rec, args.best_keep)
    if admitted:
        save_best(out, int(rec["update"]), model, args, rec)
        sync_best_files(out, new)
    return new


def append_rows(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, header)
        if new:
            w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def seed_hall_of_fame(league: League, names: str) -> None:
    """Add the manifest nets in ``names`` as permanent hall-of-fame entries (fresh runs only)."""
    from bots.net import load_weights

    for name in (x.strip() for x in names.split(",") if x.strip()):
        try:
            league.add_hof(name, load_weights(name), 0, permanent=True)
        except Exception as e:  # unknown / stale model: the run can still train without it
            print(f"WARNING: hall-of-fame seed {name!r} skipped: {e}", flush=True)


def prepare(argv: list[str]) -> tuple[argparse.Namespace, Path, dict[str, Any] | None]:
    """Parse arguments; for ``--resume`` merge the saved ``args.json`` and load the checkpoint.

    Returns (args, run dir, resume info or None). Raises ``ResumeError``.
    """
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.resume:
        out = Path(args.out_dir) / args.name
        if out.exists() and (
            (out / "latest.json").exists() or any(out.glob("ckpt_*.pt"))
        ):  # never wipe or mix into a run that has checkpoints
            raise ResumeError(
                f"{out} already holds a run; continue it with --resume {out} or pick another --name"
            )
        return args, out, None
    run = Path(args.resume)
    if not (run / "args.json").exists():
        raise ResumeError(f"{run} is not a run directory (no args.json)")
    saved = json.loads((run / "args.json").read_text())
    merged, changes = merge_resume_args(saved, explicit_options(ap, argv), vars(ap.parse_args([])))
    merged["resume"] = None
    merged["players"] = tuple(merged["players"])
    new_args = argparse.Namespace(**merged)
    found = find_latest_checkpoint(run)
    if found is None:
        raise ResumeError(f"no loadable checkpoint in {run}")
    update, ck = found
    meta = ck.get("meta", {})
    hidden = [int(x) for x in new_args.hidden.split(",") if x]
    problems = []
    if "hidden" in meta and list(meta["hidden"]) != hidden:
        problems.append(f"hidden {meta['hidden']} vs {hidden}")
    if meta.get("value_trunk", "shared") != new_args.value_trunk:
        problems.append(
            f"value_trunk {meta.get('value_trunk', 'shared')} vs {new_args.value_trunk}"
        )
    if "features_version" in meta and meta["features_version"] != FEATURES_VERSION:
        problems.append(f"features_version {meta['features_version']} vs {FEATURES_VERSION}")
    if "layout_hash" in meta and meta["layout_hash"] != layout_hash():
        problems.append("observation layout hash differs (obs size/layout changed)")
    if "seed" in meta and meta["seed"] != new_args.seed:
        problems.append(f"seed {meta['seed']} vs {new_args.seed}")
    if problems:
        raise ResumeError(
            "checkpoint is incompatible with this code/config: " + "; ".join(problems)
        )
    return new_args, run, {"update": update, "ckpt": ck, "changes": changes, "argv": argv}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args, out, res = prepare(argv)
    except ResumeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rollout_mod._OPP_CACHE.clear()  # in-process runs must not reuse another run's snapshot files
    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_name)
    model = BusNet(hidden=hidden, separate_value=args.value_trunk == "separate").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    pcfg = PPOConfig(
        lr=args.lr, lr_decay=args.lr_decay, clip=args.clip, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, ent_coef_final=args.ent_coef_final,
        max_grad_norm=args.max_grad_norm, epochs=args.epochs, minibatches=args.minibatches,
        target_kl=args.target_kl,
    )  # fmt: skip
    gen = torch.Generator().manual_seed(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    start_update = 0
    total_games = 0
    elapsed_base = 0.0
    best: list[dict[str, Any]] = []
    pool_ids: list[int] = []
    league_state: dict[str, Any] | None = None
    if res is not None:
        ck = res["ckpt"]
        start_update = int(res["update"])
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck.get("state") or {}
        if st:
            gen.set_state(st["gen"])
            _restore_rng(st["rng"])
            total_games = int(st["total_games"])
            elapsed_base = float(st["elapsed_s"])
            best = list(st.get("best_list") or ([st["best"]] if st.get("best") else []))
            league_state = st.get("league")
        else:
            print(
                "WARNING: checkpoint has no RNG state (pre-8a); resuming approximately", flush=True
            )
            total_games = int(ck["meta"].get("games", 0))
        pool_ids = list(ck["meta"].get("pool_ids", []))
        if not best and (out / "best.json").exists():  # run from before the shortlist
            best = [json.loads((out / "best.json").read_text())]
            top = int(best[0]["update"])
            for ext in ("pt", "npz"):
                if (out / f"best.{ext}").exists() and not (out / f"best_{top}.{ext}").exists():
                    shutil.copyfile(out / f"best.{ext}", out / f"best_{top}.{ext}")
        if best:  # drop files of records made after this checkpoint, re-point best.*
            sync_best_files(out, best)
        if not (out / "args.original.json").exists():
            atomic_write_json(
                out / "args.original.json", json.loads((out / "args.json").read_text())
            )
        append_resume_log(
            out,
            {"time": now_iso(), "from_update": start_update,
             "checkpoint": f"ckpt_{start_update}.pt", "overrides": res["changes"],
             "horizon_change": "updates" in res["changes"],
             "argv": res["argv"]},
        )  # fmt: skip
    atomic_write_json(out / "args.json", vars(args))
    remove_stray_tmp(out)
    stale = cleanup_stale()
    if stale:
        print(f"removed {len(stale)} stale rollout dir(s) of dead processes", flush=True)
    lprobs = parse_league(args.league)
    league: League | None = None
    if lprobs[0] < 1.0:
        league = League(
            lprobs, args.pool_size, args.snapshot_sampling, out / "pool", hof_cap=args.hof_cap,
            hof_share=args.hof_share, pfsp=args.pfsp, pfsp_floor=args.pfsp_floor,
            ema_alpha=args.pfsp_ema,
        )  # fmt: skip
        if res is not None:
            if league_state is not None:
                league.restore_state(league_state, max_id=start_update)
            else:  # checkpoint from before the hall of fame existed
                league.restore(pool_ids, max_id=start_update)
        else:
            seed_hall_of_fame(league, args.hof_seed)
    player_weights = resolve_player_weights(args.players, args.player_weights)
    dropped = truncate_csv(out / "log.csv", COLUMNS, start_update)
    truncate_csv(out / "ratings.csv", RATINGS_COLUMNS, start_update)
    edropped = truncate_csv(out / "eval.csv", EVAL_COLUMNS, start_update)
    eval_rows: list[dict[str, Any]] = read_eval_rows(out / "eval.csv") if res is not None else []
    if res is not None:
        print(
            f"resumed {out} at update {start_update} (dropped {dropped} log / {edropped} eval "
            f"rows beyond it); overrides: {res['changes'] or 'none'}",
            flush=True,
        )
    sig = StopSignals()
    sig.install()
    runner = Runner(args.workers, args.task_timeout, args.max_retries)
    session_t0 = time.perf_counter()
    completed = start_update
    last_ckpt = start_update
    session_updates = 0
    exit_code = 0
    forced = False
    print(f"device={device} workers={args.workers} hidden={hidden} out={out}", flush=True)

    def elapsed() -> float:
        return elapsed_base + time.perf_counter() - session_t0

    def checkpoint(update: int) -> None:
        nonlocal last_ckpt
        state = {
            "gen": gen.get_state(), "rng": _rng_state(), "total_games": total_games,
            "elapsed_s": elapsed(),
            "best": best[0] if best else None, "best_list": best,
            "league": league.state() if league is not None else None,
        }  # fmt: skip
        save_checkpoint(
            out, update, model, opt, args, mix_at(update - 1, args.mix, args.mix_anneal_updates),
            total_games, league.ids if league is not None else [], state,
        )  # fmt: skip
        last_ckpt = update

    try:
        with (
            open(out / "log.csv", "a", newline="") as fh,
            open(out / "eval.csv", "a", newline="") as efh,
        ):
            writer = csv.DictWriter(fh, COLUMNS)
            ewriter = csv.DictWriter(efh, EVAL_COLUMNS)
            try:
                for update in range(start_update + 1, args.updates + 1):
                    if sig.requested:
                        break
                    mix = mix_at(update - 1, args.mix, args.mix_anneal_updates)
                    lr = lr_at(
                        update - 1,
                        args.lr,
                        args.lr_decay,
                        args.lr_schedule,
                        args.updates,
                        args.lr_final_frac,
                    )
                    for g in opt.param_groups:
                        g["lr"] = lr
                    ent_coef = ent_at(
                        update - 1,
                        args.ent_coef,
                        args.ent_coef_final,
                        args.updates,
                        args.ent_floor_at,
                    )
                    rcfg = RolloutConfig(
                        players=args.players, gamma=args.gamma, lam=args.lam, mix=mix,
                        envs_per_worker=args.envs_per_worker,
                        league=league.spec() if league is not None else None,
                        player_weights=player_weights,
                    )  # fmt: skip
                    weights = net_to_weights(model)
                    te = time.perf_counter()
                    sizes = _chunks(args.games_per_update, max(args.workers, 1))
                    base = args.seed * 1_000_003 + update * 1009 + args.league_seed * 7_919_011
                    opp = league.opponents() if league is not None else {}
                    tasks = [
                        (weights, sz, base + i, rcfg, None, 0, opp) for i, sz in enumerate(sizes)
                    ]
                    results = runner.map(tasks, sig.overdue)
                    env_s = time.perf_counter() - te
                    del tasks

                    tl = time.perf_counter()
                    stats = [r["stats"] for r in results]
                    busy = sum(r["seconds"] for r in results)
                    parts = [r["samples"] for r in results if r["samples"] is not None]
                    del results
                    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                    del parts
                    n_samples = len(cat["action"])
                    ev = explained_variance(cat["value"], cat["vtarget"][:, 0])
                    ev_q = ev_by_quartile(cat["value"], cat["vtarget"][:, 0], cat["prog"])
                    data = {
                        k: torch.from_numpy(cat[k]).to(device)
                        for k in ("obs", "mask", "action", "logp", "adv", "vtarget", "vmask")
                    }
                    del cat  # on cuda the host copies are no longer needed
                    diag = ppo_update(model, opt, data, pcfg, ent_coef, gen)
                    del data
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    learn_s = time.perf_counter() - tl

                    games = sum(s["games"] for s in stats)
                    decisions = sum(s["decisions"] for s in stats)
                    lengths = [x for s in stats for x in s["lengths"]]
                    total_games += games
                    row: dict[str, Any] = {
                        "update": update, "games": total_games, "decisions": decisions,
                        "samples": n_samples, "decisions_per_s": decisions / max(env_s, 1e-9),
                        "games_per_s": games / max(env_s, 1e-9), "env_s": env_s,
                        "learn_s": learn_s,
                        "worker_busy": busy / max(env_s * max(args.workers, 1), 1e-9),
                        **diag, "explained_var": ev, "mean_len": float(np.mean(lengths)),
                        "illegal": sum(s["illegal"] for s in stats), "mix": mix, "lr": lr,
                        "ent_coef": ent_coef,
                    }  # fmt: skip
                    for t in END_TYPES:
                        row[f"end_{t}"] = sum(s["end"][t] for s in stats) / max(games, 1)
                    by_type: dict[str, list[float]] = {}
                    for s in stats:
                        for t, v in s["by_type"].items():
                            acc = by_type.setdefault(t, [0, 0, 0, 0.0])
                            for j in range(4):
                                acc[j] += v[j]
                    for t in GAME_TYPES:
                        g_, seats_, wins_, ret_ = by_type.get(t, [0, 0, 0, 0.0])
                        row[f"frac_{t}"] = g_ / max(games, 1)
                        if t != "self" and seats_:
                            row[f"win_{t}"] = wins_ / seats_
                            row[f"ret_{t}"] = ret_ / seats_
                    for qi, qv in enumerate(ev_q):
                        row[f"ev_q{qi + 1}"] = qv
                    by_label: dict[str, list[float]] = {}
                    for s in stats:
                        for lab, v in s["by_label"].items():
                            acc = by_label.setdefault(lab, [0.0, 0.0, 0.0])
                            for j in range(3):
                                acc[j] += v[j]
                    if league is not None:
                        league.update_results(by_label)
                    row["pool_size"] = len(league) if league is not None else 0
                    row["hof_size"] = len(league.hof_names) if league is not None else 0
                    new_w = None
                    snap_now = bool(
                        league is not None
                        and args.snapshot_every
                        and update % args.snapshot_every == 0
                    )
                    do_eval = bool(args.eval_every and update % args.eval_every == 0)
                    hof_now = bool(
                        league is not None and args.hof_every and update % args.hof_every == 0
                    )
                    do_ratings = bool(args.ratings_every and update % args.ratings_every == 0)
                    if snap_now or do_eval or hof_now or do_ratings:
                        new_w = net_to_weights(model)  # post-update weights
                    if do_eval:
                        te = time.perf_counter()
                        try:
                            erows = evaluate_all(runner, new_w, league, args, update, sig.overdue)
                        except (
                            AbortRollout
                        ):  # model is already updated: skip, don't lose the update
                            erows = []
                            print(f"  eval u{update} skipped (stop requested)", flush=True)
                        except WorkerFailure as e:
                            erows = []
                            exit_code = 1
                            print(f"  eval u{update} skipped: {e}", flush=True)
                        if erows:
                            row["eval_s"] = time.perf_counter() - te
                            first = erows[0]  # vs 3 random: the historical eval_* columns
                            row.update(
                                eval_games=first["games"], eval_win=first["win_rate"],
                                eval_lo=first["lo"], eval_hi=first["hi"],
                            )  # fmt: skip
                            for er in erows:
                                ewriter.writerow(er)
                            efh.flush()
                            eval_rows += erows
                            print(
                                f"  eval u{update} ({row['eval_s']:.1f}s): "
                                + " ".join(f"{e['opponent']}={e['win_rate']:.2f}" for e in erows),
                                flush=True,
                            )
                            comp, detail = composite_score(erows)
                            row["eval_composite"] = comp
                            heur = pooled(erows, "heuristic")
                            if heur is not None:
                                rec = {
                                    "update": update, "composite": comp, **detail,
                                    "lo": heur["lo"], "win_rate": heur["win_rate"],
                                    "games": heur["games"],
                                    "criterion": "0.8 * mean win rate vs 3 heuristic + "
                                    "0.2 * mean win rate vs hall-of-fame reference, "
                                    "means over player counts",
                                }  # fmt: skip
                                best = update_shortlist(out, best, rec, model, args)
                    if do_ratings and new_w is not None:
                        tr = time.perf_counter()
                        try:
                            rrows = rating_round(runner, new_w, league, args, update, sig.overdue)
                        except AbortRollout:
                            rrows = []
                            print(f"  ratings u{update} skipped (stop requested)", flush=True)
                        except WorkerFailure as e:
                            rrows = []
                            exit_code = 1
                            print(f"  ratings u{update} skipped: {e}", flush=True)
                        if rrows:
                            row["ratings_s"] = time.perf_counter() - tr
                            row["rating_latest"] = rrows[0]["rating"]
                            append_rows(out / "ratings.csv", RATINGS_COLUMNS, rrows)
                            print(
                                f"  ratings u{update} ({row['ratings_s']:.1f}s): "
                                + " ".join(f"{r['entrant']}={r['rating']:.0f}" for r in rrows),
                                flush=True,
                            )
                    if snap_now and league is not None and new_w is not None:
                        league.add(update, new_w)
                    if hof_now and league is not None and new_w is not None:
                        thinned = league.add_hof(f"m{update}", new_w, update)
                        if thinned:
                            print(f"  hall of fame: thinned {thinned}", flush=True)
                    row["elapsed_s"] = elapsed()
                    row["rss_mb"] = rss_mb()
                    writer.writerow({c: row.get(c, "") for c in COLUMNS})
                    fh.flush()
                    completed = update
                    session_updates += 1
                    print(
                        f"u{update} games={total_games} dec/s={row['decisions_per_s']:.0f} "
                        f"env={env_s:.1f}s learn={learn_s:.1f}s ent={row['entropy']:.3f} "
                        f"kl={row['approx_kl']:.4f} vl={row['value_loss']:.4f} ev={ev:.2f} "
                        f"mix={mix:.2f} types="
                        + "/".join(f"{row[f'frac_{t}']:.2f}" for t in GAME_TYPES)
                        + (f" eval={row['eval_win']:.2f}[{row['eval_lo']:.2f},{row['eval_hi']:.2f}]"
                           if "eval_win" in row else ""),
                        flush=True,
                    )  # fmt: skip
                    stop = (
                        sig.requested
                        or exit_code != 0
                        or (
                            bool(args.max_minutes)
                            and time.perf_counter() - session_t0 > args.max_minutes * 60
                        )
                        or bool(args.stop_after and session_updates >= args.stop_after)
                    )
                    if (
                        (args.ckpt_every and update % args.ckpt_every == 0)
                        or update == args.updates
                        or stop
                    ):
                        checkpoint(update)
                    if stop:
                        break
            except AbortRollout:
                print(f"rollout of update {completed + 1} aborted (stop requested)", flush=True)
            except WorkerFailure as e:
                exit_code = 1
                print(f"ERROR: {e}; saving a checkpoint and exiting", file=sys.stderr, flush=True)
            if completed > last_ckpt:  # stopped between checkpoints (abort, failure)
                checkpoint(completed)
    except ForcedExit:
        forced = True
        exit_code = 130
        print("second signal: aborted without a final checkpoint", flush=True)
    finally:
        if forced or exit_code != 0 or sig.requested:
            runner.kill()
        else:
            runner.close()
        cleanup_own(os.getpid())
        sig.restore()
    if eval_rows:
        print(format_eval_table(eval_rows), flush=True)
    if forced:
        return exit_code
    print(f"done: last checkpoint ckpt_{last_ckpt}", flush=True)
    if last_ckpt < args.updates:
        print(f"resume with: uv run python -m tools.train_ppo --resume {out}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
