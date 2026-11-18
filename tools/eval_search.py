"""Search evaluation: one candidate agent vs (n-1) baseline seats.

    uv run python -m tools.eval_search --candidates net,mcts:50 --baselines ppo_sample \\
        --players 3,4,5 --games 200 --out runs/eval_search/<name>

Plays seeded games in a spawn process pool and reports, per (candidate, baseline table, player
count) cell, the candidate's win share with a Wilson 95% interval against chance ``1/n``, its mean
score, the opponents' mean score and its mean decision latency (non-forced decisions only).

Agents (nets are loaded from a file path, not from the model manifest):

- ``net``: the ``--net`` model, greedy (argmax). ``net_sample``: the same, sampled at temperature 1.
- ``heuristic``: ``bots.heuristic.HeuristicBot``.
- ``mcts:<SIMS>[,c=<uct_c>][,t=<prior temp>][,fpu=<value>]``: PUCT search guided by the ``--net``
  model, the same search as ``bots.mcts_net`` (OpenSpiel Python ``MCTSBot``, net prior, net value
  head instead of rollouts, no Dirichlet noise, most-visited child played). ``c`` defaults to
  ``bots.mcts_net.UCT_C``; ``t`` (default 1) rescales the prior as ``p ** (1 / t)``; ``fpu`` is the
  value assumed for an unvisited child (default 0, OpenSpiel's behaviour).
- ``gumbel:<SIMS>[,c=<c_scale>][,v=<c_visit>][,m=<considered>][,t=<opp temp>][,e=<opp mix>]
  [,scope=opponents|nonroot]``: the ``bots.gumbel`` search (Gumbel top-m + sequential halving,
  completed-Q, per-seat values, g = 0) guided by the ``--net`` model; defaults are
  ``bots.gumbel.GumbelConfig`` (c 0.1, v 50, m 16, t 1, e 0, scope opponents). ``t`` / ``e``
  set the opponent model at non-root nodes (prior temperature, uniform share).

Baseline tables (the ``n - 1`` non-candidate seats; ``--baseline-net`` defaults to ``--net``):
``ppo_greedy``, ``ppo_sample``, ``heuristic`` and ``mixed`` (sampled net and heuristic seats
alternating, starting with the net, in seat order after the candidate).

Rotation: game ``g`` of a cell puts the candidate in seat ``g % n`` (the games count is rounded up
to a multiple of ``n``); the baseline seats follow it in seat order. Seeds: game ``g`` of a cell
uses ``tools.tournament.game_seed(cell_seed, g)`` with ``cell_seed`` depending on ``--seed``, the
player count and the baseline table only, so every candidate meets the same seeds (common random
numbers) and results do not depend on ``--workers``. Win accounting follows ``tools.tournament``:
a k-way shared win earns 1/k. ``distinct`` counts distinct (seat types, final scores, rounds)
outcomes, a cheap flag for deterministic tables that replay the same few games.

Outputs in ``--out``: ``games.jsonl`` (one row per game, appended as games finish; a rerun with
the same arguments skips games already there), ``summary.csv`` and ``summary.json``.
"""

# pyright: basic
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# One BLAS thread per process (spawned workers inherit this before numpy loads there); set before
# numpy is imported here too, else a single-process run spreads each forward over every core.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python.algorithms import mcts  # noqa: E402

from bots.gumbel import GumbelBot, GumbelConfig  # noqa: E402
from bots.heuristic import HeuristicBot  # noqa: E402
from bots.mcts import MAX_SIMS, _TimedMCTSBot  # noqa: E402
from bots.mcts_net import UCT_C, NetEvaluator  # noqa: E402
from bots.net import NetBot  # noqa: E402
from bots.net_forward import NetWeights  # noqa: E402
from bots.registry import BOTS  # noqa: E402
from tools.tournament import play_game, wilson_interval  # noqa: E402

BASELINES = ("ppo_greedy", "ppo_sample", "heuristic", "mixed")
DEFAULT_NET = Path("runs/long/best.npz")
SUMMARY_COLUMNS = [
    "candidate", "baseline", "players", "games", "win_share", "lo", "hi", "chance", "delta",
    "outright_wins", "mean_score", "mean_opp_score", "ms_per_decision", "decisions_per_game",
    "distinct",
]  # fmt: skip


# ---------------------------------------------------------------- agents


@dataclass(frozen=True)
class AgentSpec:
    kind: str  # "net" | "heuristic" | "mcts" | "gumbel"
    sample: bool = False
    sims: int = 0
    uct_c: float = UCT_C
    prior_temp: float = 1.0
    fpu: float = 0.0
    gumbel: GumbelConfig | None = None


def parse_agent(text: str) -> AgentSpec:
    """Parse a candidate agent spec (see the module docstring); ValueError if malformed."""
    text = text.strip()
    if text == "net":
        return AgentSpec("net")
    if text == "net_sample":
        return AgentSpec("net", sample=True)
    if text == "heuristic":
        return AgentSpec("heuristic")
    if text.startswith("mcts:"):
        sims_s, *opts = text[5:].split(",")
        if not sims_s.isdigit() or not 1 <= int(sims_s) <= MAX_SIMS:
            raise ValueError(f"mcts simulations must be 1..{MAX_SIMS}: {text!r}")
        kw: dict[str, float] = {}
        names = {"c": "uct_c", "t": "prior_temp", "fpu": "fpu"}
        for o in opts:
            k, eq, v = o.partition("=")
            if not eq or k not in names:
                raise ValueError(f"bad mcts option {o!r} (use c=, t=, fpu=)")
            kw[names[k]] = float(v)
        spec = AgentSpec(
            "mcts", sims=int(sims_s), uct_c=kw.get("uct_c", UCT_C),
            prior_temp=kw.get("prior_temp", 1.0), fpu=kw.get("fpu", 0.0),
        )  # fmt: skip
        if spec.prior_temp <= 0 or spec.uct_c < 0:
            raise ValueError(f"mcts needs t > 0 and c >= 0: {text!r}")
        return spec
    if text.startswith("gumbel:"):
        sims_s, *opts = text[7:].split(",")
        if not sims_s.isdigit() or not 1 <= int(sims_s) <= MAX_SIMS:
            raise ValueError(f"gumbel simulations must be 1..{MAX_SIMS}: {text!r}")
        gkw: dict[str, Any] = {}
        gnames = {
            "c": "c_scale", "v": "c_visit", "m": "m", "t": "opp_temp", "e": "opp_mix",
            "scope": "opp_scope",
        }  # fmt: skip
        for o in opts:
            k, eq, v = o.partition("=")
            if not eq or k not in gnames:
                raise ValueError(f"bad gumbel option {o!r} (use c=, v=, m=, t=, e=, scope=)")
            gkw[gnames[k]] = int(v) if k == "m" else v if k == "scope" else float(v)
        return AgentSpec("gumbel", sims=int(sims_s), gumbel=GumbelConfig(**gkw))
    raise ValueError(f"unknown agent {text!r}")


def baseline_specs(baseline: str, players: int) -> list[str]:
    """Agent specs of the ``players - 1`` baseline seats, in seat order after the candidate."""
    k = players - 1
    if baseline == "ppo_greedy":
        return ["net"] * k
    if baseline == "ppo_sample":
        return ["net_sample"] * k
    if baseline == "heuristic":
        return ["heuristic"] * k
    if baseline == "mixed":
        return ["net_sample" if j % 2 == 0 else "heuristic" for j in range(k)]
    raise ValueError(f"unknown baseline {baseline!r}")


class PathNetBot(NetBot):
    """``bots.net.NetBot`` on weights loaded from a file path."""

    def __init__(self, weights: NetWeights, sample: bool = False) -> None:  # noqa: D107
        self.name = "path"
        self.sample = sample
        self.weights = weights
        self._fallback = HeuristicBot()


class PathNetEvaluator(NetEvaluator):
    """``bots.mcts_net.NetEvaluator`` on given weights, with an optional prior temperature."""

    def __init__(self, weights: NetWeights, prior_temp: float = 1.0) -> None:  # noqa: D107
        self.weights = weights
        self.prior_temp = prior_temp
        self._cache: dict = {}

    def prior(self, state):
        pr = list(self._forward(state)[0])
        if self.prior_temp == 1.0:
            return pr
        p = np.array([x for _, x in pr], np.float64) ** (1.0 / self.prior_temp)
        p /= p.sum() if p.sum() > 0 else 1.0
        return [(a, float(q)) for (a, _), q in zip(pr, p, strict=True)]


def make_puct(fpu: float) -> Callable[[Any, int, float], float]:
    """PUCT child value as ``mcts.SearchNode.puct_value``, with ``fpu`` for unvisited children."""

    def puct(c: Any, parent_n: int, uct_c: float) -> float:
        if c.outcome is not None:
            return c.outcome[c.player]
        q = c.total_reward / c.explore_count if c.explore_count else fpu
        return q + uct_c * c.prior * math.sqrt(parent_n) / (c.explore_count + 1)

    return puct


class PathNetMCTSBot:
    """The ``bots.mcts_net.NetMCTSBot`` search on given weights, with tuning knobs."""

    def __init__(self, weights: NetWeights, spec: AgentSpec) -> None:  # noqa: D107
        self.weights = weights
        self.spec = spec
        self._fallback = PathNetBot(weights)
        self._select = mcts.SearchNode.puct_value if spec.fpu == 0.0 else make_puct(spec.fpu)

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = list(state.legal_actions())
        if len(legal) == 1:
            return legal[0]
        rs = np.random.RandomState(rng.randrange(2**31))
        bot = _TimedMCTSBot(
            state.get_game(), self.spec.uct_c, self.spec.sims,
            PathNetEvaluator(self.weights, self.spec.prior_temp), solve=False, random_state=rs,
            child_selection_fn=self._select, dirichlet_noise=None,
        )  # fmt: skip
        a = int(bot.step(state))
        return a if a in legal else self._fallback.choose(state, rng)


_WEIGHTS: dict[str, NetWeights] = {}


def _weights(path: str) -> NetWeights:
    if path not in _WEIGHTS:
        _WEIGHTS[path] = NetWeights.load(path)
    return _WEIGHTS[path]


def build_agent(text: str, net_path: str) -> Any:
    spec = parse_agent(text)
    if spec.kind == "heuristic":
        return HeuristicBot()
    if spec.kind == "net":
        return PathNetBot(_weights(net_path), spec.sample)
    if spec.kind == "gumbel":
        return GumbelBot(_weights(net_path), spec.sims, spec.gumbel)
    return PathNetMCTSBot(_weights(net_path), spec)


def seat_key(text: str, net_path: str) -> str:
    """Registry key of an agent; every key used by this tool starts with ``evs/``."""
    return f"evs/{text}@{net_path}"


def ensure_agent(key: str) -> None:
    """Register agent ``key`` in ``bots.registry.BOTS`` so ``tournament.play_game`` can seat it."""
    if key not in BOTS:
        text, _, path = key[4:].rpartition("@")
        BOTS[key] = build_agent(text, path)


# ---------------------------------------------------------------- jobs


def cell_seed(seed: int, players: int, baseline: str) -> int:
    return (seed * 10 + players) * 10 + BASELINES.index(baseline)


def round_games(games: int, players: int) -> int:
    return -(-games // players) * players


def seat_types(cand_key: str, opp_keys: Sequence[str], game_index: int) -> tuple[list[str], int]:
    """(seat types, candidate seat): the candidate sits at ``game_index % n``, opponents follow."""
    n = len(opp_keys) + 1
    cs = game_index % n
    types = [""] * n
    types[cs] = cand_key
    for j, k in enumerate(opp_keys):
        types[(cs + 1 + j) % n] = k
    return types, cs


def build_jobs(
    candidates: Sequence[str], baselines: Sequence[str], players: Sequence[int], games: int,
    seed: int, net: str, baseline_net: str,
) -> list[dict[str, Any]]:  # fmt: skip
    jobs = []
    for cand in candidates:
        parse_agent(cand)
        ck = seat_key(cand, net)
        for base in baselines:
            for n in players:
                opp = [seat_key(s, baseline_net) for s in baseline_specs(base, n)]
                cs_ = cell_seed(seed, n, base)
                for g in range(round_games(games, n)):
                    types, seat = seat_types(ck, opp, g)
                    jobs.append({
                        "candidate": cand, "baseline": base, "players": n, "game": g,
                        "cand_seat": seat, "types": types, "seed": cs_,
                    })  # fmt: skip
    return jobs


def job_id(j: dict[str, Any]) -> tuple[str, str, int, int]:
    return (j["candidate"], j["baseline"], j["players"], j["game"])


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Pool worker: play one game through ``tools.tournament.play_game``."""
    for k in job["types"]:
        ensure_agent(k)
    res = play_game((job["game"], job["seed"], job["types"], "win_loss", None))
    s = job["cand_seat"]
    return {
        "candidate": job["candidate"], "baseline": job["baseline"], "players": job["players"],
        "game": job["game"], "cand_seat": s, "seed": job["seed"],
        "seat_types": [t[4:] for t in job["types"]], "winners": res["winners"],
        "final_scores": res["final_scores"], "rounds": res["rounds"],
        "end_types": res["end_types"], "cand_seconds": res["decision_seconds"][s],
        "cand_decisions": res["decisions"][s],
    }  # fmt: skip


def job_cost(j: dict[str, Any]) -> float:
    spec = parse_agent(j["candidate"])
    return spec.sims * j["players"] if spec.kind in ("mcts", "gumbel") else j["players"]


def run_jobs(
    jobs: list[dict[str, Any]], workers: int, sink: Callable[[dict[str, Any]], None],
    progress: Callable[[str], None] | None = None,
) -> None:  # fmt: skip
    """Play ``jobs`` (heaviest first) and pass each finished row to ``sink`` in completion order."""
    jobs = sorted(jobs, key=job_cost, reverse=True)
    t0 = time.monotonic()
    every = max(1, len(jobs) // 20)

    def tick(i: int) -> None:
        if progress is not None and (i % every == 0 or i == len(jobs)):
            progress(f"  {i}/{len(jobs)} games, {time.monotonic() - t0:.0f}s")

    if workers <= 1:
        for i, j in enumerate(jobs, 1):
            sink(run_job(j))
            tick(i)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            sink(row)
            tick(i)


# ---------------------------------------------------------------- summary


def summarize(
    rows: Sequence[dict[str, Any]], candidates: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """One row per (candidate, baseline, players) cell, ordered by baseline, players, candidate."""
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault((r["candidate"], r["baseline"], r["players"]), []).append(r)
    out = []
    for (cand, base, n), rs in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        g = len(rs)
        credit = outright = score = opp = secs = 0.0
        decs = 0
        seen = set()
        for r in rs:
            s, sc, w = r["cand_seat"], r["final_scores"], r["winners"]
            if s in w:
                credit += 1 / len(w)
                outright += len(w) == 1
            score += sc[s]
            opp += (sum(sc) - sc[s]) / (n - 1)
            secs += r["cand_seconds"]
            decs += r["cand_decisions"]
            seen.add((tuple(r["seat_types"]), tuple(sc), r["rounds"]))
        lo, hi = wilson_interval(round(credit), g)
        out.append({
            "candidate": cand, "baseline": base, "players": n, "games": g,
            "win_share": credit / g, "lo": lo, "hi": hi, "chance": 1 / n,
            "delta": credit / g - 1 / n, "outright_wins": int(outright),
            "mean_score": score / g, "mean_opp_score": opp / g,
            "ms_per_decision": 1000 * secs / decs if decs else 0.0,
            "decisions_per_game": decs / g, "distinct": len(seen),
        })  # fmt: skip
    seen_c = [*(candidates or ()), *(r["candidate"] for r in rows)]
    order = {c: i for i, c in enumerate(dict.fromkeys(seen_c))}
    out.sort(key=lambda r: (r["baseline"], r["players"], order[r["candidate"]]))
    return out


def format_table(summary: Sequence[dict[str, Any]]) -> str:
    head = (
        f"{'candidate':26}{'baseline':12}{'n':>2}{'games':>6}{'win%':>7}{'lo95':>7}{'hi95':>7}"
        f"{'chance':>7}{'delta':>7}{'score':>7}{'opp':>7}{'ms/dec':>8}{'dec/g':>6}{'dist':>5}"
    )
    lines = [head]
    for r in summary:
        lines.append(
            f"{r['candidate']:26}{r['baseline']:12}{r['players']:>2}{r['games']:>6}"
            f"{100 * r['win_share']:>7.1f}{100 * r['lo']:>7.1f}{100 * r['hi']:>7.1f}"
            f"{100 * r['chance']:>7.1f}{100 * r['delta']:>+7.1f}{r['mean_score']:>7.2f}"
            f"{r['mean_opp_score']:>7.2f}{r['ms_per_decision']:>8.1f}"
            f"{r['decisions_per_game']:>6.1f}{r['distinct']:>5}"
        )
    return "\n".join(lines)


def _split(text: str) -> list[str]:
    return [s.strip() for s in text.split(";" if ";" in text else " ") if s.strip()]


def _print(line: str) -> None:
    print(line, flush=True)  # progress shows up promptly when stdout is a log file


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = _print) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.eval_search", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    ap.add_argument(
        "--candidates", default="net mcts:50 mcts:200",
        help="space- (or ';'-) separated candidate specs",
    )  # fmt: skip
    ap.add_argument("--baselines", default="ppo_greedy,ppo_sample", help="comma-separated tables")
    ap.add_argument("--players", default="3,4,5")
    ap.add_argument(
        "--games", type=int, default=200, help="per cell, rounded up to a multiple of n"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--net", default=str(DEFAULT_NET), help="candidate net .npz path")
    ap.add_argument("--baseline-net", default=None, help="baseline net .npz (default: --net)")
    ap.add_argument("--out", type=Path, default=Path("runs/eval_search/default"))
    ap.add_argument("--quiet", action="store_true", help="no progress lines")
    args = ap.parse_args(argv)
    try:
        cands = _split(args.candidates)
        for c in cands:
            parse_agent(c)
        bases = [b.strip() for b in args.baselines.split(",") if b.strip()]
        for b in bases:
            baseline_specs(b, 3)
        players = [int(p) for p in args.players.split(",")]
        if any(not 3 <= p <= 5 for p in players):
            raise ValueError("players must be 3..5")
        if args.games < 1 or args.workers < 1:
            raise ValueError("--games and --workers must be positive")
        net = str(args.net)
        bnet = str(args.baseline_net or args.net)
        for p in {net, bnet}:
            NetWeights.load(p)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "candidates": cands, "baselines": bases, "players": players, "games": args.games,
        "seed": args.seed, "net": net, "baseline_net": bnet, "workers": args.workers,
    }  # fmt: skip
    jobs = build_jobs(cands, bases, players, args.games, args.seed, net, bnet)
    games_path = args.out / "games.jsonl"
    rows: list[dict[str, Any]] = []
    if games_path.exists():
        want = {job_id(j): [t[4:] for t in j["types"]] for j in jobs}
        for line in games_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ok_seed = r["seed"] == cell_seed(args.seed, r["players"], r["baseline"])
            if want.get(job_id(r)) == r["seat_types"] and ok_seed:
                rows.append(r)
        done = {job_id(r) for r in rows}
        jobs = [j for j in jobs if job_id(j) not in done]
        if rows:
            out(f"resuming: {len(rows)} games already in {games_path}")
    out(f"eval_search: {len(jobs)} games to play, {args.workers} workers, out {args.out}")
    with open(games_path, "a", encoding="utf-8") as f:

        def sink(row: dict[str, Any]) -> None:
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

        run_jobs(jobs, args.workers, sink, None if args.quiet else out)
    summary = summarize(rows, cands)
    (args.out / "summary.json").write_text(
        json.dumps({"config": config, "summary": summary}, indent=1), encoding="utf-8"
    )
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(summary)
    out(format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
