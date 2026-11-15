"""Bot tournament: ``uv run python -m tools.tournament --players 4 --seats a,b,c,d``.

Plays seeded games between bot seat types (see ``bots/registry.py``) in a process pool and reports,
per seat type and per seat position, win share with a Wilson 95% interval, mean score, mean margin
and mean decision latency. No rule logic lives here.

Seat rotation: game ``g`` uses the seat list rotated by ``g % players``, so each rotation gets an
equal number of games and seat/start-player advantage averages out. (The game has no start-player
parameter, seat 0 always starts, D5, so rotating the types is what varies it.)

Win accounting: a sole winner earns 1 win; a k-way shared win earns 1/k of a win (tracked as
``shared_credit``, separately from ``outright_wins``). ``win_share`` is the total credit over
games; the Wilson interval is computed on ``round(credit)`` successes out of ``games`` trials.

Reproducibility: game ``g`` is seeded from ``(seed, g)`` only, never from the worker, so results
do not depend on ``--workers``. Wall-clock latency fields naturally vary between runs; with
``--think-time`` MCTS decisions also depend on timing.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyspiel

from bots.base import Bot
from bots.registry import is_valid_seat_type, make_bot
from bus import constants as C
from bus.game import BusState, game_result


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``wins`` successes in ``n`` trials, clipped to [0, 1]."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rotation_for(game_index: int, players: int) -> int:
    return game_index % players


def seat_types_for(seats: Sequence[str], game_index: int, rotate: bool) -> list[str]:
    """Seat types of game ``game_index``: seat i plays ``seats[(i + r) % n]``."""
    n = len(seats)
    r = rotation_for(game_index, n) if rotate else 0
    return [seats[(i + r) % n] for i in range(n)]


def game_seed(seed: int, game_index: int) -> int:
    return seed * 1_000_003 + game_index


def play_game(job: tuple[int, int, list[str], str, float | None]) -> dict[str, Any]:
    """Play one game (pool worker). job = (index, seed, seat_types, returns, think_time)."""
    index, seed, types, returns, think_time = job
    players = len(types)
    game = pyspiel.load_game("python_bus", {"players": players, "returns": returns})
    state = game.new_initial_state()
    assert isinstance(state, BusState)
    rng = random.Random(game_seed(seed, index))
    bots: list[Bot] = []
    for t in types:
        bot: Bot = make_bot(t)
        limiter: Any = getattr(bot, "with_time_limit", None)
        if think_time is not None and limiter is not None:
            bot = limiter(think_time)
        bots.append(bot)
    lat = [0.0] * players
    decisions = [0] * players
    while not state.is_terminal():
        seat = state.current_player()
        legal = state.legal_actions()
        if len(legal) == 1:
            state.apply_action(legal[0])
            continue
        t0 = time.perf_counter()
        action = bots[seat].choose(state.clone(), rng)
        lat[seat] += time.perf_counter() - t0
        decisions[seat] += 1
        state.apply_action(action)
    res = game_result(state)
    assert res is not None
    return {
        "game": index,
        "seat_types": types,
        "winners": res["winners"],
        "final_scores": res["final_scores"],
        "returns": res["returns"],
        "end_types": res["end_types"],
        "rounds": state.round,
        "decision_seconds": lat,
        "decisions": decisions,
    }


def _blank() -> dict[str, float]:
    return {
        "games": 0,
        "outright_wins": 0,
        "shared_credit": 0.0,
        "score": 0.0,
        "margin": 0.0,
        "seconds": 0.0,
        "decisions": 0,
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per seat type and per seat position summaries from per-game rows."""
    by_type: dict[str, dict[str, float]] = defaultdict(_blank)
    by_pos: dict[int, dict[str, float]] = defaultdict(_blank)
    for row in rows:
        scores = row["final_scores"]
        n = len(scores)
        k = len(row["winners"])
        for seat, t in enumerate(row["seat_types"]):
            others = (sum(scores) - scores[seat]) / (n - 1)
            for acc in (by_type[t], by_pos[seat]):
                acc["games"] += 1
                if seat in row["winners"]:
                    if k == 1:
                        acc["outright_wins"] += 1
                    else:
                        acc["shared_credit"] += 1 / k
                acc["score"] += scores[seat]
                acc["margin"] += scores[seat] - others
                acc["seconds"] += row["decision_seconds"][seat]
                acc["decisions"] += row["decisions"][seat]

    def finish(acc: dict[str, float]) -> dict[str, Any]:
        g = int(acc["games"])
        credit = acc["outright_wins"] + acc["shared_credit"]
        lo, hi = wilson_interval(round(credit), g)
        return {
            "games": g,
            "outright_wins": int(acc["outright_wins"]),
            "shared_credit": acc["shared_credit"],
            "win_share": credit / g if g else 0.0,
            "wilson_low": lo,
            "wilson_high": hi,
            "mean_score": acc["score"] / g if g else 0.0,
            "mean_margin": acc["margin"] / g if g else 0.0,
            "mean_decision_ms": (
                1000 * acc["seconds"] / acc["decisions"] if acc["decisions"] else 0.0
            ),
        }

    return {
        "by_seat_type": {t: finish(a) for t, a in sorted(by_type.items())},
        "by_position": {str(s): finish(a) for s, a in sorted(by_pos.items())},
    }


def run_tournament(
    seats: Sequence[str],
    games: int,
    seed: int = 0,
    workers: int = 1,
    rotate: bool = True,
    think_time: float | None = None,
    returns: str = "win_loss",
) -> dict[str, Any]:
    """Run the tournament; ``games`` must already be a multiple of len(seats) when rotating."""
    jobs = [(g, seed, seat_types_for(seats, g, rotate), returns, think_time) for g in range(games)]
    if workers <= 1:
        rows = [play_game(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            rows = pool.map(play_game, jobs, chunksize=1)
    end_mix: dict[str, int] = defaultdict(int)
    for r in rows:
        end_mix["+".join(r["end_types"])] += 1
    return {
        "config": {
            "players": len(seats),
            "seats": list(seats),
            "games": games,
            "seed": seed,
            "rotate": rotate,
            "think_time": think_time,
            "returns": returns,
            "workers": workers,
        },
        "summary": aggregate(rows),
        "mean_rounds": sum(r["rounds"] for r in rows) / max(len(rows), 1),
        "end_types": dict(sorted(end_mix.items())),
        "games": rows,
    }


def format_report(data: dict[str, Any]) -> str:
    cfg = data["config"]
    out = [
        f"tournament: {cfg['games']} games, {cfg['players']} players, seats "
        f"{','.join(cfg['seats'])}, seed {cfg['seed']}, rotate {cfg['rotate']}, "
        f"returns {cfg['returns']}",
        f"mean rounds {data['mean_rounds']:.1f}; end types "
        + ", ".join(f"{k} {v}" for k, v in data["end_types"].items()),
        "",
    ]
    head = (
        f"{'':22}{'games':>6}{'win%':>7}{'lo95':>7}{'hi95':>7}{'score':>7}"
        f"{'margin':>8}{'ms/dec':>9}"
    )

    def line(label: str, r: dict[str, Any]) -> str:
        return (
            f"{label:22}{r['games']:>6}{100 * r['win_share']:>7.1f}"
            f"{100 * r['wilson_low']:>7.1f}{100 * r['wilson_high']:>7.1f}"
            f"{r['mean_score']:>7.2f}{r['mean_margin']:>8.2f}{r['mean_decision_ms']:>9.2f}"
        )

    out.append("By seat type")
    out.append(head)
    out += [line(t, r) for t, r in data["summary"]["by_seat_type"].items()]
    out.append("")
    out.append("By seat position")
    out.append(head)
    out += [line(f"P{s}", r) for s, r in data["summary"]["by_position"].items()]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="tools.tournament", description=__doc__)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seats", required=True, help="comma-separated seat types, one per player")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--think-time", type=float, default=None, help="MCTS seconds per decision")
    ap.add_argument("--out", type=Path, help="write full JSON results here")
    ap.add_argument("--returns", choices=C.RETURNS_MODES, default=C.DEFAULT_RETURNS)
    args = ap.parse_args(argv)
    seats = [s.strip() for s in args.seats.split(",")]
    if not C.MIN_PLAYERS <= args.players <= C.MAX_PLAYERS:
        print(f"error: players must be {C.MIN_PLAYERS}..{C.MAX_PLAYERS}", file=sys.stderr)
        return 2
    if len(seats) != args.players:
        print(
            f"error: --seats lists {len(seats)} seat types but --players is {args.players}",
            file=sys.stderr,
        )
        return 2
    for s in seats:
        if s == "human" or not is_valid_seat_type(s):
            print(f"error: unknown or unusable seat type {s!r}", file=sys.stderr)
            return 2
    if args.games < 1 or args.workers < 1:
        print("error: --games and --workers must be positive", file=sys.stderr)
        return 2
    games = args.games
    if args.rotate and games % args.players:
        games = -(-games // args.players) * args.players
        out(f"note: --games rounded up from {args.games} to {games} (multiple of players)")
    data = run_tournament(
        seats, games, args.seed, args.workers, args.rotate, args.think_time, args.returns
    )
    out(format_report(data))
    if args.out:
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
