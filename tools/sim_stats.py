"""Random-play statistics: ``uv run python -m tools.sim_stats --games N --players 3,4,5 --seed S``.

Plays seeded uniform-random games through the engine API and reports end types, length, scores
and resource use per player count. No rule logic lives here.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyspiel

from bus import constants as C
from bus.actions import decode_action
from bus.board import load_board
from bus.game import BusState, game_result
from bus.rules import forced_action


def random_game(
    players: int, returns: str, seed: int
) -> tuple[BusState, list[int], float, int, int]:
    """Play one random game; returns (final state, history, mean branching, moves, deliveries)."""
    game = pyspiel.load_game("python_bus", {"players": players, "returns": returns})
    state = game.new_initial_state()
    assert isinstance(state, BusState)
    rng = random.Random(seed)
    history: list[int] = []
    branching = 0
    deliveries = 0
    while not state.is_terminal():
        legal = state.legal_actions()
        branching += len(legal)
        forced = forced_action(state)
        action = forced if forced is not None else rng.choice(legal)
        if decode_action(action)[0] == "deliver_to":
            deliveries += 1
        history.append(action)
        state.apply_action(action)
    return state, history, branching / max(len(history), 1), deliveries, len(history)


def _dist(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def run_players(games: int, players: int, seed: int, returns: str = "win_loss") -> dict[str, Any]:
    """Statistics for ``games`` random games of one player count."""
    max_len = 0
    end_mix: Counter[str] = Counter()
    rounds: list[int] = []
    lengths: list[int] = []
    all_scores: list[int] = []
    win_scores: Counter[int] = Counter()
    n_winners: Counter[int] = Counter()
    seat_wins = [0.0] * players
    seat_points = [0.0] * players
    lm_used = [0.0] * players
    lm_full = [0] * players
    action_left: list[int] = []
    stones: list[int] = []
    buildings: list[int] = []
    delivered: list[int] = []
    branching: list[float] = []
    times: list[float] = []
    sites_full_games = 0
    for g in range(games):
        t0 = time.perf_counter()
        state, _hist, br, deliv, n = random_game(players, returns, seed * 1_000_003 + g)
        times.append(time.perf_counter() - t0)
        res = game_result(state)
        assert res is not None
        max_len = state.get_game().max_game_length()
        end_mix["+".join(res["end_types"])] += 1
        if "sites_full" in res["end_types"]:
            sites_full_games += 1
        rounds.append(state.round)
        lengths.append(n)
        winners: list[int] = res["winners"]
        n_winners[len(winners)] += 1
        for w in winners:
            seat_wins[w] += 1 / len(winners)
            win_scores[res["final_scores"][w]] += 1
        all_scores.extend(res["final_scores"])
        for i in range(players):
            s = state.seat(i)
            seat_points[i] += res["final_scores"][i]
            used = C.LINE_MARKERS_PER_PLAYER - s.line_markers
            lm_used[i] += used
            lm_full[i] += used >= C.LINE_MARKERS_PER_PLAYER
        action_left.append(sum(state.seat(i).action_markers for i in range(players)))
        stones.append(sum(state.seat(i).stones for i in range(players)))
        # Buildings have no owner (SCENARIOS §9.6): count built sites on the board.
        n_sites = len(load_board().sites)
        buildings.append(sum(state.site_building(t) is not None for t in range(n_sites)))
        delivered.append(deliv)
        branching.append(br)
    d = max(games, 1)
    return {
        "players": players,
        "games": games,
        "end_types": dict(sorted(end_mix.items())),
        "rounds": _dist(rounds),
        "moves": {
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "max": max(lengths, default=0),
        },
        "max_game_length": max_len,
        "final_scores": {
            "mean": statistics.fmean(all_scores) if all_scores else 0.0,
            "max": max(all_scores, default=0),
        },
        "winner_score_dist": {str(k): v for k, v in sorted(win_scores.items())},
        "winner_count_dist": {str(k): v for k, v in sorted(n_winners.items())},
        "seat_win_rate": [w / d for w in seat_wins],
        "seat_mean_score": [p / d for p in seat_points],
        "line_markers_used_mean": [u / d for u in lm_used],
        "line_markers_hit_25_rate": [f / d for f in lm_full],
        "action_markers_left_mean": statistics.fmean(action_left) if action_left else 0.0,
        "stones_taken_mean": statistics.fmean(stones) if stones else 0.0,
        "buildings_built_mean": statistics.fmean(buildings) if buildings else 0.0,
        "sites_full_rate": sites_full_games / d,
        "passengers_delivered_mean": statistics.fmean(delivered) if delivered else 0.0,
        "branching_mean": statistics.fmean(branching) if branching else 0.0,
        "seconds_per_game": statistics.fmean(times) if times else 0.0,
    }


def run(
    games: int, players: Sequence[int] = (3, 4, 5), seed: int = 0, returns: str = "win_loss"
) -> dict[str, Any]:
    """Statistics for each player count; keys of ``by_players`` are the counts as strings."""
    return {
        "games": games,
        "seed": seed,
        "returns": returns,
        "by_players": {str(p): run_players(games, p, seed, returns) for p in players},
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / max(total, 1):.1f}%"


def _row(values: Sequence[float], fmt: str = "{:.2f}") -> str:
    return " ".join(f"P{i}={fmt.format(v)}" for i, v in enumerate(values))


def format_report(data: dict[str, Any]) -> str:
    """Compact text report."""
    out = [
        f"sim_stats: {data['games']} games per count, seed {data['seed']}, "
        f"returns {data['returns']}"
    ]
    for r in data["by_players"].values():
        n = r["games"]
        out.append("")
        out.append(f"== {r['players']} players ({n} games) ==")
        out.append(
            "end types: " + ", ".join(f"{k} {v} ({_pct(v, n)})" for k, v in r["end_types"].items())
        )
        ro = r["rounds"]
        out.append(
            f"rounds: mean {ro['mean']:.1f} min {ro['min']:.0f} "
            f"median {ro['median']:.0f} max {ro['max']:.0f}"
        )
        out.append(
            f"moves: mean {r['moves']['mean']:.0f} max {r['moves']['max']} "
            f"(max_game_length {r['max_game_length']})"
        )
        out.append(
            f"final scores: mean {r['final_scores']['mean']:.2f} max {r['final_scores']['max']}"
        )
        out.append(
            "winner score: " + ", ".join(f"{k}:{v}" for k, v in r["winner_score_dist"].items())
        )
        out.append(
            "winners per game: " + ", ".join(f"{k}:{v}" for k, v in r["winner_count_dist"].items())
        )
        out.append("seat win rate: " + _row(r["seat_win_rate"], "{:.3f}"))
        out.append("seat mean score: " + _row(r["seat_mean_score"]))
        out.append("line markers used (mean): " + _row(r["line_markers_used_mean"], "{:.1f}"))
        out.append("line markers hit 25: " + _row(r["line_markers_hit_25_rate"], "{:.3f}"))
        out.append(f"action markers left (all seats): {r['action_markers_left_mean']:.1f}")
        out.append(f"stones taken: {r['stones_taken_mean']:.2f}")
        out.append(
            f"buildings built: {r['buildings_built_mean']:.1f}; sites full at end: "
            f"{_pct(round(r['sites_full_rate'] * n), n)}"
        )
        out.append(f"passengers delivered: {r['passengers_delivered_mean']:.1f}")
        out.append(f"branching factor (mean legal actions): {r['branching_mean']:.2f}")
        out.append(f"time per game: {r['seconds_per_game']:.2f}s")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="tools.sim_stats", description=__doc__)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--players", default="3,4,5", help="comma-separated counts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--returns", choices=C.RETURNS_MODES, default=C.DEFAULT_RETURNS)
    ap.add_argument("--json", type=Path, help="also write the raw statistics here")
    args = ap.parse_args(argv)
    try:
        counts = [int(p) for p in args.players.split(",") if p.strip()]
        data = run(args.games, counts, args.seed, args.returns)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out(format_report(data))
    if args.json:
        args.json.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
