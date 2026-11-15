"""Speed benchmark: ``uv run python -m tools.bench_clone [--games N] [--clones N]``.

Reports state.clone() time (mid-game state) and random-rollout games per second for 4 players.
"""

from __future__ import annotations

import argparse
import random
import time

import pyspiel

from bus.game import BusState
from bus.rules import forced_action


def _new_state(players: int) -> BusState:
    game = pyspiel.load_game("python_bus", {"players": players})
    state = game.new_initial_state()
    assert isinstance(state, BusState)
    return state


def _step(state: BusState, rng: random.Random) -> None:
    forced = forced_action(state)
    state.apply_action(forced if forced is not None else rng.choice(state.legal_actions()))


def bench_clone(players: int, clones: int) -> float:
    """Mean microseconds per clone of a state about 150 moves into a random game."""
    rng = random.Random(1)
    state = _new_state(players)
    for _ in range(150):
        if state.is_terminal():
            break
        _step(state, rng)
    start = time.perf_counter()
    for _ in range(clones):
        state.clone()
    return (time.perf_counter() - start) / clones * 1e6


def bench_rollouts(players: int, games: int) -> float:
    """Random games completed per second."""
    rng = random.Random(2)
    start = time.perf_counter()
    for _ in range(games):
        state = _new_state(players)
        while not state.is_terminal():
            _step(state, rng)
    return games / (time.perf_counter() - start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--clones", type=int, default=5000)
    args = ap.parse_args()
    print(f"clone: {bench_clone(args.players, args.clones):.1f} us")
    print(f"rollouts: {bench_rollouts(args.players, args.games):.2f} games/s")


if __name__ == "__main__":
    main()
