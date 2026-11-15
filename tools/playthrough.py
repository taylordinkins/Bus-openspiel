"""Golden playthroughs: ``uv run python -m tools.playthrough --regen`` rewrites tests/playthroughs/.

Each file records parameters, the action history, per move the action label and the observer
string after the move, and the final game result. ``check`` replays a file and names the first
differing move.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyspiel

from bus.game import BusState, game_result
from bus.rules import forced_action

DIR = Path(__file__).resolve().parent.parent / "tests" / "playthroughs"
SEEDS: tuple[tuple[int, int], ...] = ((3, 1), (4, 2), (5, 3))  # (players, seed)
RETURNS = "win_loss"


def _new(players: int, returns: str) -> BusState:
    game = pyspiel.load_game("python_bus", {"players": players, "returns": returns})
    state = game.new_initial_state()
    assert isinstance(state, BusState)
    return state


def record(players: int, seed: int, returns: str = RETURNS) -> dict[str, Any]:
    """Play a seeded random game and record it."""
    rng = random.Random(seed)
    state = _new(players, returns)
    moves: list[dict[str, Any]] = []
    while not state.is_terminal():
        seat = state.current_player()
        forced = forced_action(state)
        action = forced if forced is not None else rng.choice(state.legal_actions())
        label = state.action_to_string(seat, action)
        state.apply_action(action)
        moves.append({"action": action, "player": seat, "label": label, "state": str(state)})
    return {
        "format": 1,
        "parameters": {"players": players, "returns": returns},
        "seed": seed,
        "moves": moves,
        "result": game_result(state),
    }


def check(doc: dict[str, Any]) -> str | None:
    """Replay a golden document; return a description of the first difference, or None."""
    p = doc["parameters"]
    state = _new(p["players"], p["returns"])
    for n, mv in enumerate(doc["moves"], 1):
        seat = state.current_player()
        if seat != mv["player"]:
            return f"move {n}: player {seat}, expected {mv['player']}"
        if mv["action"] not in state.legal_actions():
            return f"move {n}: action {mv['action']} no longer legal"
        label = state.action_to_string(seat, mv["action"])
        if label != mv["label"]:
            return f"move {n}: label {label!r}, expected {mv['label']!r}"
        state.apply_action(mv["action"])
        if str(state) != mv["state"]:
            return f"move {n} ({label}): observer string differs"
    if not state.is_terminal():
        return f"game not finished after {len(doc['moves'])} moves"
    if game_result(state) != doc["result"]:
        return f"final result {game_result(state)} != {doc['result']}"
    return None


def path_for(players: int, seed: int) -> Path:
    return DIR / f"{players}p_seed{seed}.json"


def regen() -> list[Path]:
    DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for players, seed in SEEDS:
        f = path_for(players, seed)
        f.write_text(json.dumps(record(players, seed), indent=1) + "\n", encoding="utf-8")
        paths.append(f)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.playthrough", description=__doc__)
    ap.add_argument("--regen", action="store_true", help="rewrite the golden files")
    args = ap.parse_args(argv)
    if not args.regen:
        ap.error("nothing to do; pass --regen")
    for f in regen():
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
