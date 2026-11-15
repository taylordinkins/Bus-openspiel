"""Play Bus in the terminal: ``uv run python -m tools.play_text --players 4 --seed 7 [--human 0]``.

Seats not listed in ``--human`` are random bots seeded by ``--seed``. Thin wrapper over the
engine API; no rule logic. All I/O is injectable (``input_fn`` / ``out``) for tests.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyspiel

from bus.game import BusState
from bus.rules import forced_action
from bus.scenario import load_scenario
from tools.replay import format_result

HELP = "Enter a number from the list, #<action id>, u (undo last human move) or q (quit)."


def parse_humans(values: Sequence[str], players: int) -> set[int]:
    """Parse repeated / comma-separated seat lists (seats numbered from 0)."""
    seats: set[int] = set()
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                seat = int(part)
                if not 0 <= seat < players:
                    raise ValueError(f"seat {seat} out of range 0..{players - 1}")
                seats.add(seat)
    return seats


def fresh_document(players: int, returns: str, history: Sequence[int]) -> dict[str, Any]:
    """Replay file: a fresh-game scenario plus the action history (PLAN D11)."""
    return {
        "format": 1,
        "parameters": {"players": players, "returns": returns},
        "history": list(history),
    }


def _build(players: int, returns: str, history: Sequence[int]) -> BusState:
    return load_scenario(fresh_document(players, returns, history))


def play(
    *,
    players: int = 4,
    seed: int = 0,
    humans: set[int] | None = None,
    returns: str = "win_loss",
    autoforce: bool = True,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> tuple[BusState, list[int], bool]:
    """Play one game; returns (final state, history, quit_flag)."""
    humans = humans or set()
    rng = random.Random(seed)
    history: list[int] = []
    marks: list[int] = []  # history length before each human move (for undo)
    state = _build(players, returns, history)
    while not state.is_terminal():
        seat = state.current_player()
        legal = state.legal_actions()
        forced = forced_action(state)
        if seat not in humans:
            action = forced if forced is not None else rng.choice(legal)
        elif forced is not None and autoforce:
            action = forced
            out(f"P{seat} (forced): {state.action_to_string(seat, action)}")
        else:
            out(str(state))
            out(f"P{seat} to move:")
            for i, a in enumerate(legal):
                out(f"  {i}) {state.action_to_string(seat, a)}  [#{a}]")
            choice = _prompt(input_fn, out, legal)
            if choice == "q":
                return state, history, True
            if choice == "u":
                if not marks:
                    out("nothing to undo")
                else:
                    del history[marks.pop() :]
                    state = _build(players, returns, history)
                continue
            action = int(choice)
            marks.append(len(history))
        history.append(action)
        state.apply_action(action)
    return state, history, False


def _prompt(
    input_fn: Callable[[str], str], out: Callable[[str], None], legal: list[int]
) -> int | str:
    while True:
        try:
            raw = input_fn("> ").strip().lower()
        except EOFError:
            return "q"
        if raw in ("q", "u"):
            return raw
        try:
            if raw.startswith("#"):
                a = int(raw[1:])
                if a in legal:
                    return a
            else:
                i = int(raw)
                if 0 <= i < len(legal):
                    return legal[i]
        except ValueError:
            pass
        out(f"invalid entry {raw!r}. {HELP}")


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    ap = argparse.ArgumentParser(prog="tools.play_text", description=__doc__)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human", action="append", default=[], help="human seat(s), from 0")
    ap.add_argument("--returns", choices=("win_loss", "margin"), default="win_loss")
    ap.add_argument("--no-autoforce", action="store_true", help="prompt on forced moves too")
    ap.add_argument("--save", type=Path, help="write the replay file (scenario + history)")
    args = ap.parse_args(argv)
    try:
        humans = parse_humans(args.human, args.players)
        state, history, quit_flag = play(
            players=args.players,
            seed=args.seed,
            humans=humans,
            returns=args.returns,
            autoforce=not args.no_autoforce,
            input_fn=input_fn,
            out=out,
        )
    except (ValueError, pyspiel.SpielError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.save:
        doc = fresh_document(args.players, args.returns, history)
        args.save.write_text(json.dumps(doc), encoding="utf-8")
        out(f"saved {len(history)} moves to {args.save}")
    if quit_flag:
        out("quit")
        return 0
    out(str(state))
    for line in format_result(state):
        out(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
