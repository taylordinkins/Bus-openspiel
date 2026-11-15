"""Replay a saved game: ``uv run python -m tools.replay <file.json> [--steps] [--compact]``.

The file is a scenario document (docs/SCENARIOS.md) with its ``history`` list (PLAN D11).
This is a thin wrapper over ``bus.scenario``; it holds no rule logic.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bus.game import BusState, game_result
from bus.scenario import ScenarioError, load_scenario


def format_result(state: BusState) -> list[str]:
    """Lines describing the game result, or the unfinished position."""
    res = game_result(state)
    if res is None:
        return [f"game not finished: round {state.round} phase {state.phase}"]
    return [
        "end types: " + ", ".join(res["end_types"]),
        "winners: " + ", ".join(f"P{w}" for w in res["winners"]),
        "final scores: " + ", ".join(f"P{i}={s}" for i, s in enumerate(res["final_scores"])),
        "returns: " + ", ".join(f"P{i}={r:g}" for i, r in enumerate(res["returns"])),
    ]


def compact_line(state: BusState) -> str:
    """One-line summary of a state."""
    if state.is_terminal():
        return f"round {state.round} END"
    return f"round {state.round} phase {state.phase} to move P{state.current_player()}"


def replay(
    doc: dict[str, Any],
    *,
    steps: bool = False,
    compact: bool = False,
    out: Callable[[str], None] = print,
) -> BusState:
    """Apply the document's history; raises ``ScenarioError`` (S-HISTORY) on an illegal id."""
    final = load_scenario(doc)  # validates the whole history
    if steps:
        base = copy.deepcopy(doc)
        history: list[int] = list(base.pop("history", []))
        state = load_scenario(base)
        for n, action in enumerate(history, 1):
            label = state.action_to_string(state.current_player(), action)
            state.apply_action(action)
            out(f"move {n}: {label}")
            if not compact:
                out(str(state))
            else:
                out(compact_line(state))
    return final


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="tools.replay", description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--steps", action="store_true", help="print every move and the state after it")
    ap.add_argument("--compact", action="store_true", help="with --steps, one line per state")
    args = ap.parse_args(argv)
    try:
        doc = json.loads(args.file.read_text(encoding="utf-8"))
        state = replay(doc, steps=args.steps, compact=args.compact, out=out)
    except ScenarioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    out(str(state))
    for line in format_result(state):
        out(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
