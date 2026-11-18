"""Let an agent play Bus against bots, one decision per command, state kept in a JSON file.

    uv run python -m tools.agent_play new --players 4 --agent 0 --bots ppo-long --seed 7 \
        --out g.json
    uv run python -m tools.agent_play show --game g.json
    uv run python -m tools.agent_play act 150 --game g.json --note "why"
    uv run python -m tools.agent_play notes --game g.json

Thin wrapper over the engine and bot registry; no rule logic. The file stores config plus the full
action history (bots' actions included, never re-queried), and the state is rebuilt by replaying
it. A bot decision is seeded from ``(seed, history length)`` so it does not depend on call order.
Forced moves (a single legal action) are applied automatically for every seat. At game end a
replay file ``<game>.replay.json`` (``tools.replay`` / web import format) is written.
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

from bots.net import check_players
from bots.registry import KINDS, is_valid_seat_type, make_bot, seat_type_error
from bus.game import BusState
from bus.scenario import load_scenario
from tools.play_text import fresh_document, parse_humans
from tools.replay import format_result
from train.atomic import atomic_write_json

FORMAT = 1


class AgentPlayError(Exception):
    """A user-facing error (bad arguments or an illegal action)."""


def normalize_bot_spec(spec: str) -> str:
    """Accept a registry seat type (``heuristic``, ``net:ppo-long``) or a bare model name."""
    spec = spec.strip()
    kind = spec.partition(":")[0]
    if kind in KINDS:
        return spec
    return f"net:{spec}"


def expand_bots(bots: str, bot_seats: Sequence[int]) -> dict[int, str]:
    parts = [normalize_bot_spec(p) for p in bots.split(",") if p.strip()]
    if len(parts) == 1:
        parts = parts * len(bot_seats)
    if len(parts) != len(bot_seats):
        raise AgentPlayError(f"--bots needs 1 or {len(bot_seats)} specs, got {len(parts)}")
    return dict(zip(bot_seats, parts, strict=True))


def build_state(cfg: dict[str, Any], history: Sequence[int]) -> BusState:
    return load_scenario(fresh_document(cfg["players"], cfg["returns"], history))


def bot_rng(seed: int, n_moves: int) -> random.Random:
    return random.Random(seed * 1_000_003 + n_moves)


def load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AgentPlayError(f"cannot read {path}: {e}") from None
    if data.get("format") != FORMAT:
        raise AgentPlayError(f"{path} is not an agent_play game file")
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    atomic_write_json(path, data, indent=None)


def replay_path(path: Path) -> Path:
    return path.with_name(path.stem + ".replay.json")


def advance(data: dict[str, Any]) -> BusState:
    """Apply bot and forced moves until an agent decision or game end; mutates ``data``."""
    cfg = data["config"]
    agents = set(cfg["agents"])
    state = build_state(cfg, data["history"])
    while not state.is_terminal():
        seat = state.current_player()
        legal = state.legal_actions()
        if len(legal) == 1:
            action, who = legal[0], "forced"
        elif seat in agents:
            break
        else:
            bot = make_bot(cfg["bots"][str(seat)])
            action = bot.choose(state.clone(), bot_rng(cfg["seed"], len(data["history"])))
            who = cfg["bots"][str(seat)]
        data["log"].append(
            {
                "n": len(data["history"]),
                "seat": seat,
                "action": action,
                "label": state.action_to_string(seat, action),
                "by": who,
            }
        )
        data["history"].append(action)
        state.apply_action(action)
    return state


def finish(path: Path, data: dict[str, Any], state: BusState) -> None:
    if state.is_terminal():
        cfg = data["config"]
        doc = fresh_document(cfg["players"], cfg["returns"], data["history"])
        path_r = replay_path(path)
        atomic_write_json(path_r, doc, indent=None)


def render(path: Path, data: dict[str, Any], state: BusState, out: Callable[[str], None]) -> None:
    cfg = data["config"]
    agents = set(cfg["agents"])
    marker = data["shown"]
    out(f"== Bus: {cfg['players']} players, agent seat(s) {sorted(agents)}, seed {cfg['seed']} ==")
    out(
        "bots: "
        + ", ".join(f"P{s}={t}" for s, t in sorted(cfg["bots"].items(), key=lambda x: int(x[0])))
    )
    new = [e for e in data["log"] if e["n"] >= marker]
    out(f"-- events since your last decision ({len(new)}) --")
    for e in new:
        out(f"  #{e['n']} P{e['seat']} [{e['by']}]: {e['label']}")
    out("-- state --")
    out(str(state))
    if state.is_terminal():
        out("-- GAME OVER --")
        for line in format_result(state):
            out(line)
        out(f"replay written to {replay_path(path)}")
        return
    seat = state.current_player()
    out(f"-- P{seat} to move; legal actions ({len(state.legal_actions())}) --")
    for a in state.legal_actions():
        out(f"  {a}: {state.action_to_string(seat, a)}")


def cmd_legend(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    from tools.board_legend import legend

    out(legend())
    return 0


def render_summary(state: BusState, out: Callable[[str], None]) -> None:
    from tools.board_legend import summary

    out(summary(state.core))


def cmd_new(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    agents = sorted(parse_humans([args.agent], args.players))
    if not agents:
        raise AgentPlayError("--agent needs at least one seat")
    bot_seats = [s for s in range(args.players) if s not in agents]
    bots = expand_bots(args.bots, bot_seats) if bot_seats else {}
    for t in bots.values():
        if not is_valid_seat_type(t):
            raise AgentPlayError(seat_type_error(t) or f"bad bot spec {t!r}")
        try:
            check_players(t, args.players)
        except ValueError as e:
            raise AgentPlayError(str(e)) from None
    if args.out.exists() and not args.force:
        raise AgentPlayError(f"{args.out} exists (use --force to overwrite)")
    data: dict[str, Any] = {
        "format": FORMAT,
        "config": {
            "players": args.players,
            "agents": agents,
            "bots": {str(s): t for s, t in bots.items()},
            "seed": args.seed,
            "returns": args.returns,
        },
        "history": [],
        "log": [],
        "notes": [],
        "shown": 0,
    }
    state = advance(data)
    finish(args.out, data, state)
    save(args.out, data)
    render(args.out, data, state, out)
    data["shown"] = len(data["history"])
    save(args.out, data)
    return 0


def cmd_show(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    data = load(args.game)
    state = build_state(data["config"], data["history"])
    render(args.game, data, state, out)
    if args.summary:
        render_summary(state, out)
    return 0


def cmd_act(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    data = load(args.game)
    state = build_state(data["config"], data["history"])
    if state.is_terminal():
        raise AgentPlayError("game is over")
    seat = state.current_player()
    if args.action not in state.legal_actions():
        raise AgentPlayError(
            f"action {args.action} is not legal for P{seat}; legal: {state.legal_actions()}"
        )
    n = len(data["history"])
    data["log"].append(
        {
            "n": n,
            "seat": seat,
            "action": args.action,
            "label": state.action_to_string(seat, args.action),
            "by": "agent",
        }
    )
    data["notes"].append(
        {
            "n": n,
            "seat": seat,
            "action": args.action,
            "label": data["log"][-1]["label"],
            "note": args.note,
        }
    )
    data["history"].append(args.action)
    data["shown"] = len(data["history"])  # the agent's own move is not "since" its decision
    state = advance(data)
    finish(args.game, data, state)
    save(args.game, data)
    render(args.game, data, state, out)
    if args.summary:
        render_summary(state, out)
    data["shown"] = len(data["history"])
    save(args.game, data)
    return 0


def cmd_notes(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    data = load(args.game)
    for m in data["notes"]:
        out(f"#{m['n']} P{m['seat']} {m['action']}: {m['label']}")
        if m["note"]:
            out(f"    note: {m['note']}")
    return 0


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="tools.agent_play", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="create a game and advance to the agent's first decision")
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--agent", default="0", help="agent seat(s) from 0, comma separated")
    p.add_argument("--bots", default="heuristic", help="seat type for bot seats, or a comma list")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--returns", choices=("win_loss", "margin"), default="win_loss")
    p.add_argument("--force", action="store_true", help="overwrite an existing game file")
    p.set_defaults(fn=cmd_new)
    p = sub.add_parser("show", help="print the current decision")
    p.add_argument("--game", type=Path, required=True)
    p.add_argument("--summary", action="store_true", help="append a per-seat situational summary")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("act", help="play an action id, advance bots, show the next decision")
    p.add_argument("action", type=int)
    p.add_argument("--game", type=Path, required=True)
    p.add_argument("--note", default="")
    p.add_argument("--summary", action="store_true", help="append a per-seat situational summary")
    p.set_defaults(fn=cmd_act)
    p = sub.add_parser("legend", help="print the board map legend and a key to the state text")
    p.set_defaults(fn=cmd_legend)
    p = sub.add_parser("notes", help="dump the agent's moves with notes")
    p.add_argument("--game", type=Path, required=True)
    p.set_defaults(fn=cmd_notes)
    args = ap.parse_args(argv)
    try:
        return args.fn(args, out)
    except (AgentPlayError, ValueError, pyspiel.SpielError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
