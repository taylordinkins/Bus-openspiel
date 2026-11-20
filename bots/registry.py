"""Registry of bot kinds and seat-type parsing.

A seat type is `human`, a bare bot kind (`random`) or `kind:param` (`mcts:200`). Adding a kind is
one `register(...)` call with a factory taking the optional param string; the factory raises
ValueError for a bad param.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from bots.base import Bot
from bots.gumbel import parse_gumbel_param
from bots.heuristic import HeuristicBot
from bots.mcts import parse_mcts_param
from bots.mcts_net import parse_mcts_net_param
from bots.net import parse_net_param
from bots.planner import parse_planner_param, parse_tree_param
from bots.random_bot import RandomBot

HUMAN = "human"
# Permissive shape for schemas; is_valid_seat_type() does the real validation.
SEAT_TYPE_PATTERN = r"^(human|[a-z_]+(:[A-Za-z0-9_.-]+)?)$"
_SEAT_RE = re.compile(r"[a-z_]+(?::[A-Za-z0-9_.-]+)?")

Factory = Callable[[str | None], Bot]


@dataclass(frozen=True)
class BotKind:
    factory: Factory


KINDS: dict[str, BotKind] = {}
# Instances by full seat type. Shared dict (tests may monkeypatch entries).
BOTS: dict[str, Bot] = {}


def register(kind: str, factory: Factory) -> None:
    KINDS[kind] = BotKind(factory)


def _random_factory(param: str | None) -> Bot:
    if param is not None:
        raise ValueError("random takes no parameter")
    return RandomBot()


register("random", _random_factory)


def _heuristic_factory(param: str | None) -> Bot:
    if param is not None:
        raise ValueError("heuristic takes no parameter")
    return HeuristicBot()


register("heuristic", _heuristic_factory)


def _mcts_factory(param: str | None) -> Bot:
    return parse_mcts_param(param)


register("mcts", _mcts_factory)


def _net_factory(param: str | None) -> Bot:
    return parse_net_param(param)


register("net", _net_factory)


def _net_sample_factory(param: str | None) -> Bot:
    bot = parse_net_param(param)
    bot.sample = True
    return bot


# Tool-only: the same net sampling from its policy instead of greedy (evaluation opponents).
register("net_sample", _net_sample_factory)


def _mcts_net_factory(param: str | None) -> Bot:
    return parse_mcts_net_param(param)


register("mcts_net", _mcts_net_factory)


def _gumbel_factory(param: str | None) -> Bot:
    return parse_gumbel_param(param)


# Tool-only (registry, tournament, eval_search); the lobby's model list does not offer it (Q-AZ7).
register("gumbel", _gumbel_factory)


def _planner_factory(param: str | None) -> Bot:
    return parse_planner_param(param)


# Flat model-based planner (tool-only); the tree version below is offered in the lobby.
register("planner", _planner_factory)


def _tree_factory(param: str | None) -> Bot:
    return parse_tree_param(param)


# Model-based tree planner (docs/PLANNER.md); lobby preset "Planner · Strongest".
register("tplanner", _tree_factory)


def parse_seat_type(s: str) -> tuple[str, str | None]:
    """Split into (kind, param); ValueError if malformed. Does not check the kind exists."""
    if not _SEAT_RE.fullmatch(s) and s != HUMAN:
        raise ValueError(f"malformed seat type {s!r}")
    kind, _, param = s.partition(":")
    return kind, (param or None) if ":" in s else None


def is_valid_seat_type(seat_type: str) -> bool:
    if seat_type == HUMAN:
        return True
    try:
        kind, param = parse_seat_type(seat_type)
    except ValueError:
        return False
    if kind not in KINDS:
        return False
    if seat_type in BOTS:
        return True
    try:
        KINDS[kind].factory(param)
    except ValueError:
        return False
    return True


def seat_type_error(seat_type: str) -> str | None:
    """Why `seat_type` is not valid (None if it is); keeps model errors readable."""
    if seat_type == HUMAN or seat_type in BOTS:
        return None
    try:
        kind, param = parse_seat_type(seat_type)
        if kind not in KINDS:
            return f"unknown seat type {seat_type!r}"
        KINDS[kind].factory(param)
    except ValueError as e:
        return f"invalid seat type {seat_type!r}: {e}"
    return None


def make_bot(seat_type: str) -> Bot:
    """Resolve a bot seat type to a (cached) Bot; ValueError if invalid or human."""
    if seat_type in BOTS:
        return BOTS[seat_type]
    kind, param = parse_seat_type(seat_type)
    if kind not in KINDS:
        raise ValueError(f"unknown bot kind {kind!r}")
    bot = KINDS[kind].factory(param)
    BOTS[seat_type] = bot
    return bot
