"""Bot seats: thin layer over the top-level `bots` package."""

from __future__ import annotations

import random

from bots.base import Bot
from bots.random_bot import RandomBot, bot_candidates
from bots.registry import (
    BOTS,
    HUMAN,
    SEAT_TYPE_PATTERN,
    is_valid_seat_type,
    make_bot,
)

__all__ = [
    "BOTS",
    "HUMAN",
    "SEAT_TYPE_PATTERN",
    "Bot",
    "RandomBot",
    "bot_candidates",
    "bot_rng",
    "get_bot",
    "is_valid_seat_type",
]


def get_bot(seat_type: str) -> Bot:
    """Resolve a bot seat type to a Bot (ValueError if unknown)."""
    return make_bot(seat_type)


def bot_rng(seed: int, history_length: int, salt: str = "") -> random.Random:
    """Reproducible from (seed, secret salt, position in history); distinct from chance RNG."""
    return random.Random(
        f"{seed}:{salt}:bot:{history_length}" if salt else f"{seed}:bot:{history_length}"
    )
