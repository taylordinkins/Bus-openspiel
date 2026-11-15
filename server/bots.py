"""Bot seats: a registry of seat type -> Bot. Only the random bot exists so far."""

from __future__ import annotations

import random
from typing import Protocol

import pyspiel

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


class Bot(Protocol):
    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        """Return a legal action id for the current player of `state`.

        Bots get the same information a player has: they must not read other seats' hidden
        information from `state` (the server passes a full-information clone).
        """
        ...


def bot_candidates(state: pyspiel.State) -> list[int]:
    """Actions a random bot may pick. unit 2: filter pointless actions here if Bus needs it."""
    return list(state.legal_actions())


class RandomBot:
    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        return rng.choice(bot_candidates(state))


BOTS: dict[str, Bot] = {"random": RandomBot()}

HUMAN = "human"
# Keep in sync with the seat types accepted by is_valid_seat_type().
SEAT_TYPE_PATTERN = r"^(human|random)$"


def is_valid_seat_type(seat_type: str) -> bool:
    return seat_type == HUMAN or seat_type in BOTS


def get_bot(seat_type: str) -> Bot:
    """Resolve a bot seat type to a Bot (KeyError if unknown)."""
    return BOTS[seat_type]


def bot_rng(seed: int, history_length: int, salt: str = "") -> random.Random:
    """Reproducible from (seed, secret salt, position in history); distinct from chance RNG."""
    return random.Random(
        f"{seed}:{salt}:bot:{history_length}" if salt else f"{seed}:bot:{history_length}"
    )
