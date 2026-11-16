"""The Bot protocol."""

from __future__ import annotations

import random
from typing import Protocol

import pyspiel


class Bot(Protocol):
    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        """Return a legal action id for the current player of `state`.

        Bots get the same information a player has: they must not read other seats' hidden
        information from `state` (the server passes a full-information clone).
        """
        ...
