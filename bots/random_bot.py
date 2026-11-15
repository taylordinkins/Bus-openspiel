"""Uniformly random bot."""

from __future__ import annotations

import random

import pyspiel


def bot_candidates(state: pyspiel.State) -> list[int]:
    """Actions a random bot may pick (every legal action; Bus has no pointless ones)."""
    return list(state.legal_actions())


class RandomBot:
    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        return rng.choice(bot_candidates(state))
