"""Helpers shared by the unit-13 tests (observer, action_meta, forced_action, game_result)."""

from __future__ import annotations

import copy
import importlib
import json
import random
from typing import Any

import pyspiel

from tests import harness as H


def game(n: int = 4, **kw: Any) -> Any:
    importlib.import_module("bus.game")  # registers python_bus

    return pyspiel.load_game("python_bus", {"players": n, **kw})


def dump(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s)


def reload(s: Any) -> Any:
    return H.load_doc_state(copy.deepcopy(dump(s)), s.get_game())


def key(s: Any) -> str:
    return json.dumps(dump(s), sort_keys=True)


def random_game(n: int, seed: int, returns: str = "win_loss", every: int = 1) -> list[Any]:
    """Plays a seeded random game; returns clones of the state after every ``every`` actions
    (always including the initial and the terminal state)."""
    rng = random.Random(seed)
    s = game(n, returns=returns).new_initial_state()
    out = [s.clone()]
    i = 0
    while not s.is_terminal():
        s.apply_action(rng.choice(s.legal_actions()))
        i += 1
        if i % every == 0 or s.is_terminal():
            out.append(s.clone())
    return out


def find(name: str) -> Any:
    """First attribute ``name`` found in the engine modules that may host it."""
    for mod in ("bus.rules", "bus.actions", "bus.observer", "bus.game", "bus"):
        m = importlib.import_module(mod)
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(f"{name} not found in bus.rules / actions / observer / game")


def field(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict-like or attribute-style result."""
    if isinstance(obj, dict):
        d: dict[str, Any] = obj  # pyright: ignore[reportUnknownVariableType]
        return d[name]
    return getattr(obj, name)
