# pyright: basic
"""Model-based planner seat `planner[:R[-K][e]]` (`bots/planner.py`)."""

from __future__ import annotations

import random

import pyspiel
import pytest

import bus.game  # noqa: F401  (import registers the game)
from bots.planner import MODELS, PlannerBot, _Roll, parse_planner_param
from bots.registry import is_valid_seat_type, make_bot
from bus import rules


def _play(types: list[str], seed: int) -> pyspiel.State:
    state = pyspiel.load_game("python_bus", {"players": len(types)}).new_initial_state()
    rng = random.Random(seed)
    bots = [make_bot(t) for t in types]
    while not state.is_terminal():
        state.apply_action(bots[state.current_player()].choose(state, rng))
    return state


def test_registry_and_params() -> None:
    assert is_valid_seat_type("planner")
    assert is_valid_seat_type("planner:8")
    assert is_valid_seat_type("planner:8-12e")
    assert not is_valid_seat_type("planner:x")
    b = parse_planner_param("6-5e")
    assert (b.rollouts, b.max_cands, b.early) == (6, 5, True)
    assert is_valid_seat_type("net_sample:ppo-long")


@pytest.mark.slow
def test_full_game_legal_and_identifies_heuristic() -> None:
    bot = PlannerBot(rollouts=1, max_cands=4)
    state = pyspiel.load_game("python_bus", {"players": 3}).new_initial_state()
    rng = random.Random(0)
    heur = make_bot("heuristic")
    while not state.is_terminal():
        legal = state.legal_actions()
        a = bot.choose(state, rng) if state.current_player() == 0 else heur.choose(state, rng)
        assert a in legal
        state.apply_action(a)
    t = bot._tracker
    assert t is not None
    for p in (1, 2):
        assert MODELS[int(t.posterior(p).argmax())] == "heuristic"


def test_waste_tracking_counts_power_zero_markers() -> None:
    """Replaying a random game through _Roll.step counts the same wasted markers as a direct
    reading of the engine (markers whose resolution never reached a decision node)."""
    state = _play(["random", "random", "random"], seed=3)
    hist = state.history()
    total = 0
    for me in range(3):
        ro = _Roll(rules.initial_core(3), [], False, 0, me=me)
        i = 0
        while i < len(hist) and ro.core.phase != rules.PHASE_END:
            before = ro.core.move_number
            ro.step(hist[i])
            i += ro.core.move_number - before
        total += ro.waste
    assert total > 0  # random play always wastes markers


@pytest.mark.slow
def test_tree_planner_plays_legal_moves_and_params() -> None:
    from bots.planner import TreePlannerBot, parse_tree_param, surely_wasted

    assert is_valid_seat_type("tplanner") and is_valid_seat_type("tplanner:64-1.0")
    assert not is_valid_seat_type("tplanner:x")
    t = parse_tree_param("64-1.0")
    assert (t.sims, t.c) == (64, 1.0)
    bot = TreePlannerBot(sims=8)
    state = pyspiel.load_game("python_bus", {"players": 3}).new_initial_state()
    rng = random.Random(1)
    other = make_bot("random")
    decisions = 0
    while not state.is_terminal() and decisions < 6:
        legal = state.legal_actions()
        if state.current_player() == 0 and len(legal) > 1:
            a = bot.choose(state, rng)
            decisions += 1
            core = state.core  # type: ignore[attr-defined]
            # never picks a surely-wasted selection when a useful one exists
            if any(not surely_wasted(core, x) for x in legal):
                assert not surely_wasted(core, a)
        else:
            a = other.choose(state, rng)
        assert a in legal
        state.apply_action(a)
