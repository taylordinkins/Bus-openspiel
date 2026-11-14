"""OpenSpiel conformance (PLAN section 6 item 4): random_sim_test, round trips, declared bounds."""

from __future__ import annotations

import pickle
import random
from typing import Any

import pyspiel
import pytest

from tests.support.play import game as make_game

PLAYERS = (3, 4, 5)
MODES = ("win_loss", "margin")


def _modes() -> list[Any]:
    return ["win_loss", "margin"]


def _sim(n: int, mode: str, sims: int) -> None:
    g = make_game(n, returns=mode)
    pyspiel.random_sim_test(g, num_sims=sims, serialize=True, verbose=False)


@pytest.mark.parametrize("mode", _modes())
@pytest.mark.parametrize("n", PLAYERS)
def test_CONF_random_sim_test_fast(n: int, mode: str) -> None:
    """random_sim_test with serialization on (also checks max_game_length and masks)."""
    _sim(n, mode, 2)


@pytest.mark.slow
@pytest.mark.parametrize("mode", _modes())
@pytest.mark.parametrize("n", PLAYERS)
def test_CONF_random_sim_test_slow(n: int, mode: str) -> None:
    _sim(n, mode, 25)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", PLAYERS)
def test_CONF_clone_and_serialize_round_trip(n: int, mode: str) -> None:
    """At sampled states clone, serialize/deserialize and pickle give states that agree and then
    play on identically."""
    g = make_game(n, returns=mode)
    rng = random.Random(n * 10 + len(mode))
    s = g.new_initial_state()
    i = 0
    while not s.is_terminal():
        if i % 40 == 0:
            _check_copies(g, s)
        s.apply_action(rng.choice(s.legal_actions()))
        i += 1
    _check_copies(g, s)


def _check_copies(g: Any, s: Any) -> None:
    c = s.clone()
    assert str(c) == str(s) and c.history() == s.history()
    g2, d = pyspiel.deserialize_game_and_state(pyspiel.serialize_game_and_state(g, s))
    assert str(d) == str(s) and d.history() == s.history()
    assert g2.get_parameters() == g.get_parameters()
    p = pickle.loads(pickle.dumps(s))
    assert str(p) == str(s)
    if not s.is_terminal():
        assert c.legal_actions() == d.legal_actions() == s.legal_actions()
        a = s.legal_actions()[0]
        for x in (s, c, d):
            x.apply_action(a)
        assert str(c) == str(d) == str(s)
        # s advanced; the copies must not have been aliased before that
    assert s.history() == c.history() == d.history()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", PLAYERS)
def test_CONF_game_length_and_returns_bounds(n: int, mode: str) -> None:
    g = make_game(n, returns=mode)
    rng = random.Random(100 + n)
    s = g.new_initial_state()
    decisions = 0
    while not s.is_terminal():
        s.apply_action(rng.choice(s.legal_actions()))
        decisions += 1
    assert decisions <= g.max_game_length()
    assert len(s.history()) <= g.max_history_length()
    assert g.max_chance_nodes_in_history() == 0
    r = s.returns()
    assert len(r) == n
    assert all(g.min_utility() - 1e-9 <= x <= g.max_utility() + 1e-9 for x in r)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", PLAYERS)
def test_CONF_game_type_fields(n: int, mode: str) -> None:
    """PLAN 5.2 game definition."""
    g = make_game(n, returns=mode)
    t = g.get_type()
    assert t.short_name == "python_bus"
    assert t.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert t.chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC
    assert t.information == pyspiel.GameType.Information.PERFECT_INFORMATION
    assert t.reward_model == pyspiel.GameType.RewardModel.TERMINAL
    want = (
        pyspiel.GameType.Utility.ZERO_SUM
        if mode == "win_loss"
        else pyspiel.GameType.Utility.GENERAL_SUM
    )
    assert t.utility == want
    assert (t.min_num_players, t.max_num_players) == (3, 5)
    assert t.provides_observation_string and t.provides_observation_tensor
    assert g.num_players() == n
    assert g.get_parameters() == {"players": n, "returns": mode}
    assert make_game(n).get_parameters()["returns"] == "win_loss"
