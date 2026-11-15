import random

import pyspiel
import pytest

import bus.game  # noqa: F401  (import registers the game)
from bots.random_bot import RandomBot
from bots.registry import KINDS, is_valid_seat_type, make_bot, parse_seat_type, register


def test_parse() -> None:
    assert parse_seat_type("random") == ("random", None)
    assert parse_seat_type("mcts:200") == ("mcts", "200")
    assert parse_seat_type("human") == ("human", None)


@pytest.mark.parametrize("s", ["", "Human", "random ", "a:", "a:b:c", ":5", "x\n"])
def test_parse_malformed(s: str) -> None:
    with pytest.raises(ValueError):
        parse_seat_type(s)


@pytest.mark.parametrize("s", ["human", "random"])
def test_valid(s: str) -> None:
    assert is_valid_seat_type(s)


@pytest.mark.parametrize("s", ["alien", "mcts:50", "random:3", "random:", "Human", ""])
def test_invalid(s: str) -> None:
    assert not is_valid_seat_type(s)


def test_make_bot_caches_and_rejects() -> None:
    assert make_bot("random") is make_bot("random")
    with pytest.raises(ValueError):
        make_bot("alien")
    with pytest.raises(ValueError):
        make_bot("human")


def test_param_validation_hook() -> None:
    def factory(param: str | None) -> RandomBot:
        if param is None or not param.isdigit():
            raise ValueError("need int")
        return RandomBot()

    register("dummy", factory)
    try:
        assert is_valid_seat_type("dummy:5")
        assert not is_valid_seat_type("dummy")
        assert not is_valid_seat_type("dummy:x")
    finally:
        del KINDS["dummy"]


def test_random_bot_plays_legal_4p_game() -> None:
    game = pyspiel.load_game("python_bus", {"players": 4})
    state = game.new_initial_state()
    bot = make_bot("random")
    rng = random.Random(3)
    steps = 0
    while not state.is_terminal() and steps < 3000:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            state.apply_action(rng.choice([a for a, _ in outcomes]))
        else:
            a = bot.choose(state, rng)
            assert a in state.legal_actions()
            state.apply_action(a)
        steps += 1
    assert steps > 10
