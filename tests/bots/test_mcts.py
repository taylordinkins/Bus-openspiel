import random
import time

import pyspiel
import pytest

import bus.game  # noqa: F401  (import registers the game)
from bots.mcts import MCTSSeatBot
from bots.registry import is_valid_seat_type, make_bot, parse_seat_type
from server.bots import is_valid_seat_type as server_valid


def _play(players: int, seats: list[str], seed: int) -> pyspiel.State:
    game = pyspiel.load_game("python_bus", {"players": players})
    state = game.new_initial_state()
    rng = random.Random(seed)
    bots = [make_bot(s) for s in seats]
    while not state.is_terminal():
        a = bots[state.current_player()].choose(state, rng)
        assert a in state.legal_actions()
        state.apply_action(a)
    return state


def _mid_game(seed: int = 5, steps: int = 80) -> pyspiel.State:
    state = pyspiel.load_game("python_bus", {"players": 4}).new_initial_state()
    rng = random.Random(seed)
    for _ in range(steps):
        state.apply_action(rng.choice(state.legal_actions()))
    assert not state.is_terminal()
    return state


@pytest.mark.parametrize("players", [3, 4, 5])
def test_mcts_legal_over_full_game(players: int) -> None:
    seats = ["mcts:10"] + ["random"] * (players - 1)
    _play(players, seats, players)


@pytest.mark.parametrize("s", ["mcts", "mcts:50", "mcts:50-heuristic", "mcts:50-random"])
def test_valid_seats(s: str) -> None:
    assert is_valid_seat_type(s)
    assert server_valid(s)
    assert parse_seat_type(s)[0] == "mcts"


@pytest.mark.parametrize(
    "s", ["mcts:0", "mcts:abc", "mcts:5:foo", "mcts:5-foo", "mcts:20001", "mcts:-3", "mcts:5-"]
)
def test_bad_seats(s: str) -> None:
    assert not is_valid_seat_type(s)
    assert not server_valid(s)


def test_deterministic_with_sim_cap() -> None:
    state = _mid_game()
    for kind in ("mcts:20", "mcts:20-heuristic"):
        bot = make_bot(kind)
        assert bot.choose(state, random.Random(9)) == bot.choose(state, random.Random(9))


def test_single_legal_action_no_search() -> None:
    state = pyspiel.load_game("python_bus", {"players": 3}).new_initial_state()
    rng = random.Random(0)
    while not state.is_terminal() and len(state.legal_actions()) != 1:
        state.apply_action(rng.choice(state.legal_actions()))
    bot = MCTSSeatBot(sims=20000)
    t = time.monotonic()
    assert bot.choose(state, rng) == state.legal_actions()[0]
    assert time.monotonic() - t < 0.5


def test_time_limit() -> None:
    state = _mid_game()
    bot = MCTSSeatBot(sims=20000).with_time_limit(0.3)
    assert bot.time_limit_s == 0.3
    t = time.monotonic()
    a = bot.choose(state, random.Random(1))
    assert time.monotonic() - t < 3.0
    assert a in state.legal_actions()
    assert make_bot("mcts:50").time_limit_s is None


def test_sim_cap_respected() -> None:
    state = _mid_game()
    t = time.monotonic()
    MCTSSeatBot(sims=3).choose(state, random.Random(1))
    assert time.monotonic() - t < 5.0


@pytest.mark.slow
def test_mcts_game_vs_random() -> None:
    state = _play(4, ["mcts:30", "random", "random", "random"], 2)
    assert state.is_terminal()
