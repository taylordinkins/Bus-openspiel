import random
import time

import pyspiel
import pytest

import bus.game  # noqa: F401  (import registers the game)
from bots.registry import is_valid_seat_type, make_bot
from server.bots import is_valid_seat_type as server_valid


def _play(players: int, seats: list[str], seed: int, max_steps: int = 4000) -> pyspiel.State:
    game = pyspiel.load_game("python_bus", {"players": players})
    state = game.new_initial_state()
    rng = random.Random(seed)
    bots = [make_bot(s) for s in seats]
    steps = 0
    while not state.is_terminal() and steps < max_steps:
        a = bots[state.current_player()].choose(state, rng)
        assert a in state.legal_actions()
        state.apply_action(a)
        steps += 1
    assert state.is_terminal()
    return state


@pytest.mark.parametrize("players", [3, 4, 5])
def test_heuristic_always_legal(players: int) -> None:
    for seed in range(2):
        seats = [("heuristic" if (i + seed) % 2 == 0 else "random") for i in range(players)]
        _play(players, seats, seed)


def test_all_heuristic_game() -> None:
    _play(4, ["heuristic"] * 4, 1)


def test_deterministic() -> None:
    seats = ["heuristic", "random", "heuristic", "random"]
    a = _play(4, seats, 11)
    b = _play(4, seats, 11)
    assert a.history() == b.history()


def test_registry_and_server_validation() -> None:
    assert is_valid_seat_type("heuristic")
    assert server_valid("heuristic")
    assert not is_valid_seat_type("heuristic:3")
    assert not server_valid("heuristic:3")


def test_decision_latency() -> None:
    game = pyspiel.load_game("python_bus", {"players": 4})
    state = game.new_initial_state()
    rng = random.Random(0)
    bot = make_bot("heuristic")
    total = 0.0
    n = 0
    while not state.is_terminal():
        t = time.perf_counter()
        a = bot.choose(state, rng)
        total += time.perf_counter() - t
        n += 1
        state.apply_action(a)
    assert total / n < 0.02  # generous bound; target is about 2 ms


@pytest.mark.slow
def test_beats_random_smoke() -> None:
    wins = 0.0
    games = 40
    for g in range(games):
        seat = g % 4
        seats = ["random"] * 4
        seats[seat] = "heuristic"
        state = _play(4, seats, 100 + g)
        wins += 1.0 if state.returns()[seat] > 0 else 0.0  # sole or shared win
    assert wins / games > 0.5
