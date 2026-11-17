"""Phase 4 unit 7: net-guided MCTS seat `mcts_net:<sims>-<model>`."""

import random
import subprocess
import sys
import time

import numpy as np
import pyspiel
import pytest

import bots.mcts_net as mn
import bus.game  # noqa: F401  (import registers the game)
from bots.mcts_net import NetEvaluator, NetMCTSBot, parse_mcts_net_param
from bots.net import NetBot, check_players, load_weights
from bots.registry import is_valid_seat_type, make_bot, seat_type_error
from train.features import encode

NAME = "ppo-long"
SEAT = f"mcts_net:3-{NAME}"


def _state(players: int, plies: int, seed: int = 0) -> pyspiel.State:
    s = pyspiel.load_game("python_bus", {"players": players}).new_initial_state()
    rng = random.Random(seed)
    for _ in range(plies):
        if s.is_terminal():
            break
        s.apply_action(rng.choice(s.legal_actions()))
    return s


def _play(players: int, seats: list[str], seed: int) -> pyspiel.State:
    state = pyspiel.load_game("python_bus", {"players": players}).new_initial_state()
    rng = random.Random(seed)
    bots = [make_bot(s) for s in seats]
    while not state.is_terminal():
        a = bots[state.current_player()].choose(state, rng)
        assert a in state.legal_actions()
        state.apply_action(a)
    return state


@pytest.mark.parametrize("players", [3, 4, 5])
def test_value_vector_maps_ego_to_absolute_seats(players: int) -> None:
    ev = NetEvaluator(NAME)
    w = load_weights(NAME)
    seen = set()
    for plies in range(0, 400, 7):
        s = _state(players, plies, seed=plies)
        if s.is_terminal():
            break
        cur = s.current_player()
        seen.add(cur)
        mask = np.zeros(w.n_actions, bool)
        mask[s.legal_actions()] = True
        _, v_ego = w.policy(encode(s.core), mask)
        v = ev.evaluate(s)
        assert v.shape == (players,)
        rotated = np.array([v[(cur + k) % players] for k in range(players)])
        np.testing.assert_allclose(rotated, v_ego[:players], atol=1e-6)
    assert len(seen) == players  # every seat was the mover at least once


def test_prior_is_distribution_over_legal_actions_only() -> None:
    ev = NetEvaluator(NAME)
    for plies in (0, 20, 60, 150):
        s = _state(4, plies, seed=plies)
        if s.is_terminal():
            continue
        pr = ev.prior(s)
        assert sorted(a for a, _ in pr) == sorted(s.legal_actions())
        assert all(p >= 0 for _, p in pr)
        assert sum(p for _, p in pr) == pytest.approx(1.0, abs=1e-5)


def test_terminal_returns_are_game_returns() -> None:
    s = _state(4, 0)
    rng = random.Random(0)
    while not s.is_terminal():
        s.apply_action(rng.choice(s.legal_actions()))
    np.testing.assert_allclose(NetEvaluator(NAME).evaluate(s), s.returns())


@pytest.mark.parametrize("players", [3, 4, 5])
def test_legal_over_full_games(players: int) -> None:
    seats = [SEAT] + ["random"] * (players - 1)
    assert _play(players, seats, 1).is_terminal()


def test_deterministic_for_same_rng() -> None:
    s = _state(4, 80, seed=3)
    bot = parse_mcts_net_param(f"8-{NAME}")
    assert bot.choose(s, random.Random(5)) == bot.choose(s, random.Random(5))


def test_forced_move_skips_search(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _state(4, 0)
    while len(s.legal_actions()) != 1:
        s.apply_action(s.legal_actions()[0])
    monkeypatch.setattr(NetMCTSBot, "_search", lambda *a: 1 / 0)
    assert make_bot(SEAT).choose(s, random.Random(0)) == s.legal_actions()[0]


def test_time_limit_is_respected_and_copy_leaves_original() -> None:
    s = _state(4, 60, seed=2)
    base = parse_mcts_net_param(f"20000-{NAME}")
    assert base.time_limit_s is None
    timed = base.with_time_limit(0.2)
    assert base.time_limit_s is None and timed.time_limit_s == 0.2
    t = time.monotonic()
    assert timed.choose(s, random.Random(0)) in s.legal_actions()
    assert time.monotonic() - t < 5


def test_fallback_on_exception_and_illegal(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _state(4, 60, seed=2)
    bot = parse_mcts_net_param(f"5-{NAME}")
    greedy = NetBot(NAME).choose(s, random.Random(0))
    monkeypatch.setattr(NetMCTSBot, "_search", lambda *a: 1 / 0)
    assert bot.choose(s, random.Random(0)) == greedy
    monkeypatch.setattr(NetMCTSBot, "_search", lambda *a: -1)
    assert bot.choose(s, random.Random(0)) == greedy


@pytest.mark.parametrize(
    "seat",
    [
        "mcts_net",
        "mcts_net:50",
        "mcts_net:ppo-long",
        "mcts_net:-ppo-long",
        "mcts_net:50-",
        "mcts_net:50-missing",
        "mcts_net:50-..",
        "mcts_net:50-ppo-long.npz",
        "mcts_net:50-/etc/passwd",
        "mcts_net:0-ppo-long",
        "mcts_net:20001-ppo-long",
    ],
)
def test_bad_seat_types_rejected(seat: str) -> None:
    assert not is_valid_seat_type(seat)


def test_valid_seat_type_and_player_check() -> None:
    assert is_valid_seat_type(SEAT)
    assert seat_type_error(SEAT) is None
    check_players(SEAT, 4)
    with pytest.raises(ValueError):
        check_players(SEAT, 2)
    with pytest.raises(ValueError):
        check_players("mcts_net:5-missing", 4)


def test_exploration_constant_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mn.NetMCTSBot(NAME).uct_c == mn.UCT_C
    monkeypatch.setenv("BUS_MCTS_NET_C", "0.7")
    assert mn.NetMCTSBot(NAME).uct_c == 0.7


def test_does_not_import_torch() -> None:
    code = (
        "import sys, random, pyspiel, bus.game\n"
        "import bots.mcts_net\n"
        "from bots.registry import make_bot\n"
        "b = make_bot('mcts_net:3-ppo-long')\n"
        "s = pyspiel.load_game('python_bus', {'players': 4}).new_initial_state()\n"
        "r = random.Random(0)\n"
        "for _ in range(40): s.apply_action(r.choice(s.legal_actions()))\n"
        "assert b.choose(s, r) in s.legal_actions()\n"
        "sys.exit(1 if any(k == 'torch' or k.startswith('torch.') for k in sys.modules) else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
