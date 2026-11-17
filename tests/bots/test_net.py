"""Phase 4 unit 6: `net:<name>` seats resolved only through models/manifest.yaml."""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyspiel
import pytest

import bots.net as netmod
import bus.game  # noqa: F401  (import registers the game)
from bots.net import ModelInfo, NetBot, check_players, load_weights, resolve_model
from bots.net_forward import StaleModelError
from bots.registry import is_valid_seat_type, make_bot, seat_type_error

NAME = "ppo-long"


def _play(players: int, seats: list[str], seed: int) -> list[int]:
    state = pyspiel.load_game("python_bus", {"players": players}).new_initial_state()
    rng = random.Random(seed)
    bots = [make_bot(s) for s in seats]
    while not state.is_terminal():
        a = bots[state.current_player()].choose(state, rng)
        assert a in state.legal_actions()
        state.apply_action(a)
    return list(state.history())


def test_manifest_lists_long_model() -> None:
    info = resolve_model(NAME)
    assert info.path.is_file() and info.label == "Trained net"
    assert {3, 4, 5} <= set(info.players)


@pytest.mark.parametrize(
    "name", ["nope", "", "../ppo-long", "ppo-long.npz", "a/b", "/etc/passwd", "ppo smoke", "a\\b"]
)
def test_resolver_rejects_unknown_and_path_like(name: str) -> None:
    with pytest.raises(ValueError):
        resolve_model(name)
    assert not is_valid_seat_type(f"net:{name}")


def test_seat_types_need_a_name_and_reject_unknown() -> None:
    assert not is_valid_seat_type("net")
    assert not is_valid_seat_type("net:missing")
    assert is_valid_seat_type(f"net:{NAME}")
    assert "unknown net model" in (seat_type_error("net:missing") or "")


def test_wrong_player_count() -> None:
    check_players(NAME, 4)  # not a net seat: no-op
    check_players(f"net:{NAME}", 4)
    with pytest.raises(ValueError, match="players"):
        check_players(f"net:{NAME}", 2)
    with pytest.raises(ValueError, match="players"):
        resolve_model(NAME, 6)


def test_stale_features_version(monkeypatch: pytest.MonkeyPatch) -> None:
    real = resolve_model(NAME)
    stale = ModelInfo("stale", real.file, "x", "", real.players, real.features_version + 1)
    monkeypatch.setattr(netmod, "manifest", lambda: {"stale": stale})
    load_weights.cache_clear()
    try:
        with pytest.raises(StaleModelError, match="features"):
            NetBot("stale")
        assert not is_valid_seat_type("net:stale")
        assert "features" in (seat_type_error("net:stale") or "")
    finally:
        load_weights.cache_clear()


def test_stale_layout_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = resolve_model(NAME)
    with np.load(real.path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta["layout_hash"] = "deadbeef"
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **arrays)
    bad = ModelInfo("bad", "bad.npz", "x", "", real.players, real.features_version)
    monkeypatch.setattr(netmod, "manifest", lambda: {"bad": bad})
    monkeypatch.setattr(netmod, "MODELS_DIR", tmp_path)
    load_weights.cache_clear()
    try:
        with pytest.raises(StaleModelError, match="layout hash"):
            NetBot("bad")
        assert not is_valid_seat_type("net:bad")
    finally:
        load_weights.cache_clear()


def test_weights_cached_per_name() -> None:
    assert load_weights(NAME) is load_weights(NAME)
    assert make_bot(f"net:{NAME}") is make_bot(f"net:{NAME}")


@pytest.mark.parametrize("players", [3, 4, 5])
def test_legal_over_full_games(players: int) -> None:
    for seed in range(2):
        seats = [f"net:{NAME}"] + ["random"] * (players - 1)
        _play(players, seats, seed)


def test_all_net_seats_and_sampling_legal() -> None:
    _play(4, [f"net:{NAME}"] * 4, 3)
    bot = NetBot(NAME, sample=True)
    state = pyspiel.load_game("python_bus", {"players": 4}).new_initial_state()
    rng = random.Random(1)
    while not state.is_terminal():
        a = (
            bot.choose(state, rng)
            if not state.is_chance_node()
            else rng.choice([o for o, _ in state.chance_outcomes()])
        )
        assert a in state.legal_actions()
        state.apply_action(a)


def test_greedy_is_deterministic() -> None:
    seats = [f"net:{NAME}", "heuristic", "random", f"net:{NAME}"]
    assert _play(4, seats, 11) == _play(4, seats, 11)
    # greedy ignores the rng
    s = [f"net:{NAME}"] * 3
    assert _play(3, s, 1) == _play(3, s, 2)


def test_fallback_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = NetBot(NAME)

    def boom(*a: object, **k: object) -> int:
        raise RuntimeError("injected")

    monkeypatch.setattr(bot, "_pick", boom)
    state = pyspiel.load_game("python_bus", {"players": 4}).new_initial_state()
    rng = random.Random(0)
    n = 0
    while not state.is_terminal() and n < 200:
        a = bot.choose(state, rng) if not state.is_chance_node() else state.chance_outcomes()[0][0]
        assert a in state.legal_actions()
        state.apply_action(a)
        n += 1


def test_fallback_on_illegal_result(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = NetBot(NAME)
    monkeypatch.setattr(bot, "_pick", lambda s, legal, rng: -1)
    state = pyspiel.load_game("python_bus", {"players": 4}).new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    assert bot.choose(state, random.Random(0)) in state.legal_actions()


def test_net_bot_does_not_import_torch() -> None:
    code = (
        "import sys, random, pyspiel, bus.game\n"
        "import bots.net as m\n"
        "from bots.registry import make_bot\n"
        "b = make_bot('net:ppo-long')\n"
        "s = pyspiel.load_game('python_bus', {'players': 4}).new_initial_state()\n"
        "r = random.Random(0)\n"
        "while s.is_chance_node(): s.apply_action(s.chance_outcomes()[0][0])\n"
        "a = b.choose(s, r)\n"
        "assert a in s.legal_actions()\n"
        "sys.exit(1 if any(k == 'torch' or k.startswith('torch.') for k in sys.modules) else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
