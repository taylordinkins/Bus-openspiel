"""Phase 4 unit 1: training env wrapper, observation v2, shaped return."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from bus.rules import PHASE_END, Core, Seat, returns_for
from train.env import BusEnv, shaped_return
from train.features import MAX_SEATS, OBS_SIZE, encode


def _play(env: BusEnv, seed: int) -> list[Core]:
    """Random game; returns a sample of cores seen at decisions."""
    env.reset(seed=seed)
    cores: list[Core] = []
    while not env.done:
        cores.append(env.state.core)
        env.step(env.rng.choice(env.legal_actions()))
    return cores


@pytest.mark.parametrize("n", [3, 4, 5])
def test_obs_size_constant(n: int) -> None:
    env = BusEnv(n, seed=1)
    env.reset()
    assert env.obs().shape == (OBS_SIZE,)
    assert env.obs().dtype == np.float32
    for _ in range(40):
        if env.done:
            break
        env.step(env.rng.choice(env.legal_actions()))
        if not env.done:
            o = env.obs()
            assert o.shape == (OBS_SIZE,) and np.isfinite(o).all()


def test_present_mask_and_count_onehot() -> None:
    from train.features import obs_layout

    lay = obs_layout()
    for n in (3, 4, 5):
        env = BusEnv(n)
        env.reset()
        o = env.obs()
        off = 0
        seg: dict[str, np.ndarray] = {}
        for k, sh in lay.items():
            size = int(np.prod(sh))
            seg[k] = o[off : off + size]
            off += size
        assert off == OBS_SIZE
        assert seg["present"].tolist() == [1.0] * n + [0.0] * (MAX_SEATS - n)
        assert seg["n_onehot"].argmax() == n - 3


def _relabel(core: Core, k: int) -> Core:
    """Same game with seat i renamed to (i + k) % n."""
    n = core.players
    seats: list[Seat] = list(core.seats)
    for i, s in enumerate(core.seats):
        seats[(i + k) % n] = s
    return replace(
        core,
        seats=tuple(seats),
        action_board=tuple(tuple((x + k) % n for x in row) for row in core.action_board),
        current=(core.current + k) % n,
        start_player=(core.start_player + k) % n,
        dropped=tuple(sorted((p + k) % n for p in core.dropped)),
    )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_ego_rotation_only_reorders(n: int) -> None:
    env = BusEnv(n, seed=3)
    cores = _play(env, 3)
    for core in cores[:: max(1, len(cores) // 15)]:
        base = encode(core)
        for k in range(1, n):
            assert np.array_equal(base, encode(_relabel(core, k)))


def test_ego_block_is_player_to_move() -> None:
    env = BusEnv(4, seed=5)
    cores = _play(env, 5)
    core = next(c for c in cores if c.current != 0 and c.seats[c.current].line)
    from train.features import obs_layout

    lay = obs_layout()
    off = 0
    for k, sh in lay.items():
        if k == "seat_buses":
            break
        off += int(np.prod(sh))
    buses = encode(core)[off : off + MAX_SEATS]
    for j in range(core.players):
        seat = (core.current + j) % core.players
        assert buses[j] == pytest.approx(core.seats[seat].buses / 5.0)


def test_mask_matches_legal_actions_and_no_forced_decisions() -> None:
    env = BusEnv(4, seed=2)
    env.reset()
    steps = 0
    while not env.done:
        legal = env.state.legal_actions()
        assert len(legal) > 1
        assert np.flatnonzero(env.legal_mask()).tolist() == legal
        assert env.legal_mask().shape == (376,)
        env.step(env.rng.choice(legal))
        steps += 1
    assert steps > 0
    assert env.forced_steps > 0


@pytest.mark.parametrize("n", [3, 4, 5])
def test_random_selfplay_terminates_finite(n: int) -> None:
    for seed in range(3):
        env = BusEnv(n, seed=seed)
        _play(env, seed)
        assert env.state.core.phase == PHASE_END
        assert all(math.isfinite(r) for r in env.returns())
        assert all(math.isfinite(r) for r in env.shaped_returns())


def test_shaped_return_mixes() -> None:
    for seed in range(4):
        env = BusEnv(4, seed=seed)
        _play(env, seed)
        core = env.state.core
        wl = returns_for(core, "win_loss")
        mg = returns_for(core, "margin")
        for p in range(4):
            assert shaped_return(env.state, p, 1.0) == wl[p]
            assert shaped_return(env.state, p, 0.0) == pytest.approx(mg[p])
            assert shaped_return(env.state, p, 0.7) == pytest.approx(0.7 * wl[p] + 0.3 * mg[p])
        assert env.returns() == wl
        # ties share the win: win_loss sums to zero either way
        assert sum(wl) == pytest.approx(0.0)


def test_shaped_return_independent_of_game_mode() -> None:
    a, b = BusEnv(4, "win_loss"), BusEnv(4, "margin")
    _play(a, 9)
    _play(b, 9)
    assert a.shaped_returns() == b.shaped_returns()


def test_deterministic_for_seed() -> None:
    def trace(seed: int) -> list[float]:
        env = BusEnv(4, seed=seed)
        env.reset()
        out: list[float] = []
        while not env.done:
            out.append(float(env.obs().sum()))
            env.step(env.rng.choice(env.legal_actions()))
        return out + env.returns()

    assert trace(7) == trace(7)
    assert trace(7) != trace(8)
