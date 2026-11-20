"""Smoke tests for the Gumbel search spike in ``tools.bench_search`` (Phase 4 unit 8f)."""

from __future__ import annotations

from typing import Any

import numpy as np

from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS
from tools.bench_search import (
    GumbelSearch,
    considered_visits,
    improved_policy,
    net_cores,
    selfplay,
    skip_forced,
)
from train.features import encode

Array = np.ndarray[Any, Any]


def _fake_forward(obs: Array, mask: Array) -> tuple[Array, Array]:
    """Uniform legal logits; values depend on the observation so searches are not degenerate."""
    logits = np.where(mask, 0.0, -1e8).astype(np.float32)
    v = np.tanh(obs[:, :5] - 0.5).astype(np.float32)
    return logits, v


def test_sequential_halving_schedule() -> None:
    # m=4, n=8: one visit to each of 4 actions, then 2 survivors get 2 more each.
    assert considered_visits(4, 8) == (0, 0, 0, 0, 1, 1, 2, 2)
    assert considered_visits(1, 3) == (0, 1, 2)
    assert len(considered_visits(16, 64)) == 64


def _run_search(core: Any, sims: int, seed: int) -> GumbelSearch:
    s = GumbelSearch(core, sims, np.random.default_rng(seed))
    c = s.want()
    while c is not None:
        legal = rules.legal_actions(c)
        mask = np.zeros((1, NUM_DISTINCT_ACTIONS), np.bool_)
        mask[0, legal] = True
        lg, v = _fake_forward(encode(c)[None, :], mask)
        s.give(lg[0], v[0])
        c = s.want()
    return s


def test_search_returns_legal_action_and_valid_target() -> None:
    for players in (3, 4, 5):
        core, legal = skip_forced(rules.initial_core(players))
        s = _run_search(core, 32, players)
        a, pi, acts = s.result()
        assert a in legal
        assert sorted(acts.tolist()) == sorted(legal)
        assert abs(float(pi.sum()) - 1.0) < 1e-9
        root = s.root
        assert root is not None
        assert int(root.N.sum()) == 32  # every simulation is backed up through the root
        # the vector backup keeps one entry per seat
        assert root.W.shape == (len(legal), players)
        assert np.allclose(improved_policy(root), pi)


def test_selfplay_runs_briefly() -> None:
    starts = net_cores(_fake_forward, 4, 2, 0)
    r = selfplay(_fake_forward, 4, 8, 4, 0.3, starts=starts)
    assert r["sims/s"] > 0
    assert r["mean batch"] == 4
