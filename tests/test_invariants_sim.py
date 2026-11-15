"""Seeded random-game invariant sweeps (PLAN section 6 item 3, Phase 1 exit).

``BUS_SIM_GAMES`` sets the total game count of the slow sweep (default 10,000). The fast variant
plays ``BUS_SIM_FAST_GAMES`` (default 20) games.
"""

from __future__ import annotations

import os
import random

import pytest

from tests.support.invariants import check_invariants
from tests.support.play import dump, game, key, reload

CONFIGS = [(n, m) for n in (3, 4, 5) for m in ("win_loss", "margin")]
SAMPLE_EVERY = 25


def _play(n: int, mode: str, seed: int) -> int:
    rng = random.Random(seed)
    g = game(n, returns=mode)
    s = g.new_initial_state()
    step = 0

    def fail(msg: str) -> None:
        pytest.fail(f"seed={seed} players={n} returns={mode} step={step}: {msg}")

    bad = check_invariants(s)
    if bad:
        fail("; ".join(bad))
    while not s.is_terminal():
        prev = s.clone()
        s.apply_action(rng.choice(s.legal_actions()))
        step += 1
        bad = check_invariants(s, prev)
        if bad:
            fail("; ".join(bad))
        if step % SAMPLE_EVERY == 0 or s.is_terminal():
            r = reload(s)
            if dump(r) != dump(s):
                fail("dump/load/dump differs")
            if str(r) != str(s) or key(r) != key(s):
                fail("str(load(dump(s))) != str(s)")
    lo, hi = g.min_utility(), g.max_utility()
    rets = s.returns()
    if len(rets) != n or any(not lo - 1e-9 <= r <= hi + 1e-9 for r in rets):
        fail(f"returns {rets} outside [{lo}, {hi}]")
    return step


def _run(total: int, offset: int) -> None:
    for i in range(total):
        n, mode = CONFIGS[i % len(CONFIGS)]
        _play(n, mode, offset + i)


def test_SIM_fast_random_games_keep_invariants() -> None:
    _run(int(os.environ.get("BUS_SIM_FAST_GAMES", "20")), 0)


@pytest.mark.slow
def test_SIM_random_games_keep_invariants() -> None:
    _run(int(os.environ.get("BUS_SIM_GAMES", "10000")), 1_000_000)
