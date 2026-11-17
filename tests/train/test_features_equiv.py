# pyright: reportPrivateUsage=false
"""Phase 4 unit 3b: the fast observation encoder equals the original implementation.

``reference_encode`` is the pre-optimisation ``train.features.encode`` (BusObserver based),
kept verbatim here so any drift of the fast encoder fails the suite.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bus import constants as C
from bus.actions import N, S
from bus.lines import legal_extensions, line_nodes
from bus.observer import BusObserver
from bus.rules import EMPTY, Core, board
from train import features as F
from train.env import BusEnv

_SITES_PER_NODE = float(max(len(x) for x in board().node_sites))
_REF_OBSERVERS: dict[int, BusObserver] = {}


def _rotate(view: np.ndarray[Any, Any], axis: int, perm: list[int]) -> np.ndarray[Any, Any]:
    taken = np.take(view, perm, axis=axis)
    shape = list(taken.shape)
    shape[axis] = F.MAX_SEATS
    out = np.zeros(shape, np.float32)
    idx: list[Any] = [slice(None)] * len(shape)
    idx[axis] = slice(0, len(perm))
    out[tuple(idx)] = taken
    return out


def _tiebreak(core: Core, perm: list[int]) -> np.ndarray[Any, Any]:
    out = np.zeros(F.MAX_SEATS, np.float32)
    scored = sorted(
        (s.last_score_move, i) for i, s in enumerate(core.seats) if s.last_score_move is not None
    )
    m = len(scored)
    pos = {seat: r for r, (_, seat) in enumerate(scored)}
    for j, seat in enumerate(perm):
        if seat in pos:
            out[j] = (m - pos[seat]) / m
    return out


def reference_encode(core: Core) -> np.ndarray[Any, Any]:
    n = core.players
    me = core.current
    perm = [(me + i) % n for i in range(n)]
    b = board()
    ob = _REF_OBSERVERS.setdefault(n, BusObserver("win_loss", n))
    ob.set_from(SimpleNamespace(core=core), me)
    d = ob.dict
    parts: dict[str, np.ndarray[Any, Any]] = {}
    for k in F._PLAIN:
        parts[k] = d[k]
    for k, ax in F._SEATED.items():
        parts[k] = _rotate(d[k], ax, perm)
    onehot = np.zeros(len(F.PLAYER_COUNTS), np.float32)
    onehot[n - C.MIN_PLAYERS] = 1.0
    parts["n_onehot"] = onehot
    present = np.zeros(F.MAX_SEATS, np.float32)
    present[:n] = 1.0
    parts["present"] = present
    score = np.zeros(F.MAX_SEATS, np.float32)
    for j, seat in enumerate(perm):
        s = core.seats[seat]
        score[j] = min(1.0, max(-1.0, (s.points - s.stones) / 10.0))
    parts["score"] = score
    parts["tiebreak"] = _tiebreak(core, perm)

    lines = [s.line for s in core.seats]
    mine: set[int] = set(line_nodes(b, lines[me])) if lines[me] else set()
    pickable = np.zeros(N, np.float32)
    free = np.zeros(N, np.float32)
    opp_at = np.zeros(N, np.float32)
    waiting_on = 0
    free_on = 0
    for node in mine:
        pickable[node] = core.waiting[node] / C.TOTAL_PASSENGERS
        waiting_on += core.waiting[node]
        e = sum(1 for t in b.node_sites[node] if core.site_types[t] == EMPTY)
        free[node] = e / _SITES_PER_NODE
        free_on += e
    contested = np.zeros(S, np.float32)
    frontier = np.zeros((S, F.MAX_SEATS), np.float32)
    for j, seat in enumerate(perm):
        line = lines[seat]
        if not line:
            continue
        if seat != me:
            contested[list(line)] = 1.0
            for node in line_nodes(b, line):
                opp_at[node] += 1.0 / (F.MAX_SEATS - 1)
        for _end, street in legal_extensions(b, lines, seat):
            frontier[street, j] = 1.0
    parts["ego_pickable"] = pickable
    parts["ego_free_sites"] = free
    parts["opp_lines_at_node"] = opp_at
    parts["street_contested"] = contested
    parts["frontier"] = frontier
    deliveries = min(core.seats[me].buses, waiting_on, free_on)
    parts["vroom"] = np.array(
        [waiting_on / C.TOTAL_PASSENGERS, free_on / 10.0, deliveries / float(C.BUSES_PER_PLAYER)],
        np.float32,
    )
    return np.concatenate([parts[k].reshape(-1) for k in F._SHAPES]).astype(np.float32, copy=False)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_fast_encode_equals_reference(n: int) -> None:
    env = BusEnv(n, seed=n)
    checked = 0
    phases: set[str] = set()
    for g in range(14):
        env.reset(seed=100 * n + g)
        while True:
            core = env.state.core
            new, ref = F.encode(core), reference_encode(core)
            assert new.dtype == np.float32 and new.shape == ref.shape
            assert np.array_equal(new, ref), (n, g, core.phase, np.flatnonzero(new != ref)[:10])
            checked += 1
            phases.add(core.phase)
            if env.done:
                break
            env.step(env.rng.choice(env.legal_actions()))
    assert checked > 1000
    assert len(phases) >= 6


def test_encode_returns_fresh_array() -> None:
    env = BusEnv(4, seed=0)
    env.reset(seed=3)
    a = F.encode(env.state.core)
    snap = a.copy()
    env.step(env.rng.choice(env.legal_actions()))
    F.encode(env.state.core)
    assert np.array_equal(a, snap)
