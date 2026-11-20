# pyright: basic
"""Phase 4 unit 8g: Gumbel search seat `gumbel:<sims>-<model>` (`bots/gumbel.py`)."""

from __future__ import annotations

import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyspiel
import pytest

import bus.game  # noqa: F401  (import registers the game)
from bots import gumbel as gb
from bots.gumbel import (
    GumbelConfig,
    GumbelSearch,
    Node,
    completed_q,
    considered_visits,
    improved_policy,
    run_searches,
    terminal_values,
)
from bots.net import check_players, load_weights
from bots.registry import is_valid_seat_type, make_bot, seat_type_error
from bus import rules
from tools.eval_search import parse_agent

NAME = "ppo-long"


def _core(players: int, plies: int, seed: int = 0) -> Any:
    """A decision core (>= 2 legal actions) after ``plies`` random moves."""
    rng = random.Random(seed)
    core = rules.initial_core(players)
    for _ in range(plies):
        legal = rules.legal_actions(core)
        if core.phase == rules.PHASE_END:
            break
        core = rules.apply(core, rng.choice(legal))
    core, legal = gb.skip_forced(core)
    assert len(legal) >= 2
    return core


def _final_core(players: int, seed: int) -> Any:
    rng = random.Random(seed)
    core = rules.initial_core(players)
    while core.phase != rules.PHASE_END:
        core = rules.apply(core, rng.choice(rules.legal_actions(core)))
    return core


def _all_nodes(root: Node) -> list[Node]:
    out, stack = [], [root]
    while stack:
        nd = stack.pop()
        out.append(nd)
        stack.extend(c for c in nd.children if c is not None)
    return out


def _drive(search: GumbelSearch, seat_value: np.ndarray) -> int:
    """Run ``search`` with uniform logits and absolute per-seat values ``seat_value``.

    The fake net returns them in ego order for each leaf's mover; returns the evaluations made.
    """
    n = search.n
    evals = 0
    while (core := search.want()) is not None:
        mover = int(core.current)
        v_ego = np.zeros(5)
        for k in range(n):
            v_ego[k] = seat_value[(mover + k) % n]
        search.give(np.zeros(gb.NUM_ACTIONS), v_ego)
        evals += 1
    return evals


# ------------------------------------------------------------------ schedule


def test_considered_visits_hand_cases() -> None:
    # m = 4, n = 8: one visit each, then halving to 2 actions with 2 visits each
    assert considered_visits(4, 8) == (0, 0, 0, 0, 1, 1, 2, 2)
    assert considered_visits(2, 5) == (0, 0, 1, 1, 2)
    assert considered_visits(1, 4) == (0, 1, 2, 3)
    assert considered_visits(16, 8) == (0,) * 8


@pytest.mark.parametrize(("m", "n"), [(16, 32), (16, 64), (5, 16), (3, 7), (8, 100)])
def test_considered_visits_is_sequential_halving(m: int, n: int) -> None:
    seq = considered_visits(m, n)
    assert len(seq) == n
    # simulate: each entry is given to the next action with exactly that many visits
    visits = [0] * m
    for c in seq:
        i = next(j for j, v in enumerate(visits) if v == c)
        visits[i] += 1
    assert sum(visits) == n
    assert visits[:m] == sorted(visits, reverse=True)
    assert min(visits) >= 1  # every considered action is visited at least once (n >= m)


def test_search_root_visits_follow_schedule() -> None:
    core = next(
        c for p in range(10, 200, 7) if len(rules.legal_actions(c := _core(4, p, seed=p))) >= 4
    )
    s = GumbelSearch(core, 8, GumbelConfig(m=4))
    _drive(s, np.array([0.1, 0.2, 0.3, 0.4]))
    assert s.root.tot == 8
    got = sorted(s.root.N.tolist(), reverse=True)
    assert got[:4] == [3, 3, 1, 1] and sum(got[4:]) == 0


# ------------------------------------------------------------------ completed Q


def _hand_node() -> Node:
    fake = SimpleNamespace(phase="PLAY", current=1, players=3)
    nd = Node(fake, [10, 11, 12], 3, 0.5)
    p = np.array([0.5, 0.3, 0.2])
    nd.logits = np.log(p) - np.log(p).max()
    nd.prior = p
    nd.value = np.array([0.9, 0.2, -0.9])  # mover is seat 1: only 0.2 matters
    nd.N = np.array([2, 1, 0])
    nd.tot, nd.nmax = 3, 2
    nd.W = np.array([[5.0, 0.8, -3.0], [-7.0, -0.1, 4.0], [0.0, 0.0, 0.0]])
    return nd


def test_completed_q_hand_case() -> None:
    nd = _hand_node()
    # visited q (seat 1): 0.4, -0.1; v_mix = (0.2 + 3 * (0.5*0.4 + 0.3*-0.1) / 0.8) / 4
    v_mix = (0.2 + 3 * (0.17 / 0.8)) / 4
    assert v_mix == pytest.approx(0.209375)
    np.testing.assert_allclose(completed_q(nd), [1.0, 0.0, (v_mix + 0.1) / 0.5])


def test_improved_policy_hand_case() -> None:
    nd = _hand_node()
    cfg = GumbelConfig(c_visit=50.0, c_scale=0.1)
    sig = (50 + 2) * 0.1 * np.array([1.0, 0.0, 0.61875])
    z = np.log([0.5, 0.3, 0.2]) + sig
    want = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    np.testing.assert_allclose(improved_policy(nd, cfg), want)
    # c_scale 0: improved policy is the prior
    np.testing.assert_allclose(improved_policy(nd, GumbelConfig(c_scale=0.0)), [0.5, 0.3, 0.2])


def test_completed_q_unvisited_node_is_flat() -> None:
    fake = SimpleNamespace(phase="PLAY", current=0, players=3)
    nd = Node(fake, [1, 2], 3, 0.5)
    np.testing.assert_allclose(completed_q(nd), [0.0, 0.0])


# ------------------------------------------------------------------ backup, forced moves


@pytest.mark.parametrize("players", [3, 4, 5])
def test_vector_backup_rotates_and_sums_full_vectors(players: int) -> None:
    seat_value = np.linspace(-0.6, 0.8, players)
    core = _core(players, 20, seed=players)
    s = GumbelSearch(core, 24, GumbelConfig())
    _drive(s, seat_value)
    for nd in _all_nodes(s.root):
        if nd.terminal:
            continue
        # the ego value was rotated back to absolute seats at every node
        np.testing.assert_allclose(nd.value, seat_value)
        for a, child in enumerate(nd.children):
            if child is None:
                assert nd.N[a] == 0
                continue
            assert child.mover != -1 or child.terminal
            if not child.terminal:
                # one visit per evaluation of the child plus everything below it
                assert nd.N[a] == 1 + child.tot
                np.testing.assert_allclose(nd.W[a], child.value + child.W.sum(axis=0), atol=1e-9)
        assert nd.tot == int(nd.N.sum())
        assert nd.nmax == int(nd.N.max(initial=0))
    res = s.result()
    np.testing.assert_allclose(res.value, seat_value)  # no terminal reached this early
    assert res.policy.sum() == pytest.approx(1.0)


def test_terminal_values_are_relative_blend() -> None:
    for seed in range(3):
        core = _final_core(4, seed)
        wl = np.array(rules.returns_for(core, "win_loss"))
        mg = np.array(rules.returns_for(core, "margin"))
        np.testing.assert_allclose(terminal_values(core), 0.5 * wl + 0.5 * mg)
        np.testing.assert_allclose(terminal_values(core, 1.0), wl)


def test_terminal_leaves_use_r_without_evaluation() -> None:
    # search near the end of a game: terminal arrivals are simulations without a forward
    rng = random.Random(5)
    core = rules.initial_core(3)
    history = []
    while core.phase != rules.PHASE_END:
        history.append(core)
        core = rules.apply(core, rng.choice(rules.legal_actions(core)))
    root = next(c for c in reversed(history) if len(rules.legal_actions(c)) >= 2)
    s = GumbelSearch(root, 16, GumbelConfig())
    evals = _drive(s, np.zeros(3))
    term = [nd for nd in _all_nodes(s.root) if nd.terminal]
    assert term, "expected terminal nodes in an end-game search"
    for nd in term:
        np.testing.assert_allclose(nd.value, terminal_values(nd.core))
    assert s.done_sims == 16
    assert evals < 1 + 16


def _root_with_forced_edge(players: int) -> tuple[Any, int]:
    for seed in range(200):
        rng = random.Random(seed)
        core = rules.initial_core(players)
        for _ in range(400):
            if core.phase == rules.PHASE_END:
                break
            legal = rules.legal_actions(core)
            if 2 <= len(legal) <= 8:
                for a in legal:
                    nxt = rules.apply(core, a)
                    if nxt.phase != rules.PHASE_END and len(rules.legal_actions(nxt)) == 1:
                        return core, a
            core = rules.apply(core, rng.choice(legal))
    raise AssertionError("no forced edge found")


def test_forced_edges_cost_no_simulations() -> None:
    core, a_forced = _root_with_forced_edge(4)
    legal = rules.legal_actions(core)
    sims = 4 * len(legal)
    s = GumbelSearch(core, sims, GumbelConfig(m=16))
    evals = _drive(s, np.zeros(4))
    assert s.done_sims == sims
    n_term = 0
    for nd in _all_nodes(s.root):
        assert nd.terminal or len(nd.actions) >= 2  # no forced node is ever stored
        n_term += nd.terminal
    # every simulation was one evaluation (plus the root's), unless it ended at a terminal
    assert evals <= 1 + sims
    if n_term == 0:
        assert evals == 1 + sims
    i = int(np.flatnonzero(s.root.actions == a_forced)[0])
    child = s.root.children[i]
    assert child is not None and s.root.N[i] >= 1
    assert child.core != rules.apply(core, a_forced)  # the forced move was applied in expansion


# ------------------------------------------------------------------ the seat


def _state(players: int, plies: int, seed: int = 0) -> pyspiel.State:
    s = pyspiel.load_game("python_bus", {"players": players}).new_initial_state()
    rng = random.Random(seed)
    for _ in range(plies):
        if s.is_terminal():
            break
        s.apply_action(rng.choice(s.legal_actions()))
    return s


def test_seat_types_and_players() -> None:
    assert is_valid_seat_type(f"gumbel:8-{NAME}")
    for bad in ("gumbel", "gumbel:8", "gumbel:x-ppo-long", "gumbel:0-ppo-long", "gumbel:8-nope"):
        assert not is_valid_seat_type(bad), bad
        assert seat_type_error(bad) is not None
    check_players(f"gumbel:8-{NAME}", 4)


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GumbelConfig.from_env() == GumbelConfig()
    monkeypatch.setenv("BUS_GUMBEL_C_SCALE", "0.3")
    monkeypatch.setenv("BUS_GUMBEL_C_VISIT", "25")
    monkeypatch.setenv("BUS_GUMBEL_M", "8")
    bot = gb.parse_gumbel_param(f"4-{NAME}")
    assert (bot.config.c_scale, bot.config.c_visit, bot.config.m) == (0.3, 25.0, 8)
    assert bot.config.gumbel_scale == 0.0


def test_eval_search_candidate_spec() -> None:
    s = parse_agent("gumbel:32,c=0.03,v=40,m=8")
    assert s.kind == "gumbel" and s.sims == 32
    assert s.gumbel == GumbelConfig(m=8, c_visit=40.0, c_scale=0.03)
    assert parse_agent("gumbel:16").gumbel == GumbelConfig()
    for bad in ("gumbel:", "gumbel:0", "gumbel:8,x=1", "gumbel:8,c=-1"):
        with pytest.raises(ValueError):
            parse_agent(bad)


@pytest.mark.parametrize("players", [3, 4, 5])
def test_legal_results_in_play(players: int) -> None:
    bot = make_bot(f"gumbel:4-{NAME}")
    for plies in (0, 40, 120):
        st = _state(players, plies, seed=plies + players)
        if st.is_terminal():
            continue
        a = bot.choose(st, random.Random(0))
        assert a in st.legal_actions()


def test_full_game_with_gumbel_seat() -> None:
    state = pyspiel.load_game("python_bus", {"players": 3}).new_initial_state()
    rng = random.Random(1)
    bots = [make_bot(f"gumbel:2-{NAME}"), make_bot(f"net:{NAME}"), make_bot("heuristic")]
    while not state.is_terminal():
        a = bots[state.current_player()].choose(state, rng)
        assert a in state.legal_actions()
        state.apply_action(a)


def test_deterministic_given_seed() -> None:
    w = load_weights(NAME)
    core = _core(4, 25, seed=9)
    cfg = GumbelConfig(gumbel_scale=1.0)
    r1 = gb.GumbelBot(w, 16, cfg).search(core, random.Random(7))
    r2 = gb.GumbelBot(w, 16, cfg).search(core, random.Random(7))
    assert r1.action == r2.action
    np.testing.assert_array_equal(r1.policy, r2.policy)
    np.testing.assert_array_equal(r1.visits, r2.visits)
    # g = 0 does not touch the game rng (common random numbers with the greedy net seat)
    rng = random.Random(3)
    before = rng.getstate()
    gb.GumbelBot(w, 8, GumbelConfig()).search(core, rng)
    assert rng.getstate() == before


def test_batched_searches_match_single() -> None:
    w = load_weights(NAME)
    cores = [_core(4, p, seed=p) for p in (10, 30, 50)]
    single = [run_searches(w.forward, [GumbelSearch(c, 12)])[0] for c in cores]
    batch = run_searches(w.forward, [GumbelSearch(c, 12) for c in cores])
    for c, a, b in zip(cores, single, batch, strict=True):
        assert a.action in rules.legal_actions(c)
        assert a.action == b.action
        np.testing.assert_allclose(a.policy, b.policy, atol=1e-4)  # float32 batch
        np.testing.assert_array_equal(a.visits, b.visits)


def test_does_not_import_torch() -> None:
    code = (
        "import sys, random, pyspiel, bus.game\n"
        "import bots.gumbel\n"
        "from bots.registry import make_bot\n"
        "b = make_bot('gumbel:3-ppo-long')\n"
        "s = pyspiel.load_game('python_bus', {'players': 4}).new_initial_state()\n"
        "r = random.Random(0)\n"
        "for _ in range(40): s.apply_action(r.choice(s.legal_actions()))\n"
        "assert b.choose(s, r) in s.legal_actions()\n"
        "sys.exit(1 if any(k == 'torch' or k.startswith('torch.') for k in sys.modules) else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sparse_forward_matches_dense() -> None:
    w = load_weights(NAME)
    cores = [_core(n, p, seed=p) for n in (3, 4, 5) for p in (0, 40, 80)]
    from train.features import encode

    obs = np.stack([encode(c) for c in cores]).astype(np.float32)
    mask = np.zeros((len(cores), gb.NUM_ACTIONS), bool)
    for i, c in enumerate(cores):
        mask[i, rules.legal_actions(c)] = True
    fwd = gb.sparse_forward(w)
    for rows in (slice(0, 1), slice(None)):
        dl, dv = w.forward(obs[rows], mask[rows])
        sl, sv = fwd(obs[rows], mask[rows])
        np.testing.assert_allclose(sl, dl, atol=1e-4)
        np.testing.assert_allclose(sv, dv, atol=1e-5)
