"""Phase 4 interaction directions 1 and 3: opponent model in the Gumbel search, value targets.

Training signals only (no rule behaviour): the targets read R-068 final scores and the R-071
win_loss return; the opponent model only changes search priors.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from bots import gumbel as gb
from bots.gumbel import (
    GumbelConfig,
    GumbelSearch,
    Node,
    opponent_prior,
    relative_term,
    terminal_values,
)
from bots.net import load_weights
from bus import rules
from tools.eval_search import parse_agent

NAME = "ppo-long"


def _core(players: int, plies: int, seed: int = 0) -> Any:
    rng = random.Random(seed)
    core = rules.initial_core(players)
    for _ in range(plies):
        if core.phase == rules.PHASE_END:
            break
        core = rules.apply(core, rng.choice(rules.legal_actions(core)))
    core, legal = gb.skip_forced(core)
    assert len(legal) >= 2
    return core


def _final_core(players: int, seed: int = 0) -> Any:
    rng = random.Random(seed)
    core = rules.initial_core(players)
    while core.phase != rules.PHASE_END:
        core = rules.apply(core, rng.choice(rules.legal_actions(core)))
    return core


def _with_scores(core: Any, points: list[int], stones: list[int], last: list[int | None]) -> Any:
    seats = tuple(
        replace(s, points=p, stones=t, last_score_move=m)
        for s, p, t, m in zip(core.seats, points, stones, last, strict=True)
    )
    return replace(core, seats=seats)


def _all_nodes(root: Node) -> list[Node]:
    out, stack = [], [root]
    while stack:
        nd = stack.pop()
        out.append(nd)
        stack.extend(c for c in nd.children if c is not None)
    return out


def _sharp_logits(core: Any) -> np.ndarray:
    """Masked full logits with the lowest legal action 8 nats above the rest (a sharp prior)."""
    lg = np.full(gb.NUM_ACTIONS, -1e8)
    legal = rules.legal_actions(core)
    lg[legal] = 0.0
    lg[legal[0]] = 8.0
    return lg


def _drive_sharp(search: GumbelSearch) -> None:
    """Fake net: sharp prior, all-zero values (so completed Q never separates actions)."""
    while (core := search.want()) is not None:
        search.give(_sharp_logits(core), np.zeros(5))


# ------------------------------------------------------------------ default unchanged

# Root results of the unit 8g search on `ppo-long`, recorded on this branch before the opponent
# model and value targets were added (key: players-plies-seed-sims-gumbel_scale; c_scale 0.1 for
# g = 0 and 0.3 for g = 1; GumbelBot.search with random.Random(seed)).
_REFERENCE = {
    "3-30-1-32-0.0": [289, [16, 16], [0.626519, 0.373481], [-0.068112, 0.481772, -0.387563]],
    "4-60-2-64-0.0": [
        0, [26, 26, 4, 4, 4], [1.0, 0.0, 0.0, 0.0, 0.0], [-0.007214, -0.210501, -0.055595, 0.28562],
    ],
    "5-90-3-32-1.0": [
        2, [16, 16], [0.898733, 0.101267], [0.408164, -0.078759, -0.078713, -0.201698, -0.102996],
    ],
    "4-120-4-48-1.0": [
        203, [1, 1, 0, 1, 1, 2, 2, 1, 11, 2, 11, 0, 1, 1, 5, 1, 5, 2],
        [0.000355, 0.000239, 0.0, 0.041483, 0.000823, 0.027634, 0.000504, 0.005658, 0.0, 0.078945,
         3e-06, 0.0, 0.018727, 0.002547, 1e-06, 0.021132, 0.0, 0.80195],
        [0.040098, 0.160115, 0.0184, -0.30011],
    ],
}  # fmt: skip


@pytest.mark.parametrize("key", sorted(_REFERENCE))
def test_default_search_matches_pre_change_reference(key: str) -> None:
    n, plies, seed, sims, gs = key.split("-")
    core = _core(int(n), int(plies), int(seed))
    g = float(gs)
    cfg = GumbelConfig(c_scale=0.1 if g == 0 else 0.3, gumbel_scale=g)
    r = gb.GumbelBot(load_weights(NAME), int(sims), cfg).search(core, random.Random(int(seed)))
    action, visits, policy, value = _REFERENCE[key]
    assert r.action == action
    assert r.visits.tolist() == visits
    np.testing.assert_allclose(r.policy, policy, atol=2e-6)
    np.testing.assert_allclose(r.value, value, atol=2e-6)


def test_identity_opponent_settings_are_bit_identical_to_default() -> None:
    w = load_weights(NAME)
    core = _core(4, 70, seed=11)
    base = gb.GumbelBot(w, 48, GumbelConfig(gumbel_scale=1.0)).search(core, random.Random(2))
    for scope in ("opponents", "nonroot"):
        cfg = GumbelConfig(gumbel_scale=1.0, opp_temp=1.0, opp_mix=0.0, opp_scope=scope)
        assert not cfg.models_opponents
        r = gb.GumbelBot(w, 48, cfg).search(core, random.Random(2))
        assert r.action == base.action
        np.testing.assert_array_equal(r.visits, base.visits)
        np.testing.assert_array_equal(r.policy, base.policy)
        np.testing.assert_array_equal(r.value, base.value)


def test_default_terminal_values_unchanged() -> None:
    for seed in range(3):
        core = _final_core(4, seed)
        wl = np.array(rules.returns_for(core, "win_loss"))
        mg = np.array(rules.returns_for(core, "margin"))
        np.testing.assert_array_equal(terminal_values(core), 0.5 * wl + 0.5 * mg)
        np.testing.assert_array_equal(terminal_values(core, 0.5, "mean"), terminal_values(core))


# ------------------------------------------------------------------ opponent model


def test_opponent_prior_hand_case() -> None:
    lg = np.array([2.0, 0.0, 0.0])
    p = np.exp(opponent_prior(lg, 2.0, 0.0))
    e = np.exp([1.0, 0.0, 0.0])
    np.testing.assert_allclose(p, e / e.sum())
    p = np.exp(opponent_prior(lg, 1.0, 0.5))
    e = np.exp(lg) / np.exp(lg).sum()
    np.testing.assert_allclose(p, 0.5 * e + 0.5 / 3)
    np.testing.assert_allclose(np.exp(opponent_prior(lg, 1e6, 0.0)), [1 / 3] * 3, atol=1e-5)


@pytest.mark.parametrize("scope", ["opponents", "nonroot"])
def test_opponent_model_changes_the_intended_priors(scope: str) -> None:
    core = _core(4, 45, seed=4)
    cfg = GumbelConfig(opp_temp=3.0, opp_mix=0.2, opp_scope=scope)
    s = GumbelSearch(core, 64, cfg)
    _drive_sharp(s)
    root = s.root
    lg = _sharp_logits(core)[root.actions]
    np.testing.assert_allclose(root.prior, np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum())
    changed = same = 0
    for nd in _all_nodes(root):
        if nd is root or nd.terminal:
            continue
        lg = _sharp_logits(nd.core)[nd.actions]
        net = np.exp(lg - lg.max())
        net /= net.sum()
        model = np.exp(opponent_prior(lg, 3.0, 0.2))
        model /= model.sum()
        if scope == "nonroot" or nd.mover != root.mover:
            np.testing.assert_allclose(nd.prior, model)
            changed += 1
        else:
            np.testing.assert_allclose(nd.prior, net)
            same += 1
    assert changed > 0
    if scope == "nonroot":
        assert same == 0


def _off_argmax_share(cfg: GumbelConfig, cores: list[Any]) -> float:
    """Share of opponent-node visits that go to an action other than the net prior's argmax."""
    off = visits = 0
    for core in cores:
        s = GumbelSearch(core, 96, cfg)
        _drive_sharp(s)
        for nd in _all_nodes(s.root):
            if nd is s.root or nd.terminal or nd.mover == s.root.mover:
                continue
            top = int(np.argmax(_sharp_logits(nd.core)[nd.actions]))
            off += nd.tot - int(nd.N[top])
            visits += nd.tot
    return off / visits


def test_opponent_model_spreads_opponent_visits() -> None:
    cores = [_core(4, p, seed=p) for p in (20, 45, 70, 95)]
    base = _off_argmax_share(GumbelConfig(), cores)
    flat = _off_argmax_share(GumbelConfig(opp_temp=4.0), cores)
    mixed = _off_argmax_share(GumbelConfig(opp_mix=0.5), cores)
    assert base < 0.05  # sharp prior, flat values: opponents play the argmax
    assert flat > base + 0.15 and mixed > base + 0.15


def test_opponent_model_leaves_root_alone_with_a_real_net() -> None:
    w = load_weights(NAME)
    core = _core(4, 60, seed=2)
    a = GumbelSearch(core, 32, GumbelConfig())
    b = GumbelSearch(core, 32, GumbelConfig(opp_temp=3.0))
    gb.run_searches(w.forward, [a])
    gb.run_searches(w.forward, [b])
    np.testing.assert_array_equal(a.root.prior, b.root.prior)
    np.testing.assert_array_equal(a.root.value, b.root.value)


def test_opponent_model_config_validation_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in ({"opp_temp": 0.0}, {"opp_mix": 1.5}, {"opp_scope": "all"}, {"target": "x"}):
        with pytest.raises(ValueError):
            GumbelConfig(**bad)
    monkeypatch.setenv("BUS_GUMBEL_OPP_TEMP", "2.5")
    monkeypatch.setenv("BUS_GUMBEL_OPP_MIX", "0.1")
    monkeypatch.setenv("BUS_GUMBEL_OPP_SCOPE", "nonroot")
    monkeypatch.setenv("BUS_GUMBEL_TARGET", "leader")
    cfg = gb.parse_gumbel_param(f"4-{NAME}").config
    assert (cfg.opp_temp, cfg.opp_mix, cfg.opp_scope, cfg.target) == (2.5, 0.1, "nonroot", "leader")


def test_eval_search_opponent_options() -> None:
    s = parse_agent("gumbel:64,c=0.1,t=2,e=0.1,scope=nonroot")
    assert s.gumbel == GumbelConfig(c_scale=0.1, opp_temp=2.0, opp_mix=0.1, opp_scope="nonroot")
    for bad in ("gumbel:8,t=0", "gumbel:8,scope=x", "gumbel:8,e=2"):
        with pytest.raises(ValueError):
            parse_agent(bad)


def test_az_trainer_flags_reach_the_actor_config() -> None:
    pytest.importorskip("torch")
    from train.az_trainer import actor_config, build_parser

    args = build_parser().parse_args(
        ["--opp-temp", "2", "--opp-mix", "0.1", "--opp-scope", "nonroot",
         "--value-quantity", "leader"]
    )  # fmt: skip
    cfg = actor_config(args).search
    assert (cfg.opp_temp, cfg.opp_mix, cfg.opp_scope, cfg.target) == (2.0, 0.1, "nonroot", "leader")
    default = actor_config(build_parser().parse_args([])).search
    assert not default.models_opponents and default.target == "mean"


# ------------------------------------------------------------------ value targets


def test_relative_terms_hand_case_with_tie_at_the_top() -> None:
    # final scores (R-068): 10, 7, 10 - 0, 4 - 1 = 3
    core = _with_scores(_final_core(4), [10, 7, 10, 4], [0, 0, 0, 1], [5, 9, 7, 3])
    np.testing.assert_allclose(relative_term(core, "leader"), [0.0, -0.3, 0.0, -0.7])
    np.testing.assert_allclose(relative_term(core, "rank"), [2 / 3, -1 / 3, 2 / 3, -1.0])
    np.testing.assert_allclose(
        relative_term(core, "mean"), [(10 - 20 / 3) / 10, (7 - 23 / 3) / 10, (10 - 20 / 3) / 10,
                                      (3 - 9) / 10]
    )  # fmt: skip
    # seat 0 reached 10 first (R-070): sole winner, +1; others -1/3 (R-071)
    wl = np.array([1.0, -1 / 3, -1 / 3, -1 / 3])
    np.testing.assert_allclose(rules.returns_for(core, "win_loss"), wl)
    np.testing.assert_allclose(
        terminal_values(core, 0.5, "leader"), 0.5 * wl + 0.5 * np.array([0.0, -0.3, 0.0, -0.7])
    )
    np.testing.assert_allclose(terminal_values(core, 1.0, "rank"), wl)


def test_relative_terms_sole_leader_and_clipping() -> None:
    core = _with_scores(_final_core(3), [25, 0, 2], [0, 0, 0], [8, None, 4])
    np.testing.assert_allclose(relative_term(core, "leader"), [1.0, -1.0, -1.0])  # clipped
    np.testing.assert_allclose(relative_term(core, "rank"), [1.0, -1.0, 0.0])
    core = _with_scores(_final_core(3), [6, 4, 2], [0, 0, 0], [8, 6, 4])
    np.testing.assert_allclose(relative_term(core, "leader"), [0.2, -0.2, -0.4])


def test_relative_terms_all_tied() -> None:
    core = _with_scores(_final_core(5), [5] * 5, [1] * 5, [None] * 5)
    for t in ("leader", "rank", "mean"):
        np.testing.assert_allclose(relative_term(core, t), np.zeros(5))
    np.testing.assert_allclose(terminal_values(core, 0.5, "rank"), np.zeros(5))  # 5-way share


def test_relative_terms_are_zero_before_the_end_and_bounded() -> None:
    core = _core(4, 30)
    for t in ("leader", "rank"):
        np.testing.assert_array_equal(relative_term(core, t), np.zeros(4))
    for seed in range(4):
        fin = _final_core(4, seed)
        for t in ("leader", "rank"):
            r = relative_term(fin, t)
            assert np.all(np.abs(r) <= 1.0)
            assert r[int(np.argmax(rules.final_scores(fin)))] >= 0.0


def test_search_terminals_use_the_configured_target() -> None:
    rng = random.Random(5)
    core = rules.initial_core(3)
    history = []
    while core.phase != rules.PHASE_END:
        history.append(core)
        core = rules.apply(core, rng.choice(rules.legal_actions(core)))
    root = next(c for c in reversed(history) if len(rules.legal_actions(c)) >= 2)
    s = GumbelSearch(root, 16, GumbelConfig(target="leader"))
    while s.want() is not None:
        s.give(np.zeros(gb.NUM_ACTIONS), np.zeros(5))
    term = [nd for nd in _all_nodes(s.root) if nd.terminal]
    assert term
    for nd in term:
        np.testing.assert_allclose(nd.value, terminal_values(nd.core, 0.5, "leader"))
