# pyright: basic
"""Observation v3 (Phase 4 interaction, direction 2): per-seat derived and interaction features.

``reference_v3`` recomputes every v3 block from the rule helpers (``bus.lines``, ``bus.rules``)
in plain Python; the fast encoder must match it on random games and scenario states. Hand cases
cover the resolution-order blocks on the R-030 example and the score gaps.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from bus import constants as C
from bus.lines import legal_extensions, line_nodes
from bus.rules import (
    EMPTY,
    PHASE_END,
    PHASE_SEL,
    Core,
    apply,
    board,
    initial_core,
    legal_actions,
    marker_power,
    resolution_positions,
    resolving_action,
    vrrooomm_moves,
)
from tests.harness import load_fixture
from train import features as F

V3 = F.FEATURES_V3
RESOLVING = ("LN", "PX", "BD", "CK", "VP", "VD")
ACT_LINES, ACT_BUSES, ACT_PASSENGERS, ACT_BUILDINGS, ACT_CLOCK, ACT_VROOM, ACT_START = range(7)


def _blocks(vec: np.ndarray, version: int = V3) -> dict[str, np.ndarray]:
    out, off = {}, 0
    for k, sh in F.obs_layout(version).items():
        size = int(np.prod(sh))
        out[k] = vec[off : off + size].reshape(sh)
        off += size
    assert off == len(vec)
    return out


def reference_v3(core: Core) -> dict[str, np.ndarray]:
    """The v3 extra blocks, straight from the module docstring and the rule helpers."""
    b = board()
    n, me = core.players, core.current
    M = F.MAX_SEATS
    seat_of = [(me + j) % n for j in range(n)]
    lines = [s.line for s in core.seats]
    nodes_on = [set(line_nodes(b, lines[s])) if lines[s] else set() for s in seat_of]
    streets_on = [set(lines[s]) for s in seat_of]
    ext = [{x for _e, x in legal_extensions(b, lines, s)} if lines[s] else set() for s in seat_of]
    reach = [
        {nd for x in ext[j] for nd in (b.streets[x].a, b.streets[x].b)} - nodes_on[j]
        for j in range(n)
    ]
    empty = [sum(1 for t in b.node_sites[v] if core.site_types[t] == EMPTY) for v in range(F.N)]
    unocc = np.zeros((F.N, 3))
    for t, ty in enumerate(core.site_types):
        if ty != EMPTY and t not in core.occupied:
            unocc[b.sites[t].node, ty] += 1
    w = core.waiting
    per_node = max(len(x) for x in b.node_sites)  # most sites at one node (board data)
    r: dict[str, np.ndarray] = {k: np.zeros(sh) for k, sh in F.obs_layout(V3).items()}
    for j in range(n):
        s = seat_of[j]
        for v in nodes_on[j]:
            r["seat_pickable"][v, j] = w[v] / 15
            r["seat_free_sites"][v, j] = empty[v] / per_node
        wo = sum(w[v] for v in nodes_on[j])
        eo = sum(empty[v] for v in nodes_on[j])
        buses = core.seats[s].buses
        r["seat_vroom"][j] = [wo / 15, eo / 10, min(buses, wo, eo) / 5]
        r["seat_vroom_now"][j] = vrrooomm_moves(core, s) / 5 if lines[s] else 0.0
        for ty in range(3):
            r["seat_targets"][j, ty] = sum(unocc[v, ty] for v in nodes_on[j]) / 10
        for v in reach[j]:
            r["seat_reach"][v, j] = 1.0
    opp_nodes = set().union(*nodes_on[1:]) if n > 1 else set()
    for v in range(F.N):
        if v in nodes_on[0] and v in opp_nodes:
            r["contested_node"][v, 0] = w[v] / 15
            r["contested_node"][v, 1] = unocc[v].sum() / per_node
        if v in nodes_on[0] or v in reach[0]:
            r["contested_node"][v, 2] = sum(v in reach[j] for j in range(1, n)) / (M - 1)
    for j in range(1, n):
        sh = nodes_on[0] & nodes_on[j]
        r["pair_overlap"][j] = [
            len(sh) / 10, sum(w[v] for v in sh) / 15, sum(unocc[v].sum() for v in sh) / 10,
            len(streets_on[0] & streets_on[j]) / 10,
        ]  # fmt: skip
    for x in ext[0]:
        r["frontier_contested"][x] = sum(x in ext[j] for j in range(1, n)) / (M - 1)
    # resolution order
    ab = core.action_board
    pos = resolution_positions(core)
    if core.phase in RESOLVING:
        act = resolving_action(core)
        assert act is not None
        start = pos.index((act, core.cursor))
    elif core.phase == PHASE_SEL:
        start = 0
    else:
        start = len(pos)
    for a in range(7):
        mine = [i for i, (aa, k) in enumerate(pos) if aa == a and ab[a][k] == me]
        for i, (aa, k) in enumerate(pos):
            if aa != a:
                continue
            j = (ab[a][k] - me) % n
            if i >= start:
                r["res_pending"][a, j] += 1 / 6
            if mine and i < mine[0]:
                r["res_before"][a, j] += 1 / 6
            if mine and i > mine[0]:
                r["res_after"][a, j] += 1 / 6
    if core.phase == PHASE_SEL or core.phase in RESOLVING:
        for owner in ab[ACT_START]:
            r["res_pending"][ACT_START, (owner - me) % n] += 1 / 6
    for a in range(7):
        row = ab[a]
        if len(row) >= C.ACTION_SPACES[a]:
            continue
        if a in (ACT_LINES, ACT_PASSENGERS, ACT_BUILDINGS):
            r["place_now"][a, 0] = max(marker_power(core, a, len(row)), 0) / 6
        else:
            r["place_now"][a, 0] = 1.0
        # where would a new marker resolve? place it hypothetically and ask the engine
        new_ab = ab[:a] + (row + (me,),) + ab[a + 1 :]
        hyp = replace(core, action_board=new_ab)
        order = [k for aa, k in resolution_positions(hyp) if aa == a]
        r["place_now"][a, 1] = 1.0 if (a == ACT_START or order[0] == len(row)) else 0.0
    for j in range(n):
        s = seat_of[j]
        active = core.phase == PHASE_SEL and s not in core.dropped
        r["sel_state"][j, 0] = float(active and core.seats[s].action_markers > 0)
        r["sel_state"][j, 1] = min(sum(row.count(s) for row in ab) / 6, 1.0)
    sc = [x.points - x.stones for x in core.seats]
    others = [sc[p] for p in range(n) if p != me]
    r["score_gap"][0] = np.clip((sc[me] - max(others)) / 10, -1, 1)
    r["score_gap"][1] = np.clip((sc[me] - np.mean(others)) / 10, -1, 1)
    beaten = sum(1.0 if sc[me] > x else 0.5 if sc[me] == x else 0.0 for x in others)
    r["score_gap"][2] = 2 * beaten / (n - 1) - 1
    return {k: r[k] for k in F.obs_layout(V3) if k not in F.obs_layout(F.FEATURES_V2)}


def _random_cores(n: int, games: int, seed: int) -> list[Core]:
    out = []
    for g in range(games):
        rng = random.Random(seed * 100 + g)
        core = initial_core(n)
        while core.phase != PHASE_END:
            legal = legal_actions(core)
            if len(legal) > 1:
                out.append(core)
            core = apply(core, rng.choice(legal))
    return out


def _assert_matches_reference(core: Core) -> None:
    got = _blocks(F.encode_v3(core))
    ref = reference_v3(core)
    for k, v in ref.items():
        np.testing.assert_allclose(got[k], v, atol=1e-6, err_msg=f"{k} in {core.phase}")


# ------------------------------------------------------------------ layout and prefix


def test_layout_sizes_and_versions() -> None:
    assert F.FEATURE_VERSIONS == (F.FEATURES_V2, F.FEATURES_V3) == (1, 2)
    assert F.FEATURES_VERSION == F.FEATURES_V2  # the default stays v2
    assert F.obs_size(F.FEATURES_V2) == F.OBS_SIZE == 2314
    lay = F.obs_layout(V3)
    assert list(lay)[: len(F.obs_layout())] == list(F.obs_layout())
    assert F.OBS_SIZE_V3 == F.obs_size(V3) == sum(int(np.prod(s)) for s in lay.values())
    assert F.version_for_obs_size(F.OBS_SIZE_V3) == V3
    assert F.version_for_obs_size(F.OBS_SIZE) == F.FEATURES_V2
    assert F.version_for_obs_size(7) is None
    assert F.encoder(F.FEATURES_V2) is F.encode and F.encoder(V3) is F.encode_v3
    with pytest.raises(ValueError):
        F.encoder(3)


def test_layout_hashes_pin_v2_and_differ_for_v3() -> None:
    from bots.net_forward import layout_hash, version_for_layout

    assert layout_hash() == layout_hash(1) == "4bb7af3e33b12f0e"  # what ppo-long stores
    assert layout_hash(2) != layout_hash(1)
    assert version_for_layout(layout_hash(2)) == 2 and version_for_layout("deadbeef") is None


@pytest.mark.parametrize("n", [3, 4, 5])
def test_v3_prefix_is_v2_and_values_bounded(n: int) -> None:
    for core in _random_cores(n, 3, n)[::3]:
        v3 = F.encode_v3(core)
        assert v3.dtype == np.float32 and v3.shape == (F.OBS_SIZE_V3,)
        np.testing.assert_array_equal(v3[: F.OBS_SIZE], F.encode(core))
        assert np.isfinite(v3).all() and np.abs(v3).max() <= 1.5


def test_input_columns_map_v2_blocks_onto_the_v3_prefix() -> None:
    cols = F.input_columns(F.FEATURES_V2, V3)
    np.testing.assert_array_equal(cols, np.arange(F.OBS_SIZE))
    np.testing.assert_array_equal(F.input_columns(V3, V3), np.arange(F.OBS_SIZE_V3))
    with pytest.raises(ValueError):
        F.input_columns(V3, F.FEATURES_V2)


# ------------------------------------------------------------------ against the reference


@pytest.mark.parametrize("n", [3, 4, 5])
def test_fast_v3_equals_reference_on_random_games(n: int) -> None:
    cores = _random_cores(n, 4, 10 + n)
    phases = set()
    for core in cores[::2]:
        _assert_matches_reference(core)
        phases.add(core.phase)
    assert {"SEL", "LN", "VP"} <= phases


@pytest.mark.parametrize(
    "name",
    ["sel_round1", "sel_pass2_dropped", "resolve_ln", "resolve_px", "resolve_bd", "resolve_vp",
     "resolve_ck", "lines/shared_end", "lines/five_player_shared", "lines/three_player_per_end",
     "lines/full_loop_shared"],
)  # fmt: skip
def test_fast_v3_equals_reference_on_scenarios(name: str) -> None:
    core = load_fixture(name).core
    if core.phase == PHASE_END or len(legal_actions(core)) == 0:
        pytest.skip("no decision")
    _assert_matches_reference(core)
    for k in range(1, core.players):  # the same position from every seat's view
        _assert_matches_reference(replace(core, current=(core.current + k) % core.players))


def test_vroom_now_is_the_rule_count_for_every_seat() -> None:
    for core in _random_cores(4, 3, 99):
        vn = _blocks(F.encode_v3(core))["seat_vroom_now"]
        for j in range(4):
            seat = (core.current + j) % 4
            want = vrrooomm_moves(core, seat) / 5 if core.seats[seat].line else 0.0
            assert vn[j] == pytest.approx(want)


def test_ego_columns_repeat_v2_ego_features() -> None:
    for core in _random_cores(4, 2, 5)[::5]:
        v = _blocks(F.encode_v3(core))
        np.testing.assert_allclose(v["seat_pickable"][:, 0], v["ego_pickable"], atol=1e-7)
        np.testing.assert_allclose(v["seat_free_sites"][:, 0], v["ego_free_sites"], atol=1e-7)
        np.testing.assert_allclose(v["seat_vroom"][0], v["vroom"], atol=1e-7)


def _relabel(core: Core, k: int) -> Core:
    n = core.players
    seats = list(core.seats)
    for i, s in enumerate(core.seats):
        seats[(i + k) % n] = s
    return replace(
        core, seats=tuple(seats),
        action_board=tuple(tuple((x + k) % n for x in row) for row in core.action_board),
        current=(core.current + k) % n, start_player=(core.start_player + k) % n,
        dropped=tuple(sorted((p + k) % n for p in core.dropped)),
    )  # fmt: skip


@pytest.mark.parametrize("n", [3, 5])
def test_v3_ego_rotation_only_reorders(n: int) -> None:
    for core in _random_cores(n, 2, 7)[::7]:
        base = F.encode_v3(core)
        for k in range(1, n):
            np.testing.assert_array_equal(base, F.encode_v3(_relabel(core, k)))


# ------------------------------------------------------------------ hand cases


def _sel_core(board_rows: dict[int, tuple[int, ...]], me: int) -> Core:
    core = load_fixture("sel_round1").core
    ab = list(core.action_board)
    for a, row in board_rows.items():
        ab[a] = row
    return replace(core, action_board=tuple(ab), current=me)


def test_resolution_order_on_the_r030_example() -> None:
    # R-030: Lines a = P1, b = P2, c = P0 resolves c (P0), b (P2), a (P1); Passengers a, b go
    # in letter order: a = P2 first, then b = P0.
    rows = {ACT_LINES: (1, 2, 0), ACT_PASSENGERS: (2, 0)}
    v = _blocks(F.encode_v3(_sel_core(rows, 0)))
    six = 1 / 6
    np.testing.assert_allclose(v["res_before"][ACT_LINES], [0, 0, 0, 0, 0])
    np.testing.assert_allclose(v["res_after"][ACT_LINES], [0, six, six, 0, 0])  # P1, P2 after
    np.testing.assert_allclose(v["res_before"][ACT_PASSENGERS], [0, 0, six, 0, 0])  # P2 first
    np.testing.assert_allclose(v["res_pending"][ACT_LINES], [six, six, six, 0, 0])
    assert v["res_before"][ACT_BUSES].sum() == 0 and v["res_pending"][ACT_BUSES].sum() == 0
    # P1's only Lines marker (a) resolves last: P0 (ego + 3) and P2 (ego + 1) go before it
    v1 = _blocks(F.encode_v3(_sel_core(rows, 1)))
    np.testing.assert_allclose(v1["res_before"][ACT_LINES], [0, six, 0, six, 0])
    np.testing.assert_allclose(v1["res_after"][ACT_LINES], [0, 0, 0, 0, 0])


def test_place_now_power_and_first_flags() -> None:
    # all seats have 1 bus at round 1, so M = 1 (R-032): letter a has power 1, b and later 0
    rows = {ACT_LINES: (1, 2, 0), ACT_PASSENGERS: (2,), ACT_CLOCK: (3,)}
    v = _blocks(F.encode_v3(_sel_core(rows, 0)))["place_now"]
    np.testing.assert_allclose(v[ACT_LINES], [0.0, 1.0])  # power 1 - 3 <= 0; resolves first
    np.testing.assert_allclose(v[ACT_PASSENGERS], [0.0, 0.0])  # letter b: power 0; after a
    np.testing.assert_allclose(v[ACT_BUILDINGS], [1 / 6, 1.0])  # empty: letter a, power 1
    np.testing.assert_allclose(v[ACT_VROOM], [1.0, 1.0])
    np.testing.assert_allclose(v[ACT_CLOCK], [0.0, 0.0])  # full (one space)
    np.testing.assert_allclose(v[ACT_BUSES], [1.0, 1.0])


def test_score_gaps_hand_case() -> None:
    core = load_fixture("sel_round1").core
    pts, stones = [6, 9, 2, 6], [0, 1, 0, 0]  # scores 6, 8, 2, 6
    seats = tuple(
        replace(s, points=p, stones=t) for s, p, t in zip(core.seats, pts, stones, strict=True)
    )
    v = _blocks(F.encode_v3(replace(core, seats=seats, current=0)))["score_gap"]
    # vs best 8: -0.2; vs mean (8 + 2 + 6) / 3: (6 - 16/3) / 10; beats P2, ties P3: 1.5 of 3
    np.testing.assert_allclose(v, [-0.2, (6 - 16 / 3) / 10, 2 * 1.5 / 3 - 1], atol=1e-6)


def test_contested_and_reach_on_shared_end_scenario() -> None:
    core = load_fixture("lines/shared_end").core
    v = _blocks(F.encode_v3(core))
    ref = reference_v3(core)
    np.testing.assert_allclose(v["seat_reach"], ref["seat_reach"])
    assert v["seat_reach"][:, 0].sum() > 0  # the mover can extend
    np.testing.assert_allclose(v["frontier_contested"], ref["frontier_contested"])


def test_encode_v3_returns_fresh_array() -> None:
    cores = _random_cores(4, 1, 3)
    a = F.encode_v3(cores[10])
    snap = a.copy()
    F.encode_v3(cores[40])
    np.testing.assert_array_equal(a, snap)
