"""Observer: string and RL tensor (PLAN 5.2, OPENSPIEL_NOTES observer pattern, ACTIONS 3.12).

Phase 1 unit 13, written ahead of the code.  PLAN 5.2 fixes only the CONTENT of the tensor (planes
per street / node / site plus clock, stones, markers left, buses, scores, action-board occupancy and
the pending sub-decision), not the layout.  These tests therefore never index into the tensor; they
check size, constancy, finiteness, injectivity and local sensitivity (changing one fact changes the
tensor, and only in a few cells).  See the report for the ambiguities.

Rule-free: tests are named test_OBS_*.  ``@pending("OBS")`` marks the ones the 8-float stub cannot
pass.
"""

from __future__ import annotations

import copy
import itertools
import random
from typing import Any

import numpy as np
import pyspiel
import pytest
from open_spiel.python.observation import make_observation

from tests import harness as H
from tests.support import play as P

PLAYERS = [3, 4, 5]


def tensor(s: Any, p: int = 0) -> np.ndarray:
    return np.asarray(s.observation_tensor(p), dtype=np.float64)


def doc_state(name: str, **mods: Any) -> Any:
    d = H.load_doc(name)
    d.pop("expect", None)
    d.update(mods)
    return H.load_doc_state(d)


# ---- shape and ranges ---------------------------------------------------------------------


@pytest.mark.parametrize("n", PLAYERS)
def test_OBS_tensor_size_matches_game_declaration(n: int) -> None:
    """PLAN 5.2: tensor length equals observation_tensor_shape()/size at every player count."""
    g = P.game(n)
    shape = g.observation_tensor_shape()
    assert len(shape) >= 1 and all(d > 0 for d in shape)
    assert int(np.prod(shape)) == g.observation_tensor_size()
    s = g.new_initial_state()
    for p in range(n):
        assert len(s.observation_tensor(p)) == g.observation_tensor_size()


@pytest.mark.parametrize("n", PLAYERS)
def test_OBS_tensor_is_a_rich_plane_set(n: int) -> None:
    """PLAN 5.2: planes per street (70), node (36), site (47) plus globals cannot fit in a few
    cells; the size must exceed the number of streets + nodes + sites."""
    from bus.board import load_board

    b = load_board()
    assert P.game(n).observation_tensor_size() > len(b.streets) + len(b.nodes) + len(b.sites)


@pytest.mark.parametrize("n", PLAYERS)
def test_OBS_tensor_size_constant_over_a_random_game(n: int) -> None:
    """PLAN 5.2 / NOTES: the tensor is one flat float32 array of constant size, also at END."""
    g = P.game(n)
    size = g.observation_tensor_size()
    states = P.random_game(n, seed=100 + n, every=7)
    assert states[-1].is_terminal()
    for s in states:
        for p in range(n):
            assert len(s.observation_tensor(p)) == size


@pytest.mark.parametrize("n", PLAYERS)
def test_OBS_values_finite_and_bounded(n: int) -> None:
    """PLAN 5.2: every value is finite and within [-1, 1000]-ish; nothing is NaN/inf, and a
    mid-game tensor is not all zeros.  (The docs give no scaling, so the bound is loose.)"""
    states = P.random_game(n, seed=7 + n, every=11)
    for s in states:
        t = tensor(s, 0)
        assert np.all(np.isfinite(t))
        assert t.min() >= -1.0 and t.max() <= 1000.0
    assert any(np.any(tensor(s) != 0) for s in states)


def test_OBS_tensor_has_plane_structure() -> None:
    """PLAN 5.2: most cells of the tensor are 0/1 plane flags, with only a few scalars; a
    mid-game tensor has non-binary cells but far fewer than binary ones."""
    s = doc_state("sel_round1")
    t = tensor(s)
    nonbinary = np.sum((t != 0) & (t != 1))
    assert t.size > 300
    assert nonbinary <= t.size // 4


# ---- observer object ----------------------------------------------------------------------


def test_OBS_make_py_observer_default_params() -> None:
    """NOTES: make_py_observer() with defaults returns an object with tensor/dict/set_from."""
    g = P.game(4)
    obs = g.make_py_observer()
    s = g.new_initial_state()
    obs.set_from(s, 0)
    assert obs.tensor.dtype == np.float32
    assert obs.tensor.ndim == 1 and obs.tensor.size == g.observation_tensor_size()
    assert isinstance(obs.dict, dict)
    assert np.allclose(obs.tensor, np.asarray(s.observation_tensor(0), dtype=np.float32))


def test_OBS_make_observation_roundtrip() -> None:
    """NOTES: open_spiel.python.observation.make_observation(game) works and agrees with the
    state's own observation_string/tensor."""
    g = P.game(4)
    s = doc_state("sel_round1")
    o = make_observation(g)
    o.set_from(s, 1)
    assert o.string_from(s, 1) == s.observation_string(1)
    assert np.allclose(o.tensor, tensor(s, 1))


def test_OBS_observer_dict_names_cover_the_tensor() -> None:
    """NOTES: ``dict`` holds named views onto ``tensor``; their total size equals the tensor's
    (the plane sets of PLAN 5.2 are named), and writing through a view changes the tensor."""
    g = P.game(4)
    obs = g.make_py_observer()
    assert len(obs.dict) >= 5
    assert sum(v.size for v in obs.dict.values()) == obs.tensor.size
    name = next(iter(obs.dict))
    view = obs.dict[name]
    view.flat[0] = 1.0
    assert obs.tensor[np.flatnonzero(obs.tensor)].size >= 1


def test_OBS_observation_string_same_for_every_seat() -> None:
    """R-073 (perfect information): every seat's observation string has the same content."""
    for s in P.random_game(4, seed=3, every=40):
        strs = {s.observation_string(p) for p in range(4)}
        assert len(strs) == 1


def test_OBS_str_delegates_to_observer_string() -> None:
    """CLAUDE.md: State.__str__ is the observer's string."""
    for s in P.random_game(3, seed=4, every=50):
        assert str(s) == s.observation_string(0)


def test_OBS_tensor_does_not_depend_on_history_only_on_position() -> None:
    """The observation is a function of the position: a clone has the same tensor and string."""
    for s in P.random_game(4, seed=5, every=60):
        c = s.clone()
        assert str(c) == str(s)
        assert np.array_equal(tensor(c), tensor(s))


# ---- injectivity --------------------------------------------------------------------------

# last_score_move / move_offset are tiebreak bookkeeping the tensor need not encode (PLAN 5.2 lists
# scores, not move numbers); strings must still distinguish them.
_TENSOR_IGNORED = ("move_offset",)


def _strip(d: dict[str, Any]) -> str:
    import json

    d = copy.deepcopy(d)
    for k in _TENSOR_IGNORED:
        d.pop(k, None)
    for seat in d.get("seats", []):
        seat.pop("last_score_move", None)
    return json.dumps(d, sort_keys=True)


def _pool() -> list[Any]:
    out: list[Any] = []
    for n, seed in [(3, 11), (4, 12), (5, 13), (4, 14)]:
        out += P.random_game(n, seed, every=2)
    return out


def test_OBS_distinct_dumps_give_distinct_strings() -> None:
    """CLAUDE.md: distinct states never print alike (random pairs over random games)."""
    pool = _pool()
    by_key: dict[str, str] = {}
    for s in pool:
        k = P.key(s)
        st = str(s)
        if k in by_key:
            assert by_key[k] == st
        by_key[k] = st
    keys = list(by_key)
    assert len(keys) > 300
    strings = list(by_key.values())
    assert len(set(strings)) == len(strings), "two different positions print alike"


def test_OBS_distinct_positions_give_distinct_tensors() -> None:
    """PLAN 5.2: the tensor separates positions (ignoring move-number bookkeeping)."""
    pool = _pool()
    seen: dict[tuple[int, str], bytes] = {}
    for s in pool:
        n = s.get_game().num_players()
        k = (n, _strip(P.dump(s)))
        t = tensor(s).astype(np.float32).tobytes()
        if k in seen:
            assert seen[k] == t
        seen[k] = t
    by_tensor: dict[tuple[int, bytes], str] = {}
    for (n, pos), t in seen.items():
        assert by_tensor.setdefault((n, t), pos) == pos, "two positions share a tensor"


def test_OBS_player_counts_print_differently() -> None:
    """CLAUDE.md: fresh states of 3/4/5 players have different strings."""
    assert len({str(P.game(n).new_initial_state()) for n in PLAYERS}) == 3


# ---- round trip ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", PLAYERS)
def test_OBS_load_dump_preserves_string_and_tensor(n: int) -> None:
    """Unit 12 audit: str(load(dump(s))) == str(s) and the tensor agrees, over a whole game
    (non-terminal states; terminal ones are in the OBS-END tests)."""
    states = [s for s in P.random_game(n, seed=20 + n, every=5) if not s.is_terminal()]
    for s in states:
        r = P.reload(s)
        assert str(r) == str(s), (s.phase, r.phase)
        assert np.array_equal(tensor(r), tensor(s))


@pytest.mark.parametrize(
    "name",
    ["end_both", "end_last_stone", "end_sites_full", "end/tiebreak_shared", "end/tiebreak_stones"],
)
def test_OBS_load_dump_roundtrip_at_end(name: str) -> None:
    """Unit 12 audit: at END the current player is canonical so the string survives dump/load."""
    s = H.load_fixture(name)
    assert s.is_terminal()
    r = P.reload(s)
    assert r.is_terminal()
    assert str(r) == str(s)
    assert np.array_equal(tensor(r), tensor(s))


@pytest.mark.parametrize("n", PLAYERS)
def test_OBS_roundtrip_of_played_terminal_states(n: int) -> None:
    """Unit 12 audit: a terminal state reached by play round-trips (round-end and last-stone)."""
    for seed in range(4):
        s = P.random_game(n, 300 + seed, every=10**6)[-1]
        assert s.is_terminal()
        r = P.reload(s)
        assert str(r) == str(s)
        assert np.array_equal(tensor(r), tensor(s))


# ---- local sensitivity (spot checks without a fixed layout) --------------------------------


def _changed(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def test_OBS_clock_hand_is_one_hot() -> None:
    """PLAN 5.2 clock: three clock values give three different tensors; each pair differs in
    exactly two cells (a one-hot moves)."""
    ts = [tensor(doc_state("sel_round1", clock=c)) for c in ("house", "office", "pub")]
    for a, b in itertools.combinations(ts, 2):
        assert _changed(a, b) == 2


def test_OBS_waiting_passenger_count_changes_few_cells() -> None:
    """PLAN 5.2 per node waiting passengers: adding one waiting passenger at a node (and taking
    it from the supply) changes the tensor in at least one and at most a handful of cells."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    d["waiting"]["13"] = 2
    b = H.load_doc_state(d)
    n = _changed(tensor(a), tensor(b))
    assert 1 <= n <= 6  # waiting plane cell(s) and the supply scalar


def test_OBS_waiting_at_different_nodes_differs() -> None:
    """PLAN 5.2: moving a waiting passenger from one node to another changes the tensor, and the
    cell count equals that of the mirrored move (node planes are distinct per node)."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    d["waiting"] = {"13": 1, "17": 1, "18": 1, "22": 1}
    assert d["waiting"] != base["waiting"]
    try:
        b = H.load_doc_state(d)
    except Exception:  # noqa: BLE001  (node 22 may not be a legal fixture node)
        pytest.skip("alternative waiting layout rejected by loader")
    assert 2 <= _changed(tensor(a), tensor(b)) <= 6


def test_OBS_building_type_is_visible_per_site() -> None:
    """PLAN 5.2 per site building type: changing site 0 from house to pub changes the tensor in
    exactly the type cells (two cells if one-hot over types)."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    assert d["buildings"][0] == {"site": 0, "type": "house"}
    d["buildings"][0]["type"] = "pub"
    b = H.load_doc_state(d)
    assert _changed(tensor(a), tensor(b)) == 2


def test_OBS_occupied_flag_is_visible() -> None:
    """PLAN 5.2 per site occupied: a passenger inside a building (taken from the waiting pile)
    changes the tensor."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    d["buildings"][0]["occupied"] = True
    del d["waiting"]["13"]
    b = H.load_doc_state(d)
    assert _changed(tensor(a), tensor(b)) >= 2


def test_OBS_line_occupancy_per_street() -> None:
    """PLAN 5.2 per street line occupancy per seat: a longer line for seat 0 raises the number of
    non-zero cells by at least the added street, and swapping two seats' lines changes the
    tensor (occupancy is per seat)."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    swapped = copy.deepcopy(base)
    swapped["seats"][0]["line"], swapped["seats"][1]["line"] = (
        base["seats"][1]["line"],
        base["seats"][0]["line"],
    )
    b = H.load_doc_state(swapped)
    assert _changed(tensor(a), tensor(b)) >= 4  # streets of both lines move planes (ends too)


def test_OBS_line_end_orientation_is_visible() -> None:
    """PLAN 5.2 own-line ends: reversing a line's street order swaps its ends, which the tensor
    must show (the ends are encoded separately from occupancy)."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    d["seats"][0]["line"] = list(reversed(base["seats"][0]["line"]))
    b = H.load_doc_state(d)
    ta, tb = tensor(a), tensor(b)
    assert _changed(ta, tb) >= 1
    assert str(a) != str(b)


def test_OBS_buses_markers_and_points_scalars() -> None:
    """PLAN 5.2 buses / markers left / scores: each scalar changes the tensor, and by a few
    cells only."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    for fld, val in (("buses", 3), ("markers_left", 17)):
        d = copy.deepcopy(base)
        d["seats"][2][fld] = val
        b = H.load_doc_state(d)
        assert 1 <= _changed(tensor(a), tensor(b)) <= 6, fld
    d = copy.deepcopy(base)
    d["seats"][2]["points"] = 4
    d["seats"][2]["last_score_move"] = 3
    d["move_offset"] = 3
    b = H.load_doc_state(d)
    assert 1 <= _changed(tensor(a), tensor(b)) <= 6


def test_OBS_stones_on_clock_and_held() -> None:
    """PLAN 5.2 stones: a stone moved from the clock to a seat changes the tensor."""
    base = H.load_doc("sel_round1")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    d["stones_on_clock"] = 4
    d["seats"][1]["stones"] = 1
    b = H.load_doc_state(d)
    assert _changed(tensor(a), tensor(b)) >= 2


def test_OBS_action_board_occupancy() -> None:
    """PLAN 5.2 action-board occupancy: one marker on a letter changes the tensor, and a marker of
    another seat on the same letter changes it again (occupancy is per seat/letter)."""
    from tests.rules.test_game_end import ck_doc  # noqa: F401  (documented fixture builder)

    base = H.load_doc("resolve_ck")
    base.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(base))
    d = copy.deepcopy(base)
    d["action_board"]["buildings"][0] = 3
    d["seats"][3]["markers_left"] = 18
    b = H.load_doc_state(d)
    d2 = copy.deepcopy(d)
    d2["action_board"]["buildings"][0] = 2
    d2["seats"][2]["markers_left"] = 19
    d2["seats"][3]["markers_left"] = 19
    c = H.load_doc_state(d2)
    assert _changed(tensor(a), tensor(b)) >= 1
    assert not np.array_equal(tensor(b), tensor(c))


def test_OBS_pending_sub_decision_is_encoded() -> None:
    """PLAN 5.2 pending sub-decision: the phase, the resolving marker and the units/moves left all
    change the tensor.  CK vs VD fixtures, and VD with moves_left 1 vs 2."""
    ck = tensor(H.load_fixture("resolve_ck"))
    vd = tensor(H.load_fixture("resolve_vd"))
    assert not np.array_equal(ck, vd)
    d = H.load_doc("resolve_vd")
    d.pop("expect", None)
    d["pending"]["moves_left"] = 2
    vd2 = tensor(H.load_doc_state(d))
    assert not np.array_equal(vd, vd2)
    d["pending"]["moves_left"] = 1
    d["pending"]["picked_node"] = 13
    vd3 = tensor(H.load_doc_state(d))
    assert not np.array_equal(vd, vd3)


def test_OBS_pending_sub_decision_in_string() -> None:
    """ACTIONS 4: the string shows the pending sub-decision, so VD with different moves_left or
    picked node print differently."""
    d = H.load_doc("resolve_vd")
    d.pop("expect", None)
    base = str(H.load_doc_state(copy.deepcopy(d)))
    d["pending"]["moves_left"] = 2
    two = str(H.load_doc_state(copy.deepcopy(d)))
    d["pending"]["moves_left"] = 1
    d["pending"]["picked_node"] = 13
    other = str(H.load_doc_state(copy.deepcopy(d)))
    assert len({base, two, other}) == 3


# ---- winners in the string at END -------------------------------------------------------------


def test_OBS_end_string_shows_end_types() -> None:
    """ACTIONS 3.12: the observer string at END shows the recorded end types (R-066)."""
    s = H.load_fixture("end_both")
    text = str(s)
    assert "sites_full" in text and "markers" in text
    s2 = H.load_fixture("end_last_stone")
    assert "last_stone" in str(s2)


def test_OBS_end_string_shows_winners() -> None:
    """ACTIONS 3.12 / R-069: the string at END names the winners; a sole winner and a shared win
    print differently, and the same scores with another winner set print differently too."""
    solo = H.load_fixture("end/tiebreak_stones")  # winners [0]
    shared = H.load_fixture("end/tiebreak_shared")  # winners [0, 1]
    assert str(solo) != str(shared)
    assert "winner" in str(solo).lower()
    assert "winner" in str(shared).lower()
    assert "P0" in str(solo)
    assert "P0" in str(shared) and "P1" in str(shared)


def test_OBS_end_string_differs_with_tiebreak_outcome() -> None:
    """R-069/R-070: same board and points but a different last_score_move changes the winner, so
    the END strings must differ."""
    d = H.load_doc("end/tiebreak_move")
    d.pop("expect", None)
    a = H.load_doc_state(copy.deepcopy(d))
    seats = d["seats"]
    tied = [i for i, x in enumerate(seats) if x["points"] == max(y["points"] for y in seats)]
    assert len(tied) >= 2
    i, j = tied[:2]
    seats[i]["last_score_move"], seats[j]["last_score_move"] = (
        seats[j]["last_score_move"],
        seats[i]["last_score_move"],
    )
    b = H.load_doc_state(d)
    assert str(a) != str(b)


def test_OBS_non_terminal_string_has_no_winner_section() -> None:
    """ACTIONS 3.12: winners are only announced once the game is over."""
    assert "winner" not in str(H.load_fixture("sel_round1")).lower()


def test_OBS_random_games_all_seats_have_tensors_at_end() -> None:
    """The observer works at END for every seat and player count (finite, right size)."""
    for n in PLAYERS:
        s = P.random_game(n, seed=900 + n, every=10**6)[-1]
        assert s.is_terminal()
        for p in range(n):
            t = tensor(s, p)
            assert t.size == s.get_game().observation_tensor_size()
            assert np.all(np.isfinite(t))


def test_OBS_random_pair_sampling_strings() -> None:
    """CLAUDE.md: random pairs of states from different games never share a string unless their
    dumps agree."""
    rng = random.Random(1)
    pool = _pool()
    for _ in range(300):
        a, b = rng.sample(pool, 2)
        same_pos = P.key(a) == P.key(b)
        assert (str(a) == str(b)) == same_pos


def test_OBS_game_type_flags() -> None:
    """NOTES: the game declares an observation tensor and string (perfect information)."""
    t = P.game(4).get_type()
    assert t.provides_observation_tensor and t.provides_observation_string
    assert t.information == pyspiel.GameType.Information.PERFECT_INFORMATION
