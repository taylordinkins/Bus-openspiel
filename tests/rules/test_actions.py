"""Action ID layout (docs/ACTIONS.md sections 2 and 4), expected in the module ``bus.actions``.

Expected names: SELECT_BASE, DROP_OUT, EXTEND_BASE, BUILD_BASE, PASSENGER_TO_BASE,
CLOCK_ADVANCE, CLOCK_PAUSE, PICK_PASSENGER_BASE, DELIVER_TO_BASE, NUM_DISTINCT_ACTIONS,
encode_select(a), encode_extend(e, s), encode_build(t, y), encode_passenger_to(k),
encode_pick_passenger(n), encode_deliver_to(t), decode_action(id) -> (kind, params tuple) with
kind in select/drop_out/extend/build/passenger_to/clock_advance/clock_pause/pick_passenger/
deliver_to, and action_to_string(seat, action_id) -> str (static; board from load_board()).
"""

from collections.abc import Callable
from typing import Any

import pytest

from bus.board import load_board

B = load_board()
N, S, T, K = len(B.nodes), len(B.streets), len(B.sites), len(B.stations)


def _a() -> Any:
    from bus import actions

    return actions


def _all_encoded() -> list[tuple[int, str, tuple[int, ...]]]:
    A = _a()
    out: list[tuple[int, str, tuple[int, ...]]] = []
    out += [(A.encode_select(a), "select", (a,)) for a in range(7)]
    out += [(A.DROP_OUT, "drop_out", ())]
    out += [(A.encode_extend(e, s), "extend", (e, s)) for e in range(2) for s in range(S)]
    out += [(A.encode_build(t, y), "build", (t, y)) for t in range(T) for y in range(3)]
    out += [(A.encode_passenger_to(k), "passenger_to", (k,)) for k in range(K)]
    out += [(A.CLOCK_ADVANCE, "clock_advance", ()), (A.CLOCK_PAUSE, "clock_pause", ())]
    out += [(A.encode_pick_passenger(n), "pick_passenger", (n,)) for n in range(N)]
    out += [(A.encode_deliver_to(t), "deliver_to", (t,)) for t in range(T)]
    return out


def test_ACTIONS2_total_is_376_for_current_board() -> None:
    """ACTIONS-2: num_distinct_actions = 376 (N=36, S=70, T=47, K=2)."""
    assert (N, S, T, K) == (36, 70, 47, 2)
    assert _a().NUM_DISTINCT_ACTIONS == 376


def test_ACTIONS2_bases_follow_board_sizes() -> None:
    """ACTIONS-2: named constants are derived from the board counts."""
    A = _a()
    assert A.SELECT_BASE == 0
    assert A.DROP_OUT == 7
    assert A.EXTEND_BASE == 8
    assert A.BUILD_BASE == A.EXTEND_BASE + 2 * S == 148
    assert A.PASSENGER_TO_BASE == A.BUILD_BASE + 3 * T == 289
    assert A.CLOCK_ADVANCE == A.PASSENGER_TO_BASE + K == 291
    assert A.CLOCK_PAUSE == 292
    assert A.PICK_PASSENGER_BASE == 293
    assert A.DELIVER_TO_BASE == A.PICK_PASSENGER_BASE + N == 329
    assert A.NUM_DISTINCT_ACTIONS == A.DELIVER_TO_BASE + T


def test_ACTIONS2_encoders_match_formulas() -> None:
    """ACTIONS-2: spot-check each formula, including first and last of each block."""
    A = _a()
    assert A.encode_select(0) == 0 and A.encode_select(6) == 6
    assert A.encode_extend(0, 0) == 8
    assert A.encode_extend(1, 0) == 8 + S
    assert A.encode_extend(1, S - 1) == 8 + 2 * S - 1
    assert A.encode_build(0, 0) == A.BUILD_BASE
    assert A.encode_build(2, 1) == A.BUILD_BASE + 7
    assert A.encode_build(T - 1, 2) == A.PASSENGER_TO_BASE - 1
    assert A.encode_passenger_to(K - 1) == A.CLOCK_ADVANCE - 1
    assert A.encode_pick_passenger(N - 1) == A.DELIVER_TO_BASE - 1
    assert A.encode_deliver_to(T - 1) == A.NUM_DISTINCT_ACTIONS - 1


def test_ACTIONS2_ids_contiguous_no_gaps_no_duplicates() -> None:
    """ACTIONS-2: ids 0..375 are each produced exactly once."""
    ids = sorted(i for i, _, _ in _all_encoded())
    assert ids == list(range(_a().NUM_DISTINCT_ACTIONS))


def test_ACTIONS2_roundtrip_every_id() -> None:
    """ACTIONS-2: decode_action inverts every encode_*."""
    A = _a()
    for i, kind, params in _all_encoded():
        assert A.decode_action(i) == (kind, params), i


@pytest.mark.parametrize("bad", [-1, 376, 1000])
def test_ACTIONS2_decode_out_of_range_raises(bad: int) -> None:
    """ACTIONS-2: ids outside 0..NUM_DISTINCT_ACTIONS-1 are rejected."""
    with pytest.raises(ValueError):
        _a().decode_action(bad)


def test_ACTIONS2_encoders_reject_out_of_range_params() -> None:
    """ACTIONS-2: encoders reject indexes outside their ranges rather than aliasing."""
    A = _a()
    calls: list[Callable[[], int]] = [
        lambda: A.encode_select(7),
        lambda: A.encode_extend(2, 0),
        lambda: A.encode_extend(0, S),
        lambda: A.encode_build(T, 0),
        lambda: A.encode_build(0, 3),
        lambda: A.encode_passenger_to(K),
        lambda: A.encode_pick_passenger(N),
        lambda: A.encode_deliver_to(T),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_ACTIONS2_string_simple_kinds() -> None:
    """ACTIONS-2 / section 4: fixed strings for select, drop out, clock."""
    A = _a()
    names = ["Lines", "Buses", "Passengers", "Buildings", "Clock", "Vrrooomm!", "Starting Player"]
    for a, name in enumerate(names):
        assert A.action_to_string(2, A.encode_select(a)) == f"P2: select {name}"
    assert A.action_to_string(2, A.DROP_OUT) == "P2: drop out"
    assert A.action_to_string(3, A.CLOCK_ADVANCE) == "P3: clock advance"
    assert A.action_to_string(3, A.CLOCK_PAUSE) == "P3: clock pause"


def test_ACTIONS2_string_board_referencing_kinds() -> None:
    """ACTIONS-2 / section 4: build, passenger, pick, deliver strings use board ids."""
    A = _a()
    t = T - 1
    node = next(n for n in range(N) if t in B.node_sites[n])
    ref = f"site {t} ({B.sites[t].tier}, n{node})"
    for y, w in enumerate(["house", "office", "pub"]):
        assert A.action_to_string(1, A.encode_build(t, y)) == f"P1: build {w} at {ref}"
    assert A.action_to_string(2, A.encode_deliver_to(t)) == f"P2: deliver to {ref}"
    st = B.stations[0]
    assert A.action_to_string(0, A.encode_passenger_to(0)) == f"P0: passenger to station n{st}"
    assert A.action_to_string(2, A.encode_pick_passenger(5)) == "P2: pick passenger at n5"


def test_ACTIONS2_string_extend_names_street() -> None:
    """ACTIONS-2 / section 4: extend strings name street and endpoints in a-b order."""
    A = _a()
    s = S - 1
    st = B.streets[s]
    text = A.action_to_string(1, A.encode_extend(1, s))
    assert text.startswith("P1: ")
    assert f"s{s} (n{st.a}-n{st.b})" in text
    assert "extend end 1 via" in text or "start line on" in text
