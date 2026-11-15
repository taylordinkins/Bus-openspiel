"""Action id layout (docs/ACTIONS.md sections 1-2 and 4). Sizes come from the board."""

from __future__ import annotations

from bus.board import load_board
from bus.constants import BUILDING_TYPES

_B = load_board()
N = len(_B.nodes)
S = len(_B.streets)
T = len(_B.sites)
K = len(_B.stations)

NUM_ACTION_KINDS = 7  # resolution order of R-022 / R-029
ACTION_NAMES = (
    "Lines",
    "Buses",
    "Passengers",
    "Buildings",
    "Clock",
    "Vrrooomm!",
    "Starting Player",
)
NUM_ENDS = 2
NUM_TYPES = len(BUILDING_TYPES)

SELECT_BASE = 0
DROP_OUT = SELECT_BASE + NUM_ACTION_KINDS
EXTEND_BASE = DROP_OUT + 1
BUILD_BASE = EXTEND_BASE + NUM_ENDS * S
PASSENGER_TO_BASE = BUILD_BASE + NUM_TYPES * T
CLOCK_ADVANCE = PASSENGER_TO_BASE + K
CLOCK_PAUSE = CLOCK_ADVANCE + 1
PICK_PASSENGER_BASE = CLOCK_PAUSE + 1
DELIVER_TO_BASE = PICK_PASSENGER_BASE + N
NUM_DISTINCT_ACTIONS = DELIVER_TO_BASE + T


def _check(value: int, size: int, what: str) -> int:
    if not 0 <= value < size:
        raise ValueError(f"{what} {value} out of range 0..{size - 1}")
    return value


def encode_select(a: int) -> int:
    return SELECT_BASE + _check(a, NUM_ACTION_KINDS, "action")


def encode_extend(e: int, s: int) -> int:
    _check(e, NUM_ENDS, "end")
    return EXTEND_BASE + e * S + _check(s, S, "street")


def encode_build(t: int, y: int) -> int:
    _check(y, NUM_TYPES, "building type")
    return BUILD_BASE + _check(t, T, "site") * NUM_TYPES + y


def encode_passenger_to(k: int) -> int:
    return PASSENGER_TO_BASE + _check(k, K, "station")


def encode_pick_passenger(n: int) -> int:
    return PICK_PASSENGER_BASE + _check(n, N, "node")


def encode_deliver_to(t: int) -> int:
    return DELIVER_TO_BASE + _check(t, T, "site")


def decode_action(action_id: int) -> tuple[str, tuple[int, ...]]:
    """Inverse of the encoders: ``(kind, params)``."""
    _check(action_id, NUM_DISTINCT_ACTIONS, "action id")
    if action_id < DROP_OUT:
        return "select", (action_id - SELECT_BASE,)
    if action_id == DROP_OUT:
        return "drop_out", ()
    if action_id < BUILD_BASE:
        e, s = divmod(action_id - EXTEND_BASE, S)
        return "extend", (e, s)
    if action_id < PASSENGER_TO_BASE:
        t, y = divmod(action_id - BUILD_BASE, NUM_TYPES)
        return "build", (t, y)
    if action_id < CLOCK_ADVANCE:
        return "passenger_to", (action_id - PASSENGER_TO_BASE,)
    if action_id == CLOCK_ADVANCE:
        return "clock_advance", ()
    if action_id == CLOCK_PAUSE:
        return "clock_pause", ()
    if action_id < DELIVER_TO_BASE:
        return "pick_passenger", (action_id - PICK_PASSENGER_BASE,)
    return "deliver_to", (action_id - DELIVER_TO_BASE,)


def site_ref(t: int) -> str:
    """``site 9 (B, n12)`` (ACTIONS.md section 4)."""
    site = _B.sites[t]
    return f"site {t} ({site.tier}, n{site.node})"


def street_ref(s: int) -> str:
    """``s34 (n7-n12)``."""
    st = _B.streets[s]
    return f"s{s} (n{st.a}-n{st.b})"


def action_to_string(seat: int, action_id: int) -> str:
    """State-independent action text (ACTIONS.md section 4)."""
    kind, p = decode_action(action_id)
    who = f"P{seat}: "
    if kind == "select":
        return f"{who}select {ACTION_NAMES[p[0]]}"
    if kind == "drop_out":
        return f"{who}drop out"
    if kind == "extend":
        return f"{who}extend end {p[0]} via {street_ref(p[1])}"
    if kind == "build":
        return f"{who}build {BUILDING_TYPES[p[1]]} at {site_ref(p[0])}"
    if kind == "passenger_to":
        return f"{who}passenger to station n{_B.stations[p[0]]}"
    if kind == "clock_advance":
        return f"{who}clock advance"
    if kind == "clock_pause":
        return f"{who}clock pause"
    if kind == "pick_passenger":
        return f"{who}pick passenger at n{p[0]}"
    return f"{who}deliver to {site_ref(p[0])}"
