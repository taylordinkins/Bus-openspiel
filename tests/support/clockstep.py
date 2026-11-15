"""Helpers for the D22 Clock-step tests (R-015, R-029, R-049..R-060, S-CLOCKSTEP).

Public API only.  Positions start from ``tests/scenarios/clockstep/report_setup.json`` (3 players,
round 1 selection, clock house, 4 stones on the clock) and are optionally edited through
``dump_scenario`` / ``load_scenario``.  Rounds are driven by a *plan*: seat -> list of selection
action ids; a seat with nothing left in its plan drops out.  Every seat must be given two entries
(the first two passes are mandatory, R-023).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from bus.actions import DELIVER_TO_BASE, DROP_OUT, PICK_PASSENGER_BASE
from bus.board import load_board
from tests import harness as H

# SELECT action ids (SELECT_BASE = 0)
LINES, BUSES, PASSENGERS, BUILDINGS, CLOCK, VRROOOMM, STARTING = range(7)
_B = load_board()
NODES = len(_B.nodes)
SITES = len(_B.sites)


def pick(n: int) -> int:
    return PICK_PASSENGER_BASE + n


def deliver(t: int) -> int:
    return DELIVER_TO_BASE + t


def dump(s: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(s)


def setup_state() -> Any:
    return H.load_fixture("clockstep/report_setup")


def setup_doc() -> dict[str, Any]:
    d = dump(setup_state())
    d.pop("supply", None)  # derived from conservation (R-005)
    return d


def load(doc: dict[str, Any]) -> Any:
    return H.load_doc_state(doc)


def set_building(doc: dict[str, Any], site: int, typ: str | None, occupied: bool = False) -> None:
    doc["buildings"] = [b for b in doc["buildings"] if b["site"] != site]
    if typ is not None:
        doc["buildings"].append({"site": site, "type": typ, "occupied": occupied})


def set_waiting(doc: dict[str, Any], waiting: dict[int, int]) -> None:
    doc["waiting"] = {str(n): c for n, c in waiting.items() if c}


def select(s: Any, plan: dict[int, list[int]]) -> None:
    """Play SEL to its end: each seat uses its plan entries in order, then drops out."""
    plan = copy.deepcopy(plan)
    guard = 0
    while s.phase == "SEL":
        q = plan.get(s.current_player(), [])
        s.apply_action(q.pop(0) if q else DROP_OUT)
        guard += 1
        assert guard < 100


def resolve_lines(s: Any) -> None:
    """Take the first legal action while a Lines marker is being resolved."""
    guard = 0
    while s.phase == "LN":
        s.apply_action(s.legal_actions()[0])
        guard += 1
        assert guard < 50


def play_round(
    s: Any, plan: dict[int, list[int]], stop: Callable[[Any], bool] | None = None
) -> None:
    """SEL with ``plan``, then Phase 2 with Lines nodes resolved by their first legal action,
    until a non-LN decision node (CK, PX, BD, VP, VD), the next SEL, or the end."""
    select(s, plan)
    resolve_lines(s)


ALL_LINES = {0: [LINES, LINES], 1: [LINES, LINES], 2: [LINES, LINES]}
"""No Clock, no Vrrooomm!: six Lines markers (letter a has power 1, the rest 0)."""


def next_round(s: Any) -> int:
    return int(dump(s)["round"])


def occupied(s: Any) -> set[int]:
    return {t for t in range(SITES) if s.is_occupied(t)}


def waiting(s: Any) -> dict[int, int]:
    return {n: s.waiting_at(n) for n in range(NODES) if s.waiting_at(n)}
