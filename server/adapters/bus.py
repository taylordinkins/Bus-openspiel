"""Bus adapter for the server shell (docs/UI_DESIGN.md section 6)."""

from __future__ import annotations

from typing import Any

import pyspiel

from bus import constants as C
from bus import rules
from bus.game import BusState, action_meta, game_result
from bus.lines import line_nodes
from bus.scenario import ACTION_KEYS, dump_scenario
from server.adapters.base import JSON, GameAdapter

_META_TOP = ("id", "kind", "player", "label")


class BusAdapter(GameAdapter):
    name = "bus"
    view_key = "view"

    def load(self, params: dict[str, Any]) -> pyspiel.Game:
        n = params.get("num_players")
        n = C.DEFAULT_PLAYERS if n is None else n
        if not isinstance(n, int) or not C.MIN_PLAYERS <= n <= C.MAX_PLAYERS:
            raise ValueError(f"num_players must be {C.MIN_PLAYERS}..{C.MAX_PLAYERS}, got {n!r}")
        g: dict[str, Any] = {"players": n}
        if params.get("returns") is not None:
            g["returns"] = params["returns"]
        return pyspiel.load_game("python_bus", g)

    def winners(self, state: pyspiel.State) -> list[int]:
        assert isinstance(state, BusState)
        return list(rules.winners(state.core))

    def action_meta(self, state: pyspiel.State, action_id: int) -> JSON:
        assert isinstance(state, BusState)
        m = action_meta(state, action_id)
        return {
            "label": m["label"],
            "kind": m["kind"],
            "params": {k: v for k, v in m.items() if k not in _META_TOP},
        }

    def view(self, state: pyspiel.State, seat: int) -> JSON:
        assert isinstance(state, BusState)
        derived = _derived(state)
        return {
            "state": dump_scenario(state),
            "derived": derived,
            "prompt": _prompt(state, derived),
            "targets": _targets(state),
        }


def _open_tier(core: rules.Core) -> str | None:
    sites = rules.lowest_open_tier_sites(core)
    return rules.board().sites[sites[0]].tier if sites else None


def _resolving(core: rules.Core) -> JSON | None:
    action = rules.resolving_action(core)
    if action is None:
        return None
    power = None
    if core.phase in (rules.PHASE_LN, rules.PHASE_PX, rules.PHASE_BD):
        power = rules.marker_power(core, action, core.cursor)
    return {
        "action": ACTION_KEYS[action],
        "letter": core.cursor,
        "owner": core.current,
        "power": power,
    }


def _derived(state: BusState) -> JSON:
    core = state.core
    b = rules.board()
    seats: list[JSON] = []
    street_lines: dict[str, list[int]] = {}
    for i, s in enumerate(core.seats):
        nodes = line_nodes(b, s.line) if s.line else []
        seats.append(
            {
                "line_markers_left": C.LINE_MARKERS_PER_PLAYER - len(s.line),
                "line_nodes": nodes,
                "ends": [nodes[0], nodes[-1]] if nodes else None,
                "dropped": core.phase == rules.PHASE_SEL and i in core.dropped,
            }
        )
        for st in sorted(set(s.line)):
            street_lines.setdefault(str(st), []).append(i)
    return {
        "seats": seats,
        "street_lines": street_lines,
        "open_tier": _open_tier(core),
        "resolving": _resolving(core),
        "result": game_result(state),
    }


def _context(core: rules.Core, derived: JSON) -> JSON:
    ph = core.phase
    if ph in rules.SETUP_PHASES:
        return {"done": core.setup_placed}
    if ph == rules.PHASE_SEL:
        return {
            "pass": core.sel_pass,
            "mandatory": core.sel_pass < C.MANDATORY_PASSES,
            "markers_left": core.seats[core.current].action_markers,
        }
    if ph == rules.PHASE_PX:
        return {"supply": core.supply}
    if ph == rules.PHASE_BD:
        return {"open_tier": derived["open_tier"]}
    clock = C.BUILDING_TYPES[core.clock]
    if ph == rules.PHASE_CK:
        nxt = C.BUILDING_TYPES[(core.clock + 1) % len(C.BUILDING_TYPES)]
        return {"clock": clock, "next_clock": nxt, "stones_on_clock": core.stones_on_clock}
    if ph in (rules.PHASE_VP, rules.PHASE_VD):
        picked = core.picked_node if ph == rules.PHASE_VD else None
        return {"clock": clock, "moves_left": core.moves_left, "picked_node": picked}
    return {}


def _prompt(state: BusState, derived: JSON) -> JSON:
    core = state.core
    ph = core.phase
    units: int | None = None
    if ph in (rules.PHASE_LN, rules.PHASE_PX, rules.PHASE_BD):
        units = core.units_left
    elif ph in (rules.PHASE_VP, rules.PHASE_VD):
        units = core.moves_left
    res = derived["resolving"]
    return {
        "phase": ph,
        "actor": None if state.is_terminal() else core.current,
        "units_left": units,
        "power": None if res is None else res["power"],
        "context": _context(core, derived),
    }


def _targets(state: BusState) -> JSON:
    t: JSON = {
        "streets": {},
        "sites": {},
        "stations": {},
        "nodes": {},
        "actions": {},
        "clock": {"advance": None, "pause": None},
        "buttons": [],
    }
    if state.is_terminal():
        return t
    for a in state.legal_actions():
        m = action_meta(state, a)
        kind = m["kind"]
        if kind == "extend":
            t["streets"].setdefault(str(m["street"]), []).append(a)
        elif kind == "build":
            t["sites"].setdefault(str(m["site"]), []).append(a)
        elif kind == "deliver_to":
            t["sites"][str(m["site"])] = [a]
        elif kind == "passenger_to":
            t["stations"][str(m["node"])] = a
        elif kind == "pick_passenger":
            t["nodes"][str(m["node"])] = a
        elif kind == "select":
            t["actions"][ACTION_KEYS[m["action"]]] = a
        elif kind == "clock_advance":
            t["clock"]["advance"] = a
        elif kind == "clock_pause":
            t["clock"]["pause"] = a
        else:
            t["buttons"].append({"id": a, "kind": kind})
    return t
