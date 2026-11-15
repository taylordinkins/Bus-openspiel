"""Bus adapter for the server shell (docs/UI_DESIGN.md section 6)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import pyspiel
import yaml

from bus import actions, rules
from bus import constants as C
from bus.actions import decode_action
from bus.board import FRAME_SIZE
from bus.game import BusState, action_meta, game_result
from bus.lines import line_nodes
from bus.scenario import ACTION_KEYS, ScenarioError, dump_scenario, load_scenario
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

    def scenario_players(self, scenario: JSON) -> int:
        return self.scenario_params(scenario)["num_players"]

    def scenario_params(self, scenario: JSON) -> JSON:
        try:
            state = load_scenario(scenario, require_legal=True)
        except ScenarioError as e:
            raise ValueError(str(e)) from None
        return {"num_players": state.core.players, "returns": state.returns_mode}

    def named_scenarios(self) -> list[JSON]:
        return copy.deepcopy(_named_scenarios())

    def initial_state(self, game: pyspiel.Game, scenario: JSON | None) -> pyspiel.State:
        if scenario is None:
            return game.new_initial_state()
        try:
            return load_scenario(scenario, game, require_legal=True)
        except ScenarioError as e:
            raise ValueError(f"invalid scenario: {e}") from None

    def events(
        self, state_before: pyspiel.State, action_id: int, state_after: pyspiel.State
    ) -> list[JSON]:
        assert isinstance(state_before, BusState) and isinstance(state_after, BusState)
        return _events(state_before.core, action_id, state_after.core)

    def board_geometry(self) -> JSON:
        return _geometry()

    def view(self, state: pyspiel.State, seat: int) -> JSON:
        assert isinstance(state, BusState)
        derived = _derived(state)
        return {
            "state": dump_scenario(state),
            "derived": derived,
            "prompt": _prompt(state, derived),
            "targets": _targets(state),
        }


SCENARIOS_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "scenarios.yaml"


@cache
def _named_scenarios(manifest: Path = SCENARIOS_MANIFEST) -> list[JSON]:
    """Lobby scenarios from `data/scenarios.yaml` (files in `data/scenarios/`), loaded strictly.

    UI_DESIGN 6.9. The files are copies of test fixtures; production code never reads `tests/`.
    """
    raw: JSON = yaml.safe_load(manifest.read_text())
    base = manifest.parent.parent / raw["dir"]
    out: list[JSON] = []
    for entry in raw["scenarios"]:
        doc: JSON = json.loads((base / entry["file"]).read_text())
        doc.pop("expect", None)  # test assertions, not part of the position
        try:
            state = load_scenario(doc, require_legal=True)
        except ScenarioError as e:
            raise ValueError(f"named scenario {entry['id']!r}: {e}") from None
        pos = dump_scenario(state)
        out.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "description": doc.get("description", ""),
                "players": state.core.players,
                "returns": state.returns_mode,
                "round": pos["round"],
                "phase": pos["phase"],
                "scenario": doc,
            }
        )
    return out


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
            # R-023: Drop out is forbidden exactly while it is not legal (UI_DESIGN 6.7c)
            "mandatory": actions.DROP_OUT not in rules.legal_actions(core),
            "markers_left": core.seats[core.current].action_markers,
        }
    if ph == rules.PHASE_PX:
        return {"supply": core.supply}
    if ph == rules.PHASE_BD:
        return {"open_tier": derived["open_tier"]}
    clock = C.BUILDING_TYPES[core.clock]
    if ph == rules.PHASE_CK:
        nxt = C.BUILDING_TYPES[(core.clock + 1) % len(C.BUILDING_TYPES)]
        # R-053: the pause action is the one that takes the last stone (UI_DESIGN 6.7b)
        return {
            "clock": clock,
            "next_clock": nxt,
            "stones_on_clock": core.stones_on_clock,
            "last_stone": core.stones_on_clock - 1 <= 0,
        }
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
        "street_ends": {},
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
            t["street_ends"].setdefault(str(m["street"]), []).append(
                {"id": a, "end": m["end"], "node": m["from_node"]}
            )
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


def _events(before: rules.Core, action: int, after: rules.Core) -> list[JSON]:
    """Automatic steps between two cores (UI_DESIGN 6.6); a pure function of the cores."""
    out: list[JSON] = []
    b = rules.board()
    if before.phase == rules.PHASE_VD:
        t = decode_action(action)[1][0]
        seat = before.current
        out.append(
            {
                "type": "scored",
                "seat": seat,
                "points": after.seats[seat].points,
                "node": before.picked_node,
                "site": t,
            }
        )
    if before.phase == rules.PHASE_CK and not (
        after.phase == rules.PHASE_END and after.end_types == ("last_stone",)
    ):  # R-053: no movement
        entered = list(after.occupied)
        exited = list(before.occupied)
        if entered or exited:
            out.append(
                {
                    "type": "passengers_moved",
                    "exited": [{"site": t, "node": b.sites[t].node} for t in exited],
                    "entered": [{"site": t, "node": b.sites[t].node} for t in entered],
                }
            )
    # Selection ending (R-028) and markers passed over without a decision (R-029, R-043).
    positions = rules.resolution_positions(before)
    start: int | None = None
    owners = before.action_board
    if before.phase == rules.PHASE_SEL:
        kind, params = decode_action(action)
        board_ = before.action_board
        if kind == "select":
            a = params[0]
            board_ = board_[:a] + (board_[a] + (before.current,),) + board_[a + 1 :]
        owners = board_
        positions = rules.resolution_positions(replace(before, action_board=board_))
        if after.phase != rules.PHASE_SEL or after.round != before.round:
            out.append({"type": "selection_ended"})
            start = -1
    else:
        res = rules.resolving_action(before)
        if res is not None:
            start = positions.index((res, before.cursor))
    if start is not None:
        res_after = rules.resolving_action(after)
        if res_after is not None and after.round == before.round:
            end = positions.index((res_after, after.cursor))
        else:
            end = len(positions)
        buses = [s.buses for s in before.seats]
        for a, k in positions[start + 1 : end]:
            owner = owners[a][k]
            if a == rules.ACT_BUSES and buses[owner] < C.BUSES_PER_PLAYER:
                buses[owner] += 1
                out.append({"type": "bus_added", "seat": owner, "buses": buses[owner]})
            else:
                out.append(
                    {
                        "type": "marker_skipped",
                        "action": ACTION_KEYS[a],
                        "letter": k,
                        "owner": owner,
                    }
                )
    if (
        before.phase not in rules.SETUP_PHASES
        and before.phase != rules.PHASE_END
        and before.round >= 1
        and (after.phase == rules.PHASE_END or after.round > before.round)
    ):
        out.append(
            {"type": "round_ended", "round": before.round, "start_player": after.start_player}
        )
    if after.phase == rules.PHASE_SEL and after.round > before.round:
        out.append({"type": "round_started", "round": after.round})
    if after.phase == rules.PHASE_END and before.phase != rules.PHASE_END:
        out.append({"type": "game_ended", "end_types": list(after.end_types)})
    return out


def _geometry() -> JSON:
    b = rules.board()
    lay = b.layout
    return {
        "map": {"width": FRAME_SIZE, "height": FRAME_SIZE},
        "nodes": [
            {"id": n.id, "x": n.x, "y": n.y, "station": n.station, "roundabout": n.roundabout}
            for n in b.nodes
        ],
        "streets": [
            {"id": st.id, "a": st.a, "b": st.b, "via": [list(p) for p in st.via]}
            for st in b.streets
        ],
        "sites": [
            {"id": t.id, "tier": t.tier, "node": t.node, "x": t.x, "y": t.y} for t in b.sites
        ],
        "layout": {
            "actions": {k: {ltr: list(p) for ltr, p in v.items()} for k, v in lay.actions.items()},
            "bus_grid": [list(p) for p in lay.bus_grid],
            "clock": {
                "x": lay.clock_x,
                "y": lay.clock_y,
                "hand": lay.clock_hand,
                "radius": lay.clock_radius,
                "angles": dict(lay.clock_angles),
            },
            "stones": [list(p) for p in lay.stones],
            "score_track": [list(p) for p in lay.score_track],
        },
    }
