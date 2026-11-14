"""Scenario loader, dumper and expectation checker (docs/SCENARIOS.md).

A scenario is a JSON/YAML position document. ``load_scenario`` validates it (SCENARIOS section 6
and 9), builds the core state, applies ``history`` and returns a ``BusState``. ``dump_scenario``
emits a fully resolved document that loads back to the same state.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pyspiel
import yaml

from bus import constants as C
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS
from bus.game import BusGame, BusState
from bus.rules import EMPTY, Core, Seat

FORMAT = 1
ACTION_KEYS = ("lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player")
RESOLVE_ACTION = {
    rules.PHASE_LN: "lines",
    rules.PHASE_PX: "passengers",
    rules.PHASE_BD: "buildings",
    rules.PHASE_CK: "clock",
    rules.PHASE_VP: "vrrooomm",
    rules.PHASE_VD: "vrrooomm",
}
BOARD_PHASES = (rules.PHASE_SEL, *RESOLVE_ACTION)  # phases where the action board may be filled
TOP_KEYS = {
    "format",
    "description",
    "parameters",
    "round",
    "phase",
    "current_player",
    "start_player",
    "clock",
    "stones_on_clock",
    "seats",
    "buildings",
    "waiting",
    "supply",
    "action_board",
    "pending",
    "end",
    "move_offset",
    "allow_unreachable",
    "strict",
    "history",
    "expect",
}
SEAT_KEYS = {"points", "stones", "buses", "markers_left", "line", "last_score_move"}
BUILDING_KEYS = {"site", "type", "occupied"}
EXPECT_KEYS = {
    "phase",
    "current_player",
    "legal_actions",
    "legal_includes",
    "legal_excludes",
    "points",
    "final_scores",
    "returns",
    "winners",
    "buses",
    "stones",
    "clock",
    "lines",
    "buildings",
    "waiting",
    "supply",
    "start_player",
    "terminal",
    "end_types",
}
MAX_LINE = C.LINE_MARKERS_PER_PLAYER
MAX_UNITS = 6  # SCENARIOS 4.2
MAX_MOVES = 5


class ScenarioError(Exception):
    """An invalid scenario; ``code`` is one of the ``S-*`` codes of SCENARIOS section 6."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _err(code: str, message: str) -> ScenarioError:
    return ScenarioError(code, message)


# ---------------------------------------------------------------- typed accessors


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _dict(v: Any, what: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise _err("S-SCHEMA", f"{what} must be an object")
    out: dict[str, Any] = {}
    for k, x in cast(dict[Any, Any], v).items():
        if _is_int(k):
            k = str(k)
        if not isinstance(k, str):
            raise _err("S-SCHEMA", f"{what} has a non-string key")
        out[k] = x
    return out


def _list(v: Any, what: str) -> list[Any]:
    if not isinstance(v, list):
        raise _err("S-SCHEMA", f"{what} must be a list")
    return cast(list[Any], v)


def _int(
    v: Any, what: str, lo: int | None = None, hi: int | None = None, code: str = "S-SCHEMA"
) -> int:
    if not _is_int(v):
        raise _err("S-SCHEMA", f"{what} must be an integer")
    n = cast(int, v)
    if (lo is not None and n < lo) or (hi is not None and n > hi):
        raise _err(code, f"{what} = {n} out of range")
    return n


def _only_keys(d: dict[str, Any], allowed: set[str], what: str) -> None:
    extra = sorted(set(d) - allowed)
    if extra:
        raise _err("S-SCHEMA", f"unknown key(s) in {what}: {extra}")


def _seat_id(v: Any, what: str, n: int) -> int:
    if not _is_int(v):
        raise _err("S-SCHEMA", f"{what} must be an integer seat")
    s = cast(int, v)
    if not 0 <= s < n:
        raise _err("S-ID", f"{what} = {s} is not a seat")
    return s


# ---------------------------------------------------------------- reading


def _read(doc: Any) -> dict[str, Any]:
    if isinstance(doc, (str, Path)):
        path = Path(doc)
        try:
            text = path.read_text(encoding="utf-8")
            data: Any = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        except (OSError, ValueError, yaml.YAMLError) as e:
            raise _err("S-SCHEMA", f"cannot read scenario {path}: {e}") from e
        return _dict(data, "scenario")
    return _dict(copy.deepcopy(doc), "scenario")


# ---------------------------------------------------------------- parts


def _check_line(line: list[int]) -> None:
    """Implements R-035, R-037: a line is a trail of distinct streets."""
    b = rules.board()
    if len(line) > MAX_LINE:
        raise _err("S-LINE", f"line longer than {MAX_LINE}")
    if len(set(line)) != len(line):
        raise _err("S-LINE", "a street appears twice in one line")
    if len(line) < 2:
        return
    s0, s1 = b.streets[line[0]], b.streets[line[1]]
    shared = {s0.a, s0.b} & {s1.a, s1.b}
    if not shared:
        raise _err("S-LINE", "consecutive streets do not share a node")
    cur = s1.other(next(iter(shared)))
    for sid in line[2:]:
        st = b.streets[sid]
        if cur not in (st.a, st.b):
            raise _err("S-LINE", f"street {sid} does not continue the line")
        cur = st.other(cur)


def _parse_seat(raw: Any, i: int, n_streets: int) -> tuple[Seat, int]:
    d = _dict(raw, f"seats[{i}]")
    _only_keys(d, SEAT_KEYS, f"seats[{i}]")
    points = _int(d.get("points", 0), "points", lo=0)
    stones = _int(d.get("stones", 0), "stones", lo=0)
    buses = _int(d.get("buses", C.INITIAL_BUSES), "buses")
    if not C.INITIAL_BUSES <= buses <= C.BUSES_PER_PLAYER:
        raise _err("S-BUS", f"buses = {buses} outside 1..{C.BUSES_PER_PLAYER}")
    markers = _int(d.get("markers_left", C.ACTION_MARKERS_PER_PLAYER), "markers_left")
    if not 0 <= markers <= C.ACTION_MARKERS_PER_PLAYER:
        raise _err("S-MARKERS", f"markers_left = {markers} outside 0..20")
    line = [_int(x, "line street") for x in _list(d.get("line", []), "line")]
    for sid in line:
        if not 0 <= sid < n_streets:
            raise _err("S-ID", f"street {sid} out of range")
    _check_line(line)
    last = d.get("last_score_move")
    if last is not None:
        last = _int(last, "last_score_move", lo=1)
    seat = Seat(
        points=points,
        buses=buses,
        action_markers=markers,
        line_markers=C.LINE_MARKERS_PER_PLAYER - len(line),
        stones=stones,
        line=tuple(line),
        last_score_move=last,
    )
    return seat, markers


def _parse_seats(d: dict[str, Any], n: int, n_streets: int) -> list[Seat]:
    raw = d.get("seats")
    items: list[Any]
    if raw is None:
        items = [{} for _ in range(n)]
    elif isinstance(raw, list):
        items = cast(list[Any], raw)
        if len(items) != n:
            raise _err("S-PLAYERS", f"seats has {len(items)} entries for {n} players")
    elif isinstance(raw, dict):
        m = _dict(raw, "seats")
        items = [{} for _ in range(n)]
        for k, v in m.items():
            if not k.isdigit():
                raise _err("S-SCHEMA", f"seats key {k!r} is not a seat number")
            if int(k) >= n:
                raise _err("S-PLAYERS", f"seats key {k} beyond {n} players")
            items[int(k)] = v
    else:
        raise _err("S-SCHEMA", "seats must be a list or an object")
    return [_parse_seat(v, i, n_streets)[0] for i, v in enumerate(items)]


def _parse_buildings(d: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    b = rules.board()
    types = [EMPTY] * len(b.sites)
    occupied: list[int] = []
    for raw in _list(d.get("buildings", []), "buildings"):
        e = _dict(raw, "building")
        _only_keys(e, BUILDING_KEYS, "building")
        if "site" not in e or "type" not in e:
            raise _err("S-SCHEMA", "building needs site and type")
        site = _int(e["site"], "site")
        if not 0 <= site < len(b.sites):
            raise _err("S-ID", f"site {site} out of range")
        ty = e["type"]
        if not isinstance(ty, str) or ty not in C.BUILDING_TYPES:
            raise _err("S-SITES", f"invalid building type {ty!r}")
        if types[site] != EMPTY:
            raise _err("S-SITES", f"site {site} appears twice")
        types[site] = C.BUILDING_TYPES.index(ty)
        occ = e.get("occupied", False)
        if not isinstance(occ, bool):
            raise _err("S-SCHEMA", "occupied must be a boolean")
        if occ:
            occupied.append(site)
    return tuple(types), tuple(sorted(occupied))


def _parse_waiting(d: dict[str, Any], rnd: int) -> tuple[int, ...]:
    b = rules.board()
    n_nodes = len(b.nodes)
    if "waiting" not in d:
        if rnd >= 1:
            raise _err("S-SCHEMA", "waiting is required from round 1")
        return tuple(1 if n in b.roundabouts else 0 for n in range(n_nodes))
    w = [0] * n_nodes
    for k, v in _dict(d["waiting"], "waiting").items():
        try:
            node = int(k)
        except ValueError as e:
            raise _err("S-PASSENGERS", f"waiting key {k!r} is not a node") from e
        if not 0 <= node < n_nodes:
            raise _err("S-ID", f"waiting node {node} out of range")
        w[node] = _int(v, "waiting count", lo=1, code="S-PASSENGERS")
    return tuple(w)


def _parse_board(d: dict[str, Any], n: int) -> tuple[tuple[int, ...], ...]:
    lay = rules.board().layout.actions
    raw = _dict(d.get("action_board", {}), "action_board")
    unknown = sorted(set(raw) - set(ACTION_KEYS))
    if unknown:
        raise _err("S-SCHEMA", f"unknown action(s) on the board: {unknown}")
    out: list[tuple[int, ...]] = []
    for key in ACTION_KEYS:
        spaces = len(lay[key])
        entries = _list(raw.get(key, [None] * spaces), f"action_board.{key}")
        if len(entries) != spaces:
            raise _err("S-BOARD", f"{key} needs {spaces} entries")
        filled: list[int] = []
        gap = False
        for e in entries:
            if e is None:
                gap = True
                continue
            if gap:
                raise _err("S-BOARD", f"{key} letters are not contiguous")
            filled.append(_seat_id(e, f"{key} marker", n))
        out.append(tuple(filled))
    return tuple(out)


def _parse_pending(
    d: dict[str, Any], phase: str, n: int, board: tuple[tuple[int, ...], ...]
) -> dict[str, Any]:
    """Pending sub-decision (SCENARIOS 4, 9.1) as Core field values."""
    defaults: dict[str, dict[str, Any]] = {
        rules.PHASE_SB: {"done": 0},
        rules.PHASE_L1: {"done": 0},
        rules.PHASE_L2: {"done": 0},
        rules.PHASE_SEL: {"pass": 0, "dropped": []},
    }
    p = _dict(d["pending"], "pending") if "pending" in d else dict(defaults.get(phase, {}))
    keys: dict[str, set[str]] = {
        rules.PHASE_SB: {"done"},
        rules.PHASE_L1: {"done"},
        rules.PHASE_L2: {"done"},
        rules.PHASE_SEL: {"pass", "dropped"},
        rules.PHASE_LN: {"cursor", "units_left"},
        rules.PHASE_PX: {"cursor", "units_left"},
        rules.PHASE_BD: {"cursor", "units_left"},
        rules.PHASE_CK: {"cursor"},
        rules.PHASE_VP: {"cursor", "moves_left"},
        rules.PHASE_VD: {"cursor", "moves_left", "picked_node"},
        rules.PHASE_END: set(),
    }
    if set(p) != keys[phase]:
        raise _err("S-PHASE", f"pending for {phase} needs exactly {sorted(keys[phase])}")
    f: dict[str, Any] = {}
    if phase == rules.PHASE_SB:
        f["setup_placed"] = _int(p["done"], "done", 0, 2 * n - 1, "S-PHASE")
    elif phase in (rules.PHASE_L1, rules.PHASE_L2):
        f["setup_placed"] = _int(p["done"], "done", 0, n - 1, "S-PHASE")
    elif phase == rules.PHASE_SEL:
        f["sel_pass"] = _int(p["pass"], "pass", lo=0)
        f["dropped"] = tuple(_seat_id(x, "dropped seat", n) for x in _list(p["dropped"], "dropped"))
    elif phase in RESOLVE_ACTION:
        cur = _dict(p["cursor"], "cursor")
        if set(cur) != {"letter"}:
            raise _err("S-PHASE", "cursor needs exactly {'letter': k}")
        letter = _int(cur["letter"], "cursor letter", lo=0)
        owners = board[ACTION_KEYS.index(RESOLVE_ACTION[phase])]
        if letter >= len(owners):
            raise _err("S-PHASE", "cursor points at an empty letter")
        f["cursor"] = letter
        if "units_left" in p:
            f["units_left"] = _int(p["units_left"], "units_left", 1, MAX_UNITS, "S-PHASE")
        if "moves_left" in p:
            f["moves_left"] = _int(p["moves_left"], "moves_left", 1, MAX_MOVES, "S-PHASE")
        if "picked_node" in p:
            node = _int(p["picked_node"], "picked_node")
            if not 0 <= node < len(rules.board().nodes):
                raise _err("S-ID", f"picked_node {node} out of range")
            f["picked_node"] = node
    return f


def _parse_end(d: dict[str, Any], phase: str) -> tuple[str, ...]:
    if phase != rules.PHASE_END:
        if "end" in d:
            raise _err("S-PHASE", "end block outside END")
        return ()
    if "end" not in d:
        raise _err("S-END", "END needs an end block")
    e = d["end"]
    if not isinstance(e, dict):
        raise _err("S-END", "end must be an object")
    ed = _dict(e, "end")
    types = ed.get("types")
    if set(ed) != {"types"} or not isinstance(types, list):
        raise _err("S-END", "end needs exactly a types list")
    tl = cast(list[Any], types)
    if not tl or any(t not in rules.END_TYPES for t in tl) or len(set(tl)) != len(tl):
        raise _err("S-END", f"end types must be a non-empty subset of {rules.END_TYPES}")
    if "last_stone" in tl and len(tl) > 1:
        raise _err("S-END", "last_stone must stand alone")
    return tuple(sorted(cast(list[str], tl)))


# ---------------------------------------------------------------- build


def _resolve_game(d: dict[str, Any], game: Any) -> tuple[int, str]:
    gp: dict[str, Any] = dict(game.get_parameters()) if game is not None else {}
    n_def = int(gp.get("players", C.DEFAULT_PLAYERS))
    r_def = str(gp.get("returns", C.DEFAULT_RETURNS))
    p = _dict(d["parameters"], "parameters") if "parameters" in d else {}
    _only_keys(p, {"players", "returns"}, "parameters")
    n = _int(p["players"], "players") if "players" in p else n_def
    if not C.MIN_PLAYERS <= n <= C.MAX_PLAYERS:
        raise _err("S-PLAYERS", f"players = {n} outside {C.MIN_PLAYERS}..{C.MAX_PLAYERS}")
    mode = p.get("returns", r_def)
    if not isinstance(mode, str) or mode not in C.RETURNS_MODES:
        raise _err("S-SCHEMA", f"returns must be one of {C.RETURNS_MODES}")
    if game is not None and n != n_def:
        raise _err("S-PLAYERS", "scenario players differ from the game's")
    if game is not None and mode != r_def:
        raise _err("S-SCHEMA", "scenario returns differ from the game's")
    return n, mode


def _derive_sel_current(
    order: tuple[int, ...],
    placed: list[int],
    seats: list[Seat],
    dropped: tuple[int, ...],
    pass_: int,
) -> int | None:
    for p in order:
        if p in dropped or seats[p].action_markers == 0:
            continue
        if placed[p] <= pass_:
            return p
    return None


def _build(d: dict[str, Any], game: Any) -> tuple[Core, str]:
    b = rules.board()
    if d.get("format") != FORMAT or not _is_int(d.get("format")):
        raise _err("S-SCHEMA", "format must be 1")
    _only_keys(d, TOP_KEYS, "scenario")
    n, mode = _resolve_game(d, game)

    rnd = _int(d.get("round", 0), "round", lo=0)
    ph: Any = d.get("phase", rules.PHASE_SEL if rnd >= 1 else rules.PHASE_SB)
    if not isinstance(ph, str) or ph not in rules.PHASES:
        raise _err("S-PHASE", f"unknown phase {ph!r}")
    phase: str = ph
    if (phase in rules.SETUP_PHASES) != (rnd == 0):
        raise _err("S-PHASE", f"phase {phase} is not compatible with round {rnd}")

    seats = _parse_seats(d, n, len(b.streets))
    site_types, occupied = _parse_buildings(d)
    waiting = _parse_waiting(d, rnd)
    board = _parse_board(d, n)
    start = _seat_id(d.get("start_player", 0), "start_player", n)
    explicit_cp = d.get("current_player")
    if explicit_cp is not None:
        explicit_cp = _seat_id(explicit_cp, "current_player", n)
    clock_name = d.get("clock", C.BUILDING_TYPES[0])
    if not isinstance(clock_name, str) or clock_name not in C.BUILDING_TYPES:
        raise _err("S-SCHEMA", f"clock must be one of {C.BUILDING_TYPES}")
    on_clock = _int(d.get("stones_on_clock", C.time_stones(n)), "stones_on_clock", lo=0)
    allow_unreachable = d.get("allow_unreachable", False)
    strict = d.get("strict", False)
    if not isinstance(allow_unreachable, bool) or not isinstance(strict, bool):
        raise _err("S-SCHEMA", "allow_unreachable and strict must be booleans")
    lasts = [s.last_score_move or 0 for s in seats]
    offset = _int(d.get("move_offset", max(lasts)), "move_offset", lo=0)
    pend = _parse_pending(d, phase, n, board)
    end_types = _parse_end(d, phase)

    # S-PASSENGERS (R-005)
    supply = C.TOTAL_PASSENGERS - sum(waiting) - len(occupied)
    if supply < 0:
        raise _err("S-PASSENGERS", "more than 15 passengers on the board")
    if "supply" in d and _int(d["supply"], "supply") != supply:
        raise _err("S-PASSENGERS", f"supply must equal the derived value {supply}")

    # S-BOARD
    if phase not in BOARD_PHASES and any(board):
        raise _err("S-BOARD", f"the action board must be empty in {phase}")

    # S-MARKERS (R-004)
    for p, s in enumerate(seats):
        on_board = sum(x.count(p) for x in board)
        if s.action_markers + on_board > C.ACTION_MARKERS_PER_PLAYER:
            raise _err("S-MARKERS", f"seat {p} holds more than 20 Action markers")

    # S-STONES (R-006, R-053)
    if on_clock + sum(s.stones for s in seats) != C.time_stones(n):
        raise _err("S-STONES", f"stones must total {C.time_stones(n)}")
    if on_clock == 0 and phase != rules.PHASE_END:
        raise _err("S-STONES", "stones_on_clock = 0 outside END")

    # S-POINTS
    for p, s in enumerate(seats):
        if not allow_unreachable and (s.last_score_move is None) != (s.points == 0):
            raise _err("S-POINTS", f"seat {p}: last_score_move is null iff points = 0")
        if (s.last_score_move or 0) > offset:
            raise _err("S-POINTS", f"seat {p}: last_score_move exceeds move_offset")

    # lines (empty only in setup)
    for p, s in enumerate(seats):
        if rnd >= 1 and not s.line:
            raise _err("S-LINE", f"seat {p} has an empty line in round {rnd}")

    core = Core(
        players=n,
        seats=tuple(seats),
        site_types=site_types,
        waiting=waiting,
        supply=supply,
        clock=C.BUILDING_TYPES.index(clock_name),
        stones_on_clock=on_clock,
        action_board=board,
        round=rnd,
        phase=phase,
        current=0,
        setup_placed=pend.get("setup_placed", 0),
        start_player=start,
        move_number=offset,
        sel_pass=pend.get("sel_pass", 0),
        dropped=pend.get("dropped", ()),
        cursor=pend.get("cursor", 0),
        units_left=pend.get("units_left", 0),
        moves_left=pend.get("moves_left", 0),
        picked_node=pend.get("picked_node", -1),
        occupied=occupied,
        end_types=end_types,
    )
    order = rules.turn_order(core)
    done = core.setup_placed
    current = start

    if phase in rules.SETUP_PHASES:
        n_bld = sum(1 for t in site_types if t != EMPTY)
        if phase == rules.PHASE_SB:
            if n_bld != done or any(s.line for s in seats):
                raise _err("S-SETUP", "SB: buildings must equal done and lines be empty")
            current = order[done // rules.SETUP_BUILDINGS_PER_SEAT]
            per = [
                min(
                    rules.SETUP_BUILDINGS_PER_SEAT,
                    max(0, done - rules.SETUP_BUILDINGS_PER_SEAT * i),
                )
                for i in range(n)
            ]
            blds = {order[i]: per[i] for i in range(n)}
        else:
            if n_bld != rules.SETUP_BUILDINGS_PER_SEAT * n:
                raise _err("S-SETUP", f"{phase} needs {2 * n} buildings")
            served = set(order[:done]) if phase == rules.PHASE_L1 else set(order[n - done :])
            base = 0 if phase == rules.PHASE_L1 else 1
            for p, s in enumerate(seats):
                if len(s.line) != base + (1 if p in served else 0):
                    raise _err("S-SETUP", f"{phase}: seat {p} line length does not fit done")
            current = order[done] if phase == rules.PHASE_L1 else order[n - 1 - done]
            blds = dict.fromkeys(range(n), rules.SETUP_BUILDINGS_PER_SEAT)
        seats = [replace(s, buildings=blds[p]) for p, s in enumerate(seats)]
    else:
        seats = [replace(s, buildings=rules.SETUP_BUILDINGS_PER_SEAT) for s in seats]
        if phase == rules.PHASE_SEL:
            placed = [sum(x.count(p) for x in board) for p in range(n)]
            for p in core.dropped:
                if placed[p] < 2:
                    raise _err("S-SEL", f"dropped seat {p} placed fewer than 2 markers")
            cp = _derive_sel_current(order, placed, seats, core.dropped, core.sel_pass)
            if cp is None:
                raise _err("S-SEL", "no seat is left to act in this pass")
            seen = False
            for p in order:
                if p in core.dropped or seats[p].action_markers == 0:
                    continue
                seen = seen or p == cp
                want = core.sel_pass if seen else core.sel_pass + 1
                if placed[p] != want:
                    raise _err("S-SEL", f"seat {p} has {placed[p]} markers, expected {want}")
            current = cp
        elif phase in RESOLVE_ACTION:
            owner = board[ACTION_KEYS.index(RESOLVE_ACTION[phase])][core.cursor]
            current = owner
            if phase == rules.PHASE_VD and waiting[core.picked_node] < 1:
                raise _err("S-PHASE", "VD: no passenger waits at picked_node")
    if phase == rules.PHASE_END:
        if explicit_cp is not None:
            raise _err("S-PHASE", "END has no current_player")
        current = 0
    elif explicit_cp is not None and explicit_cp != current:
        raise _err("S-PHASE", f"current_player must be {current} in {phase}")

    if strict and _tier_skipped(site_types):
        raise _err("S-SITES", "strict: buildings do not fill tier by tier (R-010)")

    return replace(core, seats=tuple(seats), current=current), mode


def _tier_skipped(site_types: tuple[int, ...]) -> bool:
    """True if a building stands in a tier while a lower tier has an empty site (R-010)."""
    b = rules.board()
    tiers = sorted({s.tier for s in b.sites})
    seen_empty = False
    for tier in tiers:
        sites = b.sites_of_tier(tier)
        if seen_empty and any(site_types[t] != EMPTY for t in sites):
            return True
        if any(site_types[t] == EMPTY for t in sites):
            seen_empty = True
    return False


def _check_expect_schema(d: dict[str, Any]) -> None:
    if "expect" in d:
        e = _dict(d["expect"], "expect")
        _only_keys(e, EXPECT_KEYS, "expect")


def load_scenario(doc: dict[str, Any] | str | Path, game: Any = None) -> BusState:
    """Load, validate and play out a scenario document. Implements SCENARIOS 2-6 and 9."""
    d = _read(doc)
    core, mode = _build(d, game)
    _check_expect_schema(d)
    history_raw = _list(d.get("history", []), "history")
    if game is None:
        game = pyspiel.load_game("python_bus", {"players": core.players, "returns": mode})
    state = cast(BusGame, game).new_initial_state()
    state.core = core
    for a in history_raw:
        if not _is_int(a) or not 0 <= cast(int, a) < NUM_DISTINCT_ACTIONS:
            raise _err("S-HISTORY", f"action id {a!r} out of range")
        if a not in rules.legal_actions(state.core):
            raise _err("S-HISTORY", f"action {a} is illegal in phase {state.core.phase}")
        state.apply_action(a)
    return state


# ---------------------------------------------------------------- dump


def _dump_pending(core: Core) -> dict[str, Any]:
    ph = core.phase
    if ph in rules.SETUP_PHASES:
        return {"done": core.setup_placed}
    if ph == rules.PHASE_SEL:
        return {"pass": core.sel_pass, "dropped": list(core.dropped)}
    if ph == rules.PHASE_END:
        return {}
    out: dict[str, Any] = {"cursor": {"letter": core.cursor}}
    if ph in (rules.PHASE_LN, rules.PHASE_PX, rules.PHASE_BD):
        out["units_left"] = core.units_left
    if ph in (rules.PHASE_VP, rules.PHASE_VD):
        out["moves_left"] = core.moves_left
    if ph == rules.PHASE_VD:
        out["picked_node"] = core.picked_node
    return out


def dump_scenario(state: BusState) -> dict[str, Any]:
    """Fully resolved, JSON-serialisable scenario of ``state``. Implements SCENARIOS 1, 9.5."""
    core = state.core
    lay = rules.board().layout.actions
    out: dict[str, Any] = {
        "format": FORMAT,
        "parameters": {"players": core.players, "returns": state.returns_mode},
        "round": core.round,
        "phase": core.phase,
    }
    if core.phase != rules.PHASE_END:
        out["current_player"] = core.current
    out["start_player"] = core.start_player
    out["clock"] = C.BUILDING_TYPES[core.clock]
    out["stones_on_clock"] = core.stones_on_clock
    out["seats"] = [
        {
            "points": s.points,
            "stones": s.stones,
            "buses": s.buses,
            "markers_left": s.action_markers,
            "line": list(s.line),
            "last_score_move": s.last_score_move,
        }
        for s in core.seats
    ]
    out["buildings"] = [
        {"site": t, "type": C.BUILDING_TYPES[y], "occupied": t in core.occupied}
        for t, y in enumerate(core.site_types)
        if y != EMPTY
    ]
    out["waiting"] = {str(n): w for n, w in enumerate(core.waiting) if w}
    out["supply"] = core.supply
    out["action_board"] = {
        key: list(core.action_board[i]) + [None] * (len(lay[key]) - len(core.action_board[i]))
        for i, key in enumerate(ACTION_KEYS)
    }
    out["pending"] = _dump_pending(core)
    if core.phase == rules.PHASE_END:
        out["end"] = {"types": list(core.end_types)}
    out["move_offset"] = core.move_number
    if any((s.last_score_move is None) != (s.points == 0) for s in core.seats):
        out["allow_unreachable"] = True  # the points / last_score_move link is relaxed
    return out


# ---------------------------------------------------------------- expect


def _eq(name: str, got: Any, want: Any) -> None:
    if got != want:
        raise AssertionError(f"expect {name}: got {got!r}, want {want!r}")


def check_expect(state: BusState, expect: dict[str, Any]) -> None:
    """Raise AssertionError if ``state`` violates an assertion of SCENARIOS 5."""
    e = _dict(expect, "expect")
    _only_keys(e, EXPECT_KEYS, "expect")
    core = state.core
    legal = sorted(state.legal_actions())
    for key, want in e.items():
        if key == "phase":
            _eq(key, state.phase, want)
        elif key == "current_player":
            _eq(key, state.current_player(), want)
        elif key == "legal_actions":
            _eq(key, legal, sorted(want))
        elif key == "legal_includes":
            missing = sorted(set(want) - set(legal))
            if missing:
                raise AssertionError(f"expect legal_includes: {missing} not legal")
        elif key == "legal_excludes":
            present = sorted(set(want) & set(legal))
            if present:
                raise AssertionError(f"expect legal_excludes: {present} are legal")
        elif key == "points":
            _eq(key, [s.points for s in core.seats], want)
        elif key == "buses":
            _eq(key, [s.buses for s in core.seats], want)
        elif key == "stones":
            _eq(key, [s.stones for s in core.seats], want)
        elif key == "clock":
            _eq(key, state.clock, want)
        elif key == "lines":
            _eq(key, [list(s.line) for s in core.seats], want)
        elif key == "buildings":
            got = {
                t: (C.BUILDING_TYPES[y], t in core.occupied)
                for t, y in enumerate(core.site_types)
                if y != EMPTY
            }
            for b in want:
                site = b["site"]
                if site not in got or got[site][0] != b["type"]:
                    raise AssertionError(f"expect buildings: site {site} is {got.get(site)}")
                if "occupied" in b and got[site][1] != b["occupied"]:
                    raise AssertionError(f"expect buildings: site {site} occupancy differs")
            _eq("buildings count", len(got), len(want))
        elif key == "waiting":
            _eq(key, {str(n): w for n, w in enumerate(core.waiting) if w}, want)
        elif key == "supply":
            _eq(key, state.passenger_supply, want)
        elif key == "start_player":
            _eq(key, state.start_player, want)
        elif key == "terminal":
            _eq(key, state.is_terminal(), want)
        elif key == "end_types":
            _eq(key, sorted(core.end_types), sorted(want))
        elif key == "returns":
            got_r = [float(x) for x in state.returns()]
            if len(got_r) != len(want) or any(
                abs(a - b) > 1e-9 for a, b in zip(got_r, want, strict=False)
            ):
                raise AssertionError(f"expect returns: got {got_r}, want {want}")
        else:  # final_scores, winners: scoring arrives with unit 12
            raise AssertionError(f"expect {key} is not supported yet")


__all__ = [
    "ACTION_KEYS",
    "ScenarioError",
    "check_expect",
    "dump_scenario",
    "load_scenario",
]
