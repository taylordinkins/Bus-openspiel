"""Static board topology, loaded once from ``data/board.yaml``.

The board never changes during a game; only occupancy does (bus lines, buildings, passengers).
Everything here is immutable and safe to share between pyspiel states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

BOARD_PATH = Path(__file__).resolve().parent.parent / "data" / "board.yaml"
TIERS = ("A", "B", "C", "D")


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    x: int
    y: int
    station: bool
    roundabout: bool


@dataclass(frozen=True, slots=True)
class Street:
    id: int
    a: int
    b: int
    via: tuple[tuple[int, int], ...]  # bend points, drawing only

    def other(self, node: int) -> int:
        """The street's endpoint that isn't ``node``."""
        if node == self.a:
            return self.b
        if node == self.b:
            return self.a
        raise ValueError(f"node {node} is not an endpoint of street {self.id}")


@dataclass(frozen=True, slots=True)
class Site:
    id: int
    tier: str  # "A".."D"; lower tiers fill first
    node: int
    x: int
    y: int


Point = tuple[int, int]
MULTI_USE_ACTIONS = ("lines", "passengers", "buildings", "vrrooomm")
SINGLE_USE_ACTIONS = ("buses", "clock", "starting_player")
CLOCK_SEGMENTS = ("house", "office", "pub")


@dataclass(frozen=True, slots=True)
class Layout:
    """Placement-section and score-track pixel positions (drawing only)."""

    # mappings are read-only but unhashable, so they are left out of the hash (still compared)
    actions: Mapping[str, Mapping[str, Point]] = field(hash=False)  # key -> letter -> centre
    bus_grid: tuple[Point, ...]  # 20 cells of the Buses grid
    clock_x: int
    clock_y: int
    clock_hand: int  # hand length
    clock_radius: int
    clock_angles: Mapping[str, float] = field(hash=False)  # segment -> degrees (0=east, 90=south)
    stones: tuple[Point, ...]
    score_track: tuple[Point, ...]  # cell for score 1..20


@dataclass(frozen=True, slots=True)
class Board:
    nodes: tuple[Node, ...]
    streets: tuple[Street, ...]
    sites: tuple[Site, ...]
    # node id -> street ids touching it
    incident: tuple[tuple[int, ...], ...]
    # node id -> site ids at that intersection
    node_sites: tuple[tuple[int, ...], ...]
    layout: Layout

    @property
    def stations(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes if n.station)

    @property
    def roundabouts(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes if n.roundabout)

    def sites_of_tier(self, tier: str) -> tuple[int, ...]:
        return tuple(s.id for s in self.sites if s.tier == tier)


def parse_board(raw: dict[str, Any]) -> Board:
    """Build a validated ``Board`` from the raw ``board.yaml`` dict; ValueError on bad data."""
    nodes = tuple(
        Node(
            id=int(n["id"]),
            x=int(n["x"]),
            y=int(n["y"]),
            station=bool(n.get("station", False)),
            roundabout=bool(n.get("roundabout", False)),
        )
        for n in raw["nodes"]
    )
    streets = tuple(
        Street(
            id=int(s["id"]),
            a=int(s["a"]),
            b=int(s["b"]),
            via=tuple((int(p[0]), int(p[1])) for p in s.get("via", [])),
        )
        for s in raw["streets"]
    )
    sites = tuple(
        Site(
            id=int(s["id"]), tier=str(s["tier"]), node=int(s["node"]), x=int(s["x"]), y=int(s["y"])
        )
        for s in raw["sites"]
    )
    _validate(nodes, streets, sites)
    incident = tuple(tuple(s.id for s in streets if n.id in (s.a, s.b)) for n in nodes)
    node_sites = tuple(tuple(s.id for s in sites if s.node == n.id) for n in nodes)
    layout = _parse_layout(raw["layout"])
    return Board(nodes, streets, sites, incident, node_sites, layout)


def _pt(p: Any) -> Point:
    if not isinstance(p, (list, tuple)) or len(p) != 2:  # pyright: ignore[reportUnknownArgumentType]
        raise ValueError(f"layout point {p!r} must have exactly 2 elements")
    x, y = p  # pyright: ignore[reportUnknownVariableType]
    if (
        not (isinstance(x, int) and isinstance(y, int))
        or isinstance(x, bool)
        or isinstance(y, bool)
    ):
        raise ValueError(f"layout point {p!r} must hold integers")
    if not (0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE):
        raise ValueError(f"layout point ({x}, {y}) lies outside the map frame")
    return (x, y)


def _parse_layout(raw: dict[str, Any]) -> Layout:
    actions = {k: {str(ltr): _pt(p) for ltr, p in v.items()} for k, v in raw["actions"].items()}
    unknown = set(actions) - set(MULTI_USE_ACTIONS) - set(SINGLE_USE_ACTIONS)
    if unknown:
        raise ValueError(f"layout has unknown actions {sorted(unknown)}")
    for key in MULTI_USE_ACTIONS:
        if sorted(actions.get(key, {})) != list("abcdef"):
            raise ValueError(f"layout action {key} needs letters a-f")
    for key in SINGLE_USE_ACTIONS:
        if sorted(actions.get(key, {})) != ["a"]:
            raise ValueError(f"layout action {key} needs exactly letter a")
    bus_grid = tuple(_pt(p) for p in raw["bus_grid"])
    stones = tuple(_pt(p) for p in raw["stones"])
    track = tuple(_pt(p) for p in raw["score_track"])
    if len(bus_grid) != 20 or len(stones) != 5 or len(track) != 20:
        raise ValueError("layout needs 20 bus cells, 5 stone slots and 20 score cells")
    c = raw["clock"]
    _pt((c["x"], c["y"]))
    angles = {str(k): float(v) for k, v in c["segments"].items()}
    if set(angles) != set(CLOCK_SEGMENTS):
        raise ValueError("layout clock needs house, office and pub segments")
    hand, radius = int(c["hand"]), int(c["radius"])
    if hand <= 0 or radius <= 0:
        raise ValueError("layout clock hand and radius must be positive")
    return Layout(
        MappingProxyType({k: MappingProxyType(v) for k, v in actions.items()}),
        bus_grid,
        int(c["x"]),
        int(c["y"]),
        hand,
        radius,
        MappingProxyType(angles),
        stones,
        track,
    )


FRAME_SIZE = 1991  # map frame in px (data/board.yaml `map`), both axes


def _validate(
    nodes: tuple[Node, ...], streets: tuple[Street, ...], sites: tuple[Site, ...]
) -> None:
    points = [("node", n.id, n.x, n.y) for n in nodes]
    points += [("site", s.id, s.x, s.y) for s in sites]
    points += [("street via", st.id, x, y) for st in streets for x, y in st.via]
    for kind, ident, x, y in points:
        if not (0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE):
            raise ValueError(f"{kind} {ident} at ({x}, {y}) lies outside the map frame")
    for kind, items in (("node", nodes), ("street", streets), ("site", sites)):
        ids = [item.id for item in items]
        if ids != list(range(len(ids))):
            raise ValueError(f"{kind} ids must be 0..{len(ids) - 1} in order")
    n = len(nodes)
    pairs: set[tuple[int, int]] = set()
    for s in streets:
        if not (0 <= s.a < n and 0 <= s.b < n) or s.a == s.b:
            raise ValueError(f"street {s.id} has bad endpoints {s.a}, {s.b}")
        key = (min(s.a, s.b), max(s.a, s.b))
        if key in pairs:
            raise ValueError(f"street {s.id} duplicates {key}")
        pairs.add(key)
    for site in sites:
        if site.tier not in TIERS:
            raise ValueError(f"site {site.id} has unknown tier {site.tier!r}")
        if not 0 <= site.node < n:
            raise ValueError(f"site {site.id} points at missing node {site.node}")


@cache
def load_board(path: Path = BOARD_PATH) -> Board:
    """The board, parsed and validated (cached; the result is immutable)."""
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return parse_board(raw)
