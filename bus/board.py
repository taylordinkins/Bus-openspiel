"""Static board topology, loaded once from ``data/board.yaml``.

The board never changes during a game; only occupancy does (bus lines, buildings, passengers).
Everything here is immutable and safe to share between pyspiel states.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
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


@dataclass(frozen=True, slots=True)
class Board:
    nodes: tuple[Node, ...]
    streets: tuple[Street, ...]
    sites: tuple[Site, ...]
    # node id -> street ids touching it
    incident: tuple[tuple[int, ...], ...]
    # node id -> site ids at that intersection
    node_sites: tuple[tuple[int, ...], ...]

    @property
    def stations(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes if n.station)

    @property
    def roundabouts(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes if n.roundabout)

    def sites_of_tier(self, tier: str) -> tuple[int, ...]:
        return tuple(s.id for s in self.sites if s.tier == tier)


def _parse(raw: dict[str, Any]) -> Board:
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
    return Board(nodes, streets, sites, incident, node_sites)


def _validate(
    nodes: tuple[Node, ...], streets: tuple[Street, ...], sites: tuple[Site, ...]
) -> None:
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
    return _parse(raw)
