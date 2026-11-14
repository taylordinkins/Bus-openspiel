"""Static board preview: ``uv run python -m tools.render_board [--out docs/board-preview.svg]``.

Draws the map from ``data/board.yaml`` (via ``bus.board``) plus a hard-coded ``SAMPLE`` state and
the action board, as one stdlib-built SVG. Element ids (``street-<i>``, ``site-<i>``,
``node-<i>``) are the scheme the Phase 2 web UI reuses. All artwork is our own.

Palette tokens (sampled from assets/map/bus_map_reworked_v2.png; see docs/board-palette.md):
see the constants below.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from bus.board import Board, Street, load_board

# --- Palette tokens -----------------------------------------------------------------------
PAPER = "#FFF4D6"  # board background (cream)
INK = "#1A1A1A"  # icons, text on cream
RED = "#B00A0A"  # frames, streets, slots, action spaces
RED_DARK = "#7E0606"
SLOT_TEXT = "#FFF4D6"
DISTRICT_OUTER = "#E3D5F0"  # lavender outer ring
DISTRICT_MID = "#FDE3D6"  # peach middle ring
DISTRICT_CORE = "#CDE39A"  # green centre
RIVER = "#8FE3FF"
RIVER_EDGE = "#14B4F0"
PASSENGER = "#3A3A3A"
STONE = "#8C1D1D"
# Seat colours: colour-blind-friendly (Okabe-Ito based), P0..P4.
SEAT_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#7B3FA0")
FONT = "Helvetica, Arial, sans-serif"

WIDTH = 1991
MAP_BOTTOM = 1400  # the map frame ends here, the action board starts below
HEIGHT = 2800
LINE_GAP = 11  # offset between lines sharing a street

# Hard-coded sample state. Streets/nodes/sites are picked from the loaded board by index into
# the walks derived below, never by literal id.
SAMPLE: dict[str, Any] = {
    "seats": 5,
    "line_lengths": (5, 4, 3, 4, 3),  # P0, P1 (shares P0's first street), P2 (a loop), P3, P4
    "buildings": ("house", "office", "pub", "house", "pub", "office"),  # first sites by index
    "passenger_buildings": (0, 2),  # indexes into the built sites above
    "waiting_nodes": 7,  # waiting passengers at this many intersections (spread by index)
    "clock": 1,  # 0 house, 1 office, 2 pub
    "stones_left": 4,
    "stone_holder": 2,
    "scores": (7, 3, 12, 0, 5),
    "buses": (3, 2, 1, 4, 2),
    "markers": {  # action -> list of (letter index, seat)
        "Lines": [(0, 0), (1, 1), (2, 3)],
        "Buses": [(0, 4)],
        "Passengers": [(0, 2), (1, 0)],
        "Buildings": [(0, 1)],
        "Clock": [(0, 3)],
        "Vrrooomm!": [(0, 0), (1, 2), (3, 4)],
        "Starting Player": [(0, 1)],
    },
}


def _f(v: float) -> str:
    return f"{v:.1f}"


def _text(
    x: float,
    y: float,
    s: str,
    size: int,
    fill: str = INK,
    anchor: str = "middle",
    weight: str = "bold",
) -> str:
    return (
        f'<text x="{_f(x)}" y="{_f(y)}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'dominant-baseline="central">{escape(s)}</text>'
    )


def _street_points(board: Board, s: Street) -> list[tuple[float, float]]:
    a, b = board.nodes[s.a], board.nodes[s.b]
    return [(a.x, a.y), *[(float(x), float(y)) for x, y in s.via], (b.x, b.y)]


def _offset(pts: list[tuple[float, float]], d: float) -> list[tuple[float, float]]:
    """Shift a polyline sideways by d (positive = left of travel), averaging normals at bends."""
    normals: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        normals.append(((y0 - y1) / length, (x1 - x0) / length))
    out: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(pts):
        ns = normals[max(i - 1, 0) : i + 1]
        nx = sum(n[0] for n in ns) / len(ns)
        ny = sum(n[1] for n in ns) / len(ns)
        norm = math.hypot(nx, ny) or 1.0
        out.append((x + d * nx / norm, y + d * ny / norm))
    return out


def _path(pts: list[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{_f(x)} {_f(y)}" for x, y in pts)


# --- Sample lines ---------------------------------------------------------------------------
def _walk(
    board: Board,
    start: int,
    length: int,
    first: int | None = None,
    avoid: frozenset[int] = frozenset(),
) -> list[tuple[int, int]]:
    """A trail of (street, from-node) steps: no street repeated, preferring unvisited nodes."""
    used: set[int] = set()
    visited = {start}
    node = start
    steps: list[tuple[int, int]] = []
    for i in range(length):
        options = [s for s in board.incident[node] if s not in used and s not in avoid]
        if i == 0 and first is not None:
            options = [first]
        if not options:
            break
        fresh = [s for s in options if board.streets[s].other(node) not in visited]
        pick = (fresh or options)[0 if i == 0 else len(fresh or options) // 2]
        used.add(pick)
        steps.append((pick, node))
        node = board.streets[pick].other(node)
        visited.add(node)
    return steps


def _find_triangle(board: Board) -> list[tuple[int, int]]:
    for s in board.streets:
        for t in board.incident[s.b]:
            if t == s.id:
                continue
            c = board.streets[t].other(s.b)
            for u in board.incident[c]:
                if u != t and board.streets[u].other(c) == s.a:
                    return [(s.id, s.a), (t, s.b), (u, c)]
    return []


def sample_lines(board: Board) -> list[list[tuple[int, int]]]:
    lens = SAMPLE["line_lengths"]
    hubs = sorted(range(len(board.nodes)), key=lambda n: -len(board.incident[n]))
    p0 = _walk(board, hubs[0], lens[0])
    p1 = _walk(board, p0[0][1], lens[1], first=p0[0][0], avoid=frozenset(s for s, _ in p0[1:2]))
    loop = _find_triangle(board)
    p3 = _walk(board, hubs[3], lens[3])
    p4 = _walk(board, hubs[5], lens[4], avoid=frozenset(s for s, _ in p3))
    return [p0, p1, loop, p3, p4]


# --- Pieces ---------------------------------------------------------------------------------
def passenger(x: float, y: float, scale: float = 1.0) -> str:
    """Generic figure: head and rounded body."""
    r = 7 * scale
    h = 20 * scale
    w = 9 * scale
    return (
        f'<g class="passenger" fill="{PASSENGER}" stroke="{PAPER}" stroke-width="1.5">'
        f'<circle cx="{_f(x)}" cy="{_f(y - h * 0.55)}" r="{_f(r)}"/>'
        f'<path d="M {_f(x - w)} {_f(y + h * 0.5)} Q {_f(x - w)} {_f(y - h * 0.2)} {_f(x)} '
        f'{_f(y - h * 0.2)} Q {_f(x + w)} {_f(y - h * 0.2)} {_f(x + w)} {_f(y + h * 0.5)} Z"/></g>'
    )


def icon(kind: str, cx: float, cy: float, s: float = 1.0) -> str:
    """Our own house / office / pub glyphs, about 30 px wide at s=1."""
    g = f'<g class="icon-{kind}" fill="{INK}" transform="translate({_f(cx)} {_f(cy)}) scale({s})">'
    if kind == "house":
        g += (
            '<path d="M -15 -2 L 0 -16 L 15 -2 L 11 -2 L 11 14 L -11 14 L -11 -2 Z"/>'
            f'<rect x="-3" y="4" width="6" height="10" fill="{PAPER}"/>'
        )
    elif kind == "office":
        g += '<rect x="-11" y="-16" width="22" height="31"/>'
        for row in range(4):
            for col in range(2):
                g += (
                    f'<rect x="{-8 + col * 9}" y="{-12 + row * 7}" width="5" height="4" '
                    f'fill="{PAPER}"/>'
                )
    else:  # pub: a mug
        g += (
            '<rect x="-12" y="-13" width="20" height="27" rx="3"/>'
            f'<path d="M 8 -8 h 6 a 3 3 0 0 1 3 3 v 9 a 3 3 0 0 1 -3 3 h -6" fill="none" '
            f'stroke="{INK}" stroke-width="3"/>'
            f'<rect x="-9" y="-9" width="14" height="5" fill="{PAPER}"/>'
        )
    return g + "</g>"


def bus_glyph(cx: float, cy: float, s: float = 1.0, fill: str = INK) -> str:
    return (
        f'<g class="bus-glyph" transform="translate({_f(cx)} {_f(cy)}) scale({s})" fill="{fill}">'
        f'<rect x="-20" y="-14" width="40" height="26" rx="6"/>'
        f'<rect x="-15" y="-9" width="30" height="10" rx="2" fill="{PAPER}"/>'
        f'<circle cx="-11" cy="14" r="4"/><circle cx="11" cy="14" r="4"/></g>'
    )


def train_glyph(cx: float, cy: float, s: float = 1.0) -> str:
    return (
        f'<g class="train-glyph" transform="translate({_f(cx)} {_f(cy)}) scale({s})" '
        f'fill="{PAPER}">'
        f'<rect x="-20" y="-14" width="40" height="22" rx="8"/>'
        f'<rect x="-15" y="-10" width="9" height="9" fill="{RED}"/>'
        f'<rect x="-4" y="-10" width="9" height="9" fill="{RED}"/>'
        f'<rect x="7" y="-10" width="9" height="9" fill="{RED}"/>'
        f'<path d="M -16 14 L -22 20 M 16 14 L 22 20" stroke="{PAPER}" stroke-width="3"/>'
        f'<rect x="-14" y="10" width="28" height="4"/></g>'
    )


def stone(cx: float, cy: float, r: float = 38) -> str:
    return (
        f'<g class="time-stone"><circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{STONE}"/>'
        f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r * 0.72)}" fill="{PAPER}"/>'
        f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r * 0.58)}" fill="{STONE}"/>'
        f"{_text(cx, cy, 'stone', 13, PAPER)}</g>"
    )


def marker(cx: float, cy: float, seat: int, r: float = 17) -> str:
    return (
        f'<circle class="marker seat-{seat}" cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
        f'fill="{SEAT_COLORS[seat]}" stroke="{INK}" stroke-width="2.5"/>'
        f"{_text(cx, cy + 1, f'P{seat}', 14, '#FFFFFF')}"
    )


# --- Map --------------------------------------------------------------------------------------
def _map_layer(board: Board) -> list[str]:
    out: list[str] = []
    out.append(
        f'<rect x="14" y="14" width="{WIDTH - 28}" height="{MAP_BOTTOM - 28}" rx="30" '
        f'fill="{PAPER}" stroke="{RED}" stroke-width="12"/>'
    )
    xs = [n.x for n in board.nodes]
    ys = [n.y for n in board.nodes]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out.append(
        f'<clipPath id="map-clip"><rect x="20" y="20" width="{WIDTH - 40}" '
        f'height="{MAP_BOTTOM - 40}" rx="26"/></clipPath><g clip-path="url(#map-clip)">'
    )
    for scale, color in ((1.12, DISTRICT_OUTER), (0.9, DISTRICT_MID), (0.56, DISTRICT_CORE)):
        hw, hh = (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
        c = min(hw, hh) * 0.28
        out.append(
            f'<polygon class="district" fill="{color}" points="'
            f"{_f(cx - hw + c)},{_f(cy - hh)} {_f(cx + hw - c)},{_f(cy - hh)} "
            f"{_f(cx + hw)},{_f(cy - hh + c)} {_f(cx + hw)},{_f(cy + hh - c)} "
            f"{_f(cx + hw - c)},{_f(cy + hh)} {_f(cx - hw + c)},{_f(cy + hh)} "
            f'{_f(cx - hw)},{_f(cy + hh - c)} {_f(cx - hw)},{_f(cy - hh + c)}"/>'
        )
    # river, generic decoration (not derived from the map art)
    out.append(
        f'<path d="M {x0 - 150} {_f(cy - 60)} L {_f(cx - 200)} {_f(cy - 60)} L {_f(cx - 200)} '
        f"{_f(cy + 130)} L {_f(cx + 40)} {_f(cy + 130)} L {_f(cx + 300)} {_f(cy - 130)} "
        f'L {x1 + 60} {_f(cy - 130)}" fill="none" stroke="{RIVER_EDGE}" stroke-width="46" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="0.8"/>'
        f'<path d="M {x0 - 150} {_f(cy - 60)} L {_f(cx - 200)} {_f(cy - 60)} L {_f(cx - 200)} '
        f"{_f(cy + 130)} L {_f(cx + 40)} {_f(cy + 130)} L {_f(cx + 300)} {_f(cy - 130)} "
        f'L {x1 + 60} {_f(cy - 130)}" fill="none" stroke="{RIVER}" stroke-width="36" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="0.8"/>'
    )
    out.append("</g>")
    # streets
    for s in board.streets:
        d = _path(_street_points(board, s))
        out.append(
            f'<g id="street-{s.id}" class="street"><path d="{d}" fill="none" stroke="{RED}" '
            f'stroke-width="9" stroke-linejoin="round"/><path d="{d}" fill="none" '
            f'stroke="{PAPER}" stroke-width="4" stroke-linejoin="round"/></g>'
        )
    # bus lines
    lines = sample_lines(board)[: SAMPLE["seats"]]
    users: dict[int, list[int]] = defaultdict(list)
    for seat, steps in enumerate(lines):
        for sid, _ in steps:
            users[sid].append(seat)
    for seat, steps in enumerate(lines):
        col = SEAT_COLORS[seat]
        for sid, _ in steps:
            s = board.streets[sid]
            pts = _street_points(board, s)
            k = len(users[sid])
            d = (users[sid].index(seat) - (k - 1) / 2) * LINE_GAP
            off = _offset(pts, d)
            out.append(
                f'<path class="bus-line seat-{seat}" data-street="{sid}" d="{_path(off)}" '
                f'fill="none" stroke="{col}" stroke-width="7" stroke-linecap="round" '
                f'stroke-linejoin="round" opacity="0.95"/>'
            )
        if steps:
            first, last = steps[0], steps[-1]
            fn = first[1]
            ln = board.streets[last[0]].other(last[1])
            if fn == ln:  # loop: draw a ring marker at the closing node
                n = board.nodes[fn]
                out.append(
                    f'<circle class="line-loop seat-{seat}" cx="{n.x}" cy="{n.y}" r="22" '
                    f'fill="none" stroke="{col}" stroke-width="5" stroke-dasharray="6 5"/>'
                )
            else:
                for end in (fn, ln):
                    n = board.nodes[end]
                    out.append(
                        f'<rect class="line-end seat-{seat}" x="{n.x - 8}" y="{n.y - 8}" '
                        f'width="16" height="16" fill="{col}" stroke="{INK}" stroke-width="2"/>'
                    )
    # nodes
    for n in board.nodes:
        if n.station:
            out.append(
                f'<g id="node-{n.id}" class="node station"><rect x="{n.x - 33}" y="{n.y - 33}" '
                f'width="66" height="66" rx="8" fill="{RED}" stroke="{PAPER}" stroke-width="3"/>'
                f"{train_glyph(n.x, n.y, 0.95)}</g>"
            )
        elif n.roundabout:
            out.append(
                f'<g id="node-{n.id}" class="node roundabout">'
                f'<circle cx="{n.x}" cy="{n.y}" r="17" fill="{PAPER}" stroke="{RED}" '
                f'stroke-width="5"/><circle cx="{n.x}" cy="{n.y}" r="8" fill="{PAPER}" '
                f'stroke="{RED}" stroke-width="4"/></g>'
            )
        else:
            out.append(
                f'<circle id="node-{n.id}" class="node" cx="{n.x}" cy="{n.y}" r="9" '
                f'fill="{PAPER}" stroke="{RED}" stroke-width="4"/>'
            )
    # sites
    kinds = SAMPLE["buildings"]
    for site in board.sites:
        n = board.nodes[site.node]
        out.append(
            f'<line class="site-tick" x1="{n.x}" y1="{n.y}" x2="{site.x}" y2="{site.y}" '
            f'stroke="{RED}" stroke-width="2" opacity="0.35"/>'
        )
        built = site.id < len(kinds)
        body = (
            f'<rect x="{site.x - 24}" y="{site.y - 24}" width="48" height="48" rx="7" '
            f'fill="{PAPER if built else RED}" stroke="{RED}" stroke-width="4"/>'
        )
        if built:
            body += icon(kinds[site.id], site.x, site.y + 2, 0.95)
            body += _text(site.x - 15, site.y - 17, site.tier, 12, RED)
            if site.id in SAMPLE["passenger_buildings"]:
                body += passenger(site.x + 14, site.y + 6, 0.75)
        else:
            body += _text(site.x, site.y + 2, site.tier, 34, SLOT_TEXT)
        out.append(f'<g id="site-{site.id}" class="site tier-{site.tier}">{body}</g>')
    # waiting passengers at intersections
    step = max(1, len(board.nodes) // SAMPLE["waiting_nodes"])
    for n in board.nodes[::step][: SAMPLE["waiting_nodes"]]:
        if n.station or n.roundabout:
            out.append(passenger(n.x - 22 if n.station else n.x, n.y - 30, 0.9))
        else:
            out.append(passenger(n.x, n.y - 24, 0.9))
    # score track (right edge, as on the printed map)
    tx, ty0, ty1 = WIDTH - 70, 90, MAP_BOTTOM - 90
    out.append(f'<line x1="{tx}" y1="{ty0}" x2="{tx}" y2="{ty1}" stroke="{RED}" stroke-width="6"/>')
    steps_n = 20
    for i in range(steps_n):
        y = ty0 + (ty1 - ty0) * i / (steps_n - 1)
        out.append(
            f'<rect x="{tx - 22}" y="{_f(y - 20)}" width="44" height="40" rx="7" fill="{RED}"/>'
            f"{_text(tx, y + 1, str(i + 1), 24, SLOT_TEXT)}"
        )
    for seat, pts in enumerate(SAMPLE["scores"][: SAMPLE["seats"]]):
        if pts <= 0:
            continue
        y = ty0 + (ty1 - ty0) * (min(pts, steps_n) - 1) / (steps_n - 1)
        out.append(marker(tx - 48 - 20 * seat, y, seat, 14))
    out.append(_text(tx, ty0 - 38, "Score", 22, RED))
    return out


# --- Action board and player areas ------------------------------------------------------------
def _letters(
    x: float,
    y: float,
    markers: list[tuple[int, int]],
    hints: list[str] | None,
    count: int = 6,
    gap: int = 78,
) -> list[str]:
    out: list[str] = []
    for k in range(count):
        px = x + k * gap
        out.append(
            f'<rect x="{_f(px - 27)}" y="{_f(y - 27)}" width="54" height="54" rx="8" '
            f'fill="{RED}"/>{_text(px, y + 1, "abcdef"[k], 36, SLOT_TEXT, weight="normal")}'
        )
        if hints:
            out.append(_text(px, y + 46, hints[k], 18, RED_DARK))
    for k, seat in markers:
        out.append(marker(x + k * gap, y, seat, 21))
    return out


def _panel(x: float, y: float, w: float, h: float, num: int, title: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{RED}" '
        f'stroke-width="5"/>',
        f'<rect x="{x + 28}" y="{y + 22}" width="44" height="44" rx="7" fill="{INK}"/>',
        _text(x + 50, y + 45, str(num), 32, SLOT_TEXT),
        _text(x + 88, y + 45, title, 34, INK, "start"),
    ]


def _action_board() -> list[str]:
    top = MAP_BOTTOM + 24
    out = [
        f'<rect x="14" y="{top}" width="{WIDTH - 28}" height="1020" rx="30" fill="{PAPER}" '
        f'stroke="{RED}" stroke-width="12"/>'
    ]
    m = SAMPLE["markers"]
    five = SAMPLE["seats"] == 5
    dwindle = ["M", "M-1", "M-2", "M-3", "M-4", "M-5"]
    lines_hint = [h + "+1" if five else h for h in dwindle]
    colx = (28, 560, 1090, 1620)
    mid = top + 500
    # 1 Lines
    out += _panel(colx[0], top + 20, 520, 470, 1, "Lines")
    out += _letters(colx[0] + 50, top + 190, m["Lines"], lines_hint)
    out.append(_text(colx[0] + 260, top + 290, "power = bus maximum M, dwindling", 20, RED_DARK))
    out.append(_text(colx[0] + 260, top + 320, "5 players: +1 on Lines", 20, RED_DARK))
    # 4 Buildings
    out += _panel(colx[0], mid, 520, 480, 4, "Buildings")
    for i, kind in enumerate(("office", "pub", "house")):
        out.append(icon(kind, colx[0] + 80 + i * 90, mid + 130, 1.5))
    out += _letters(colx[0] + 50, mid + 260, m["Buildings"], dwindle)
    # 2 Buses
    out += _panel(colx[1], top + 20, 520, 470, 2, "Buses")
    for r in range(4):
        for c in range(5):
            out.append(bus_glyph(colx[1] + 200 + c * 62, top + 120 + r * 58, 0.6))
    out += _letters(colx[1] + 50, top + 200, m["Buses"], None, 1)
    out.append(_text(colx[1] + 100, top + 320, "1 space", 20, RED_DARK))
    # 5 Clock
    out += _panel(colx[1], mid, 520, 480, 5, "Clock")
    ccx, ccy, cr = colx[1] + 340, mid + 270, 120
    out.append(
        f'<circle cx="{ccx}" cy="{ccy}" r="{cr}" fill="{RED}" stroke="{RED_DARK}" '
        f'stroke-width="4"/>'
    )
    for i, kind in enumerate(("house", "office", "pub")):
        a = math.radians(-90 + i * 120 + 60)
        out.append(
            f'<path d="M {ccx} {ccy} L {_f(ccx + cr * math.cos(a - 1.047))} '
            f'{_f(ccy + cr * math.sin(a - 1.047))}" stroke="{PAPER}" stroke-width="4"/>'
        )
        px, py = ccx + 68 * math.cos(a), ccy + 68 * math.sin(a)
        out.append(
            f'<circle cx="{_f(px)}" cy="{_f(py)}" r="34" fill="{PAPER}"/>{icon(kind, px, py, 0.9)}'
        )
    ha = math.radians(-90 + SAMPLE["clock"] * 120 + 60)
    out.append(
        f'<path class="clock-hand" d="M {ccx} {ccy} L {_f(ccx + 44 * math.cos(ha))} '
        f'{_f(ccy + 44 * math.sin(ha))}" stroke="{INK}" stroke-width="8" stroke-linecap="round"/>'
        f'<circle cx="{ccx}" cy="{ccy}" r="10" fill="{INK}"/>'
    )
    out.append(_text(colx[1] + 60, mid + 190, "house", 24, RED_DARK, "start"))
    out.append(_text(colx[1] + 60, mid + 220, "> office", 24, RED_DARK, "start"))
    out.append(_text(colx[1] + 60, mid + 250, "> pub", 24, RED_DARK, "start"))
    out += _letters(colx[1] + 50, mid + 400, m["Clock"], None, 1)
    # 3 Passengers
    out += _panel(colx[2], top + 20, 510, 320, 3, "Passengers")
    out.append(train_glyph(colx[2] + 66, top + 150, 1.2).replace(PAPER, INK, 1))
    out += _letters(colx[2] + 140, top + 150, m["Passengers"], dwindle, 6, 68)
    # 6 Vrrooomm!
    out += _panel(colx[2], top + 340, 510, 400, 6, "Vrrooomm!!!")
    out += _letters(colx[2] + 50, top + 480, m["Vrrooomm!"], None)
    out.append(_text(colx[2] + 240, top + 580, "not dwindling: all your buses", 20, RED_DARK))
    # 7 Starting Player
    out += _panel(colx[2], top + 740, 510, 270, 7, "Starting Player")
    out += _letters(colx[2] + 50, top + 880, m["Starting Player"], None, 1)
    # stones
    out.append(
        f'<rect x="{colx[3]}" y="{top + 20}" width="340" height="990" fill="none" stroke="{RED}" '
        f'stroke-width="5"/>'
    )
    out.append(_text(colx[3] + 170, top + 60, "Clock stones", 30, RED))
    total = 5
    for i in range(total):
        x = colx[3] + 100 + (i % 2) * 160
        y = top + 170 + (i // 2) * 170
        if i < SAMPLE["stones_left"]:
            out.append(stone(x, y))
        else:
            out.append(
                f'<circle cx="{x}" cy="{y}" r="38" fill="none" stroke="{RED}" stroke-width="3" '
                f'stroke-dasharray="6 6"/>'
            )
    return out


def _player_areas() -> list[str]:
    y0 = MAP_BOTTOM + 24 + 1020 + 30
    out: list[str] = []
    n = SAMPLE["seats"]
    w = (WIDTH - 28 - 20 * (n - 1)) / n
    for seat in range(n):
        x = 14 + seat * (w + 20)
        out.append(
            f'<g class="seat-area seat-{seat}"><rect x="{_f(x)}" y="{y0}" width="{_f(w)}" '
            f'height="{HEIGHT - y0 - 20}" rx="18" fill="{PAPER}" stroke="{SEAT_COLORS[seat]}" '
            f'stroke-width="8"/>'
        )
        out.append(marker(x + 38, y0 + 40, seat, 24))
        out.append(_text(x + 76, y0 + 40, f"P{seat}", 30, INK, "start"))
        out.append(_text(x + w - 14, y0 + 40, f"{SAMPLE['scores'][seat]} pts", 24, INK, "end"))
        out.append(_text(x + 20, y0 + 92, "Buses", 20, RED_DARK, "start"))
        for b in range(5):
            has = b < SAMPLE["buses"][seat]
            out.append(
                bus_glyph(x + 40 + b * 52, y0 + 135, 0.65, SEAT_COLORS[seat] if has else "#D8CFB8")
            )
        if seat == SAMPLE["stone_holder"]:
            out.append(stone(x + w - 52, y0 + 135, 28))
        out.append(_text(x + 20, y0 + 195, "Markers left", 20, RED_DARK, "start"))
        out.append(_text(x + w - 14, y0 + 195, "25 lines, 20 actions", 17, RED_DARK, "end"))
        out.append("</g>")
    return out


def render(board: Board | None = None) -> str:
    board = board or load_board()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#2B2B2B"/>',
        *_map_layer(board),
        *_action_board(),
        *_player_areas(),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/board-preview.svg")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
