"""Static board preview: ``uv run python -m tools.render_board [--out FILE] [--embed]``.

The user's art (assets/map/bus_map_reworked_v2.png) is the background; we draw only pieces on
top, positioned from ``data/board.yaml`` (via ``bus.board``): line markers, building discs,
passengers, action markers, buses, score discs, the clock hand and time stones. Element ids
(``street-<i>``, ``site-<i>``, ``node-<i>``) are transparent hit overlays the Phase 2 web UI
reuses. A player strip below the art shows per-seat info. Pieces are generic.
"""

from __future__ import annotations

import argparse
import base64
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
PASSENGER = "#3A3A3A"
STONE = "#8C1D1D"
# Seat colours: colour-blind-friendly (Okabe-Ito based), P0..P4.
SEAT_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#7B3FA0")
FONT = "Helvetica, Arial, sans-serif"

WIDTH = 1991
ART = 1991  # the PNG is square
STRIP = 330  # player-panel strip below the art
HEIGHT = ART + STRIP
BG_HREF = "../assets/map/bus_map_reworked_v2.png"  # relative to docs/
BG_PATH = Path(__file__).resolve().parent.parent / "assets" / "map" / "bus_map_reworked_v2.png"
LINE_GAP = 13  # offset between lines sharing a street

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
        "Vrrooomm!": [(0, 0), (1, 2), (2, 4)],
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


# --- Overlay layers ---------------------------------------------------------------------------
ACTION_KEYS = {
    "Lines": "lines",
    "Buses": "buses",
    "Passengers": "passengers",
    "Buildings": "buildings",
    "Clock": "clock",
    "Vrrooomm!": "vrrooomm",
    "Starting Player": "starting_player",
}


def _hit_layer(board: Board) -> list[str]:
    """Transparent click targets over streets, sites and intersections."""
    out: list[str] = []
    for s in board.streets:
        d = _path(_street_points(board, s))
        out.append(
            f'<path id="street-{s.id}" class="street hit" d="{d}" fill="none" stroke="#000" '
            f'stroke-opacity="0" stroke-width="22" stroke-linejoin="round"/>'
        )
    for site in board.sites:
        out.append(
            f'<rect id="site-{site.id}" class="site hit tier-{site.tier}" x="{site.x - 24}" '
            f'y="{site.y - 24}" width="48" height="48" rx="7" fill="#000" fill-opacity="0"/>'
        )
    for n in board.nodes:
        r = 34 if n.station else 16
        out.append(
            f'<circle id="node-{n.id}" class="node hit" cx="{n.x}" cy="{n.y}" r="{r}" '
            f'fill="#000" fill-opacity="0"/>'
        )
    return out


def _piece_layer(board: Board) -> list[str]:
    out: list[str] = []
    lay = board.layout
    # bus lines: dark outline, then seat colour, offset per seat on shared streets
    lines = sample_lines(board)[: SAMPLE["seats"]]
    users: dict[int, list[int]] = defaultdict(list)
    for seat, steps in enumerate(lines):
        for sid, _ in steps:
            users[sid].append(seat)
    for seat, steps in enumerate(lines):
        col = SEAT_COLORS[seat]
        for sid, _ in steps:
            k = len(users[sid])
            d = (users[sid].index(seat) - (k - 1) / 2) * LINE_GAP
            path = _path(_offset(_street_points(board, board.streets[sid]), d))
            out.append(
                f'<g class="bus-line seat-{seat}" data-street="{sid}" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round">'
                f'<path d="{path}" stroke="{INK}" stroke-width="12"/>'
                f'<path d="{path}" stroke="{col}" stroke-width="8"/></g>'
            )
        if steps:
            fn = steps[0][1]
            ln = board.streets[steps[-1][0]].other(steps[-1][1])
            if fn == ln:  # loop: ring at the closing node
                n = board.nodes[fn]
                out.append(
                    f'<circle class="line-loop seat-{seat}" cx="{n.x}" cy="{n.y}" r="24" '
                    f'fill="none" stroke="{INK}" stroke-width="11"/>'
                    f'<circle cx="{n.x}" cy="{n.y}" r="24" fill="none" stroke="{col}" '
                    f'stroke-width="7" stroke-dasharray="9 6"/>'
                )
            else:
                for end in (fn, ln):
                    n = board.nodes[end]
                    out.append(
                        f'<rect class="line-end seat-{seat}" x="{n.x - 9}" y="{n.y - 9}" '
                        f'width="18" height="18" fill="{col}" stroke="{INK}" stroke-width="3"/>'
                    )
    # building discs on sites (the sites' own letter boxes are part of the art)
    kinds = SAMPLE["buildings"]
    for site in board.sites[: len(kinds)]:
        out.append(
            f'<g class="building"><circle cx="{site.x}" cy="{site.y}" r="25" fill="{PAPER}" '
            f'stroke="{INK}" stroke-width="3"/>{icon(kinds[site.id], site.x, site.y, 0.8)}'
        )
        if site.id in SAMPLE["passenger_buildings"]:
            out.append(passenger(site.x + 20, site.y - 14, 0.8))
        out.append("</g>")
    # waiting passengers at intersections
    step = max(1, len(board.nodes) // SAMPLE["waiting_nodes"])
    for n in board.nodes[::step][: SAMPLE["waiting_nodes"]]:
        out.append(passenger(n.x + (-26 if n.station else 0), n.y - 26, 0.9))
    # action markers: earliest empty letter first, so a is filled before b
    for name, placed in SAMPLE["markers"].items():
        spaces = lay.actions[ACTION_KEYS[name]]
        for k, seat in placed:
            x, y = spaces["abcdef"[k]]
            out.append(marker(x, y, seat, 19))
    # buses in the Buses grid: owned buses, seat by seat
    cells = iter(lay.bus_grid)
    for seat, count in enumerate(SAMPLE["buses"][: SAMPLE["seats"]]):
        for _ in range(count):
            x, y = next(cells)
            out.append(
                f'<rect x="{x - 27}" y="{y - 26}" width="54" height="52" rx="6" '
                f'fill="{PAPER}" opacity="0.9"/>{bus_glyph(x, y, 0.8, SEAT_COLORS[seat])}'
            )
    # score discs on the track
    scores: list[int] = list(SAMPLE["scores"][: SAMPLE["seats"]])
    for seat, pts in enumerate(scores):
        if pts > 0:
            sx, sy = lay.score_track[min(pts, 20) - 1]
            out.append(marker(sx - 40, sy, seat, 17))
    # clock hand
    ha = math.radians(lay.clock_angles[("house", "office", "pub")[SAMPLE["clock"]]])
    out.append(
        f'<g class="clock-hand"><line x1="{lay.clock_x}" y1="{lay.clock_y}" '
        f'x2="{_f(lay.clock_x + lay.clock_hand * math.cos(ha))}" '
        f'y2="{_f(lay.clock_y + lay.clock_hand * math.sin(ha))}" stroke="{PAPER}" '
        f'stroke-width="14" stroke-linecap="round"/>'
        f'<line x1="{lay.clock_x}" y1="{lay.clock_y}" '
        f'x2="{_f(lay.clock_x + lay.clock_hand * math.cos(ha))}" '
        f'y2="{_f(lay.clock_y + lay.clock_hand * math.sin(ha))}" stroke="{INK}" '
        f'stroke-width="8" stroke-linecap="round"/>'
        f'<circle cx="{lay.clock_x}" cy="{lay.clock_y}" r="11" fill="{INK}" stroke="{PAPER}" '
        f'stroke-width="3"/></g>'
    )
    # time stones: slots still holding a stone get a ring; the taken one is covered
    for i, (x, y) in enumerate(lay.stones):
        if i < SAMPLE["stones_left"]:
            out.append(
                f'<circle class="stone-present" cx="{x}" cy="{y}" r="70" fill="none" '
                f'stroke="{INK}" stroke-width="4" stroke-dasharray="3 7"/>'
            )
        else:
            out.append(
                f'<g class="stone-taken"><circle cx="{x}" cy="{y}" r="76" fill="{PAPER}" '
                f'opacity="0.92"/>{marker(x, y, SAMPLE["stone_holder"], 26)}</g>'
            )
    return out


def _player_strip() -> list[str]:
    y0 = ART + 12
    out: list[str] = []
    n = SAMPLE["seats"]
    w = (WIDTH - 28 - 16 * (n - 1)) / n
    for seat in range(n):
        x = 14 + seat * (w + 16)
        out.append(
            f'<g class="seat-area seat-{seat}"><rect x="{_f(x)}" y="{y0}" width="{_f(w)}" '
            f'height="{STRIP - 24}" rx="18" fill="{PAPER}" stroke="{SEAT_COLORS[seat]}" '
            f'stroke-width="8"/>'
        )
        out.append(marker(x + 40, y0 + 42, seat, 24))
        out.append(_text(x + 78, y0 + 42, f"P{seat}", 30, INK, "start"))
        out.append(_text(x + w - 16, y0 + 42, f"{SAMPLE['scores'][seat]} pts", 26, INK, "end"))
        out.append(_text(x + 20, y0 + 100, "Buses", 20, RED_DARK, "start"))
        for b in range(5):
            has = b < SAMPLE["buses"][seat]
            out.append(
                bus_glyph(x + 40 + b * 52, y0 + 142, 0.65, SEAT_COLORS[seat] if has else "#D8CFB8")
            )
        out.append(_text(x + 20, y0 + 200, "Markers left", 20, RED_DARK, "start"))
        out.append(_text(x + w - 16, y0 + 200, "25 lines, 20 actions", 17, RED_DARK, "end"))
        out.append(_text(x + 20, y0 + 245, "Stones held", 20, RED_DARK, "start"))
        held = 1 if seat == SAMPLE["stone_holder"] else 0
        out.append(_text(x + w - 16, y0 + 245, str(held), 24, INK, "end"))
        out.append("</g>")
    return out


def render(board: Board | None = None, embed: bool = False) -> str:
    board = board or load_board()
    href = BG_HREF
    if embed:
        href = "data:image/png;base64," + base64.b64encode(BG_PATH.read_bytes()).decode()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#2B2B2B"/>',
        f'<image id="board-art" href="{href}" xlink:href="{href}" x="0" y="0" '
        f'width="{WIDTH}" height="{ART}"/>',
        *_piece_layer(board),
        *_hit_layer(board),
        *_player_strip(),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/board-preview.svg")
    ap.add_argument("--embed", action="store_true", help="base64-inline the background PNG")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(embed=args.embed), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
