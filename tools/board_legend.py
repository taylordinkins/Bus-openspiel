"""Text map legend for LLM players: board topology and a key to the observer string.

    uv run python -m tools.board_legend

Built from ``data/board.yaml`` through ``bus.board`` only; no ids are hard-coded. Also provides
``summary(core)``, a read-only per-seat situational summary used by ``agent_play --summary``.
It reads state fields and calls existing engine helpers (``bus.lines``, ``bus.rules``); it adds no
rule logic.
"""

# ruff: noqa: E501
from __future__ import annotations

import sys

from bus import constants as C
from bus.actions import site_ref
from bus.board import Board
from bus.lines import legal_extensions, line_ends, line_nodes
from bus.rules import EMPTY, PHASE_END, Core, board, vrrooomm_moves

KEY = """\
-- key to the state text --
Seats are P0, P1, ... Per seat: pts=points, bus=buses owned, am=action markers left,
lm=line markers left, st=time stones held (-1 point each at game end, R-068), bld=see RULES_SPEC, line=[street ids]
in order along the line (empty = no line yet), last=move number when points last changed (R-070 tiebreak).
Header: round, phase (SB/L1/L2 setup, SEL choosing actions, LN lines, PX passengers, BD buildings,
CK clock, VP/VD Vrrooomm! pick passenger / deliver, END), cur=seat to act, clock=current building type,
stones_on_clock, supply=passengers in supply, placed=setup progress, start=start player, move=moves
so far, pass=SEL pass, dropped=seats out of SEL this round, cursor=letter of the marker being resolved,
units=Lines/Passengers/Buildings units left, moves_left=Vrrooomm! moves left, picked=node of the
passenger picked in VD (-1 none).
occupied=[site ids] sites whose building holds a passenger. sites={site id: building type} for built
sites only (others are empty). waiting={node id: passengers standing there}.
board=[[...] x7] action board; rows are lines, buses, passengers, buildings, clock, vrrooomm,
starting_player; each row lists the seats on letters a, b, c, ... in order.
Action labels name streets by id (s#), nodes as n#, sites as "site t (tier, n#)".
Details of any rule: docs/RULES_SPEC.md.
"""


def legend(b: Board | None = None) -> str:
    b = b or board()
    out = [
        f"-- Bus map: {len(b.nodes)} nodes (n#), {len(b.streets)} streets (s#), "
        f"{len(b.sites)} building sites --",
        "Lines are paths of streets; buildings stand at nodes; passengers wait at nodes.",
        "Stations (Passengers action puts passengers here): "
        + ", ".join(f"n{n}" for n in b.stations),
        "Roundabouts (one starting passenger each): " + ", ".join(f"n{n}" for n in b.roundabouts),
        "Site tiers fill in order "
        + ",".join(sorted({s.tier for s in b.sites}))
        + " (lowest open tier first).",
        "",
        "-- nodes: id (x,y) [role]: neighbours via street ids; sites at node --",
    ]
    for n in b.nodes:
        roles = [r for r, on in (("station", n.station), ("roundabout", n.roundabout)) if on]
        nb = ", ".join(f"n{b.streets[s].other(n.id)} (s{s})" for s in sorted(b.incident[n.id]))
        sites = "; ".join(site_ref(t) for t in b.node_sites[n.id]) or "none"
        tag = f" [{','.join(roles)}]" if roles else ""
        out.append(f"n{n.id} ({n.x},{n.y}){tag}: {nb} | sites: {sites}")
    out.append("")
    out.append("-- streets: id: nodeA-nodeB --")
    out.append("; ".join(f"s{s.id}: n{s.a}-n{s.b}" for s in b.streets))
    out += ["", KEY]
    return "\n".join(out)


def summary(core: Core) -> str:
    """Per-seat situational summary of ``core`` from state fields and existing helpers."""
    b = board()
    types = C.BUILDING_TYPES
    out = ["-- summary (derived from the state above) --"]
    lines = [s.line for s in core.seats]
    for seat, s in enumerate(core.seats):
        head = f"P{seat}: pts={s.points} buses={s.buses} line_markers_left={s.line_markers}"
        if not s.line:
            out.append(head + "; no line yet")
            continue
        path = line_nodes(b, s.line)
        e0, e1 = line_ends(b, s.line)
        on = set(path)
        adj = {b.streets[st].other(n) for n in on for st in b.incident[n]} - on
        out.append(
            f"{head}; line path " + "-".join(f"n{n}" for n in path) + f" (ends n{e0}, n{e1})"
        )
        blds: list[str] = []
        for n in path:
            for t in b.node_sites[n]:
                y = core.site_types[t]
                if y == EMPTY:
                    continue
                blds.append(
                    f"site {t}@n{n} {types[y]}" + (" (occupied)" if t in core.occupied else "")
                )
        out.append("   buildings on line: " + (", ".join(blds) or "none"))
        wait_on = {n: core.waiting[n] for n in path if core.waiting[n]}
        wait_adj = {n: core.waiting[n] for n in sorted(adj) if core.waiting[n]}
        out.append(
            f"   waiting on line: {wait_on or 'none'}; waiting at adjacent nodes: {wait_adj or 'none'}"
        )
        if core.phase != PHASE_END:
            ext = sorted(
                (end, st, b.streets[st].other(line_ends(b, s.line)[end]))
                for end, st in legal_extensions(b, lines, seat)
            )
            out.append(
                "   legal extensions now (end, street -> node): "
                + (", ".join(f"end{e} s{st}->n{n}" for e, st, n in ext) or "none")
            )
            out.append(f"   Vrrooomm! moves now: {vrrooomm_moves(core, seat)}")
    return "\n".join(out)


def main() -> int:
    print(legend())
    return 0


if __name__ == "__main__":
    sys.exit(main())
