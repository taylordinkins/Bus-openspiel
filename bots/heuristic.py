"""Greedy one-ply heuristic bot (PLAN 5.6).

Every legal action is scored from a few hand features and the best one is played (ties broken by
the caller's rng). All numbers live in `Weights` so Phase 3 tuning can vary them.

Features:
- Lines (`extend`): value of the node the line reaches (waiting passengers, buildings and empty
  sites there, stations), fresh nodes over loops, open streets left at the node (not being cut
  off), and penalties for streets or nodes opponents already use (contestation). A first setup
  marker scores both end nodes of its street.
- Buildings (`build`): sites on or next to the bot's own line, the building type that matches the
  clock's next types for waiting passengers; setup buildings prefer well connected nodes.
- Passengers (`passenger_to`): stations on the bot's line, away from opponents' lines.
- Action selection (`select`, `drop_out`): each action space is valued by need (short line, few
  buses, passengers waiting on the line with empty buildings to receive them) times its dwindling
  power at the space the marker would take.
- Vrrooomm! (`pick_passenger`, `deliver_to`): take passengers opponents could also reach, deliver
  to nodes opponents do not use. Clock: advance, unless a pause ends the game with the bot ahead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import pyspiel

from bus import constants as C
from bus import rules
from bus.board import Board, Street
from bus.game import BusState, action_meta
from bus.lines import line_nodes

EMPTY = rules.EMPTY


@dataclass(frozen=True)
class Weights:
    # extend
    ext_passenger: float = 3.0  # per waiting passenger at the new node
    ext_site: float = 0.6  # per site (building or empty) at the new node
    ext_station: float = 1.0
    ext_open: float = 0.3  # per street at the new node not used by any line
    ext_loop: float = -3.0  # new node already on own line
    ext_shared_street: float = -2.5  # street already in an opponent's line
    ext_opp_node: float = -0.8  # new node already on an opponent's line
    setup_first_street: float = 1.0  # multiplier on the end-node value of a first marker
    # build
    build_on_line: float = 3.0
    build_near_line: float = 1.0
    build_passenger_here: float = 1.5  # per waiting passenger at the node
    build_type_match: float = 1.0  # type that the clock reaches next
    build_setup_degree: float = 0.5  # setup: per street at the node
    build_setup_sites: float = 0.3  # setup: per site at the node
    build_opp_line: float = -1.0  # per opponent line through the node
    # passenger_to
    pax_on_line: float = 3.0
    pax_opp_line: float = -1.0
    # select
    sel_lines: float = 2.2
    sel_buses: float = 3.0
    sel_passengers: float = 1.0
    sel_buildings: float = 1.2
    sel_clock: float = 0.4
    sel_vroom: float = 3.5  # per expected delivery
    sel_vroom_base: float = 0.8
    sel_start: float = 0.5
    drop_out: float = 0.3
    line_len_cap: float = 22.0  # lines value fades towards this many streets
    # vrrooomm
    pick_contested: float = 0.5  # per opponent line through the node
    pick_waiting: float = 0.1
    deliver_exclusive: float = 0.6  # per opponent line avoided
    deliver_future: float = 0.2  # per empty site left at the node
    # clock
    clock_pause: float = -5.0
    clock_pause_win: float = 20.0


DEFAULT_WEIGHTS = Weights()


class HeuristicBot:
    def __init__(self, weights: Weights = DEFAULT_WEIGHTS) -> None:
        self.w = weights

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        st = cast(BusState, state)
        ctx = _Ctx(st, self.w)
        best: list[int] = []
        best_score = -1e18
        for a in legal:
            sc = ctx.score(a)
            if sc > best_score + 1e-9:
                best_score, best = sc, [a]
            elif sc >= best_score - 1e-9:
                best.append(a)
        return best[0] if len(best) == 1 else rng.choice(best)


class _Ctx:
    """Per-decision cache of board facts."""

    def __init__(self, st: BusState, w: Weights) -> None:
        self.st, self.w = st, w
        self.core = st.core
        self.b: Board = rules.board()
        self.me = self.core.current
        seats = self.core.seats
        self.my_nodes: frozenset[int] = self._nodes(seats[self.me].line)
        self.opp_street_lines: set[int] = set()
        self.opp_count: dict[int, int] = {}  # node -> opponent lines through it
        self.used_streets: set[int] = set()
        for i, s in enumerate(seats):
            self.used_streets.update(s.line)
            if i == self.me:
                continue
            self.opp_street_lines.update(s.line)
            for n in self._nodes(s.line):
                self.opp_count[n] = self.opp_count.get(n, 0) + 1

    def _nodes(self, line: tuple[int, ...]) -> frozenset[int]:
        return frozenset(line_nodes(self.b, line)) if line else frozenset()

    def empty_sites(self, node: int) -> int:
        types = self.core.site_types
        return sum(1 for t in self.b.node_sites[node] if types[t] == EMPTY)

    def node_value(self, n: int) -> float:
        w = self.w
        v = w.ext_passenger * self.core.waiting[n]
        v += w.ext_site * len(self.b.node_sites[n])
        v += w.ext_station * (1.0 if self.b.nodes[int(n)].station else 0.0)
        v += w.ext_open * sum(1 for s in self.b.incident[n] if s not in self.used_streets)
        return v

    def score(self, a: int) -> float:
        m: dict[str, Any] = action_meta(self.st, a)
        return getattr(self, "_" + m["kind"])(m)

    # --- line / building / passengers ---
    def _extend(self, m: dict[str, Any]) -> float:
        w = self.w
        s: Street = self.b.streets[int(m["street"])]
        sc = w.ext_shared_street if m["street"] in self.opp_street_lines else 0.0
        if m["setup"] == "first":
            return sc + w.setup_first_street * (self.node_value(s.a) + self.node_value(s.b))
        to = m["to_node"]
        if to in self.my_nodes:
            return sc + w.ext_loop
        return sc + self.node_value(to) + w.ext_opp_node * self.opp_count.get(to, 0)

    def _build(self, m: dict[str, Any]) -> float:
        w = self.w
        n = int(m["node"])
        y = C.BUILDING_TYPES.index(m["building_type"])
        sc = w.build_opp_line * self.opp_count.get(n, 0)
        if m["setup"]:
            sc += w.build_setup_degree * len(self.b.incident[n])
            sc += w.build_setup_sites * len(self.b.node_sites[n])
            if self.b.nodes[int(n)].station:
                sc += 1.0
            return sc
        if n in self.my_nodes:
            sc += w.build_on_line
        elif any(self.b.streets[s].other(n) in self.my_nodes for s in self.b.incident[n]):
            sc += w.build_near_line
        sc += w.build_passenger_here * self.core.waiting[n]
        nxt = (self.core.clock + 1) % len(C.BUILDING_TYPES)
        if y == nxt:
            sc += w.build_type_match
        return sc

    def _passenger_to(self, m: dict[str, Any]) -> float:
        n = int(m["node"])
        sc = self.w.pax_opp_line * self.opp_count.get(n, 0)
        if n in self.my_nodes:
            sc += self.w.pax_on_line
        return sc

    # --- action selection ---
    def _vroom_value(self) -> float:
        core = self.core
        waiting = sum(core.waiting[n] for n in self.my_nodes)
        free = sum(self.empty_sites(n) for n in self.my_nodes)
        return float(min(core.seats[self.me].buses, waiting, free))

    def _select(self, m: dict[str, Any]) -> float:
        w, core = self.w, self.core
        me = core.seats[self.me]
        a, space = m["action"], m["space"]
        mx = rules.bus_max(core)
        power = (mx - space) if space is not None else 0
        if a == 0 and core.players == 5:
            power += 1
        power = max(power, 0)
        if a == 0:
            fade = max(0.25, 1.0 - len(me.line) / w.line_len_cap)
            return w.sel_lines * power * fade
        if a == 1:
            return w.sel_buses if me.buses < 5 else 0.0
        if a == 2:
            on_station = any(n in self.my_nodes for n in self.b.stations)
            return w.sel_passengers * min(power, core.supply) * (1.0 if on_station else 0.4)
        if a == 3:
            empty = sum(1 for t in core.site_types if t == EMPTY)
            return w.sel_buildings * min(power, empty)
        if a == 4:
            return w.sel_clock
        if a == 5:
            return w.sel_vroom_base + w.sel_vroom * self._vroom_value()
        return w.sel_start

    def _drop_out(self, m: dict[str, Any]) -> float:
        return self.w.drop_out

    # --- vrrooomm / clock ---
    def _pick_passenger(self, m: dict[str, Any]) -> float:
        n = int(m["node"])
        return self.w.pick_contested * self.opp_count.get(n, 0) + self.w.pick_waiting * m["waiting"]

    def _deliver_to(self, m: dict[str, Any]) -> float:
        n = int(m["node"])
        return -self.w.deliver_exclusive * self.opp_count.get(n, 0) + (
            self.w.deliver_future * (self.empty_sites(n) - 1)
        )

    def _clock_advance(self, m: dict[str, Any]) -> float:
        return 0.0

    def _clock_pause(self, m: dict[str, Any]) -> float:
        w, core = self.w, self.core
        if m["ends_game"]:
            final = [s.points - s.stones for s in core.seats]
            mine = final[self.me] - 1
            if all(mine > f for i, f in enumerate(final) if i != self.me):
                return w.clock_pause_win
        return w.clock_pause
