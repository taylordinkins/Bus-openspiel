"""Initial state (R-001..R-006, R-009, R-014, R-015, R-073; ACTIONS §3.1, §4).

Public accessors the implementer must provide on ``bus.game.BusState`` (small surface):

  state.round            int, 0 during setup
  state.phase            str phase code (ACTIONS §3): "SB" at the start
  state.clock            str, a member of bus.constants.BUILDING_TYPES; "house" at the start
  state.stones_on_clock  int
  state.passenger_supply int
  state.waiting_at(node) int, passengers standing at node n (not in buildings)
  state.site_building(t) None when site t is empty, else the building type str
  state.action_board     sequence of 7 lists (one per action 0..6) of seats on the letters in order
  state.seat(p)          view with int attrs: points, buses, action_markers, line_markers,
                         stones (time stones taken), buildings (count built by the seat);
                         and ``line`` (list of street ids, ordered, empty initially)
"""

from typing import Any

import pyspiel
import pytest

from bus import constants as C
from bus.actions import BUILD_BASE, NUM_DISTINCT_ACTIONS, encode_build
from bus.board import load_board

B = load_board()
PLAYER_COUNTS = [3, 4, 5]


def _game(n: int = 4, **kw: Any) -> Any:
    import bus.game  # noqa: F401

    return pyspiel.load_game("python_bus", {"players": n, **kw})


def _init(n: int = 4, **kw: Any) -> Any:
    return _game(n, **kw).new_initial_state()


def _site_node(t: int) -> int:
    return next(n for n in range(len(B.nodes)) if t in B.node_sites[n])


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R014_round_and_phase_at_start(n: int) -> None:
    """R-014/R-021: the game starts in setup (round 0) at the building step SB."""
    s = _init(n)
    assert s.round == 0
    assert s.phase == "SB"
    assert not s.is_terminal()
    assert s.move_number() == 0
    assert list(s.history()) == []


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R019_seat_zero_acts_first(n: int) -> None:
    """R-019, ACTIONS §3.1: seat 0 starts and is the first to place a building."""
    assert _init(n).current_player() == 0


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R014_seat_supplies(n: int) -> None:
    """R-014, R-004: each seat has 0 points, 1 bus, 20 Action markers, 25 Bus Line markers,
    no time stones, no buildings, and an empty line."""
    s = _init(n)
    for p in range(n):
        v = s.seat(p)
        assert v.points == 0
        assert v.buses == C.INITIAL_BUSES == 1
        assert v.action_markers == C.ACTION_MARKERS_PER_PLAYER == 20
        assert v.line_markers == C.LINE_MARKERS_PER_PLAYER == 25
        assert v.stones == 0
        assert v.buildings == 0
        assert list(v.line) == []


@pytest.mark.parametrize("n,stones", [(3, 4), (4, 5), (5, 5)])
def test_R006_all_time_stones_on_clock(n: int, stones: int) -> None:
    """R-006: 5 stones on the clock, 4 with three players; none taken yet."""
    s = _init(n)
    assert s.stones_on_clock == stones == C.time_stones(n)


def test_R014_clock_points_at_house() -> None:
    """R-014: the clock starts at house."""
    assert _init().clock == "house" == C.BUILDING_TYPES[0]


def test_R014_no_sites_filled() -> None:
    """R-014: no site is filled at the start."""
    s = _init()
    assert all(s.site_building(t) is None for t in range(len(B.sites)))


def test_R014_action_board_empty() -> None:
    """R-014: the action board is empty (7 actions, no markers on any)."""
    s = _init()
    board = list(s.action_board)
    assert len(board) == 7
    assert all(len(letters) == 0 for letters in board)


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R015_four_passengers_on_roundabouts(n: int) -> None:
    """R-015: one waiting passenger on each roundabout node; 11 stay in the supply."""
    s = _init(n)
    assert len(B.roundabouts) == C.INITIAL_BOARD_PASSENGERS == 4
    for node in B.roundabouts:
        assert s.waiting_at(node) == 1
    assert s.passenger_supply == C.TOTAL_PASSENGERS - C.INITIAL_BOARD_PASSENGERS == 11


def test_R015_no_passengers_elsewhere() -> None:
    """R-015/R-005: only roundabouts hold passengers; board + supply = 15 (not inside buildings)."""
    s = _init()
    roundabouts = set(B.roundabouts)
    for node in range(len(B.nodes)):
        if node not in roundabouts:
            assert s.waiting_at(node) == 0
    on_board = sum(s.waiting_at(node) for node in range(len(B.nodes)))
    assert on_board + s.passenger_supply == C.TOTAL_PASSENGERS == 15


def test_R015_stations_start_empty() -> None:
    """R-012/R-015: stations are not roundabouts, so they hold no passengers at the start."""
    s = _init()
    for node in B.stations:
        assert s.waiting_at(node) == 0


def test_R015_roundabout_sites_stay_empty() -> None:
    """R-015: starting passengers do not enter buildings at setup (no building exists anyway)."""
    s = _init()
    for node in B.roundabouts:
        for t in B.node_sites[node]:
            assert s.site_building(t) is None


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R016_initial_legal_actions_are_build_ids(n: int) -> None:
    """ACTIONS §3.1: the first decision is SB; legal actions are non-empty BUILD ids, sorted."""
    s = _init(n)
    legal = s.legal_actions()
    assert legal
    assert legal == sorted(legal)
    lo, hi = BUILD_BASE, BUILD_BASE + 3 * len(B.sites)
    assert all(lo <= a < hi for a in legal)
    assert all(0 <= a < NUM_DISTINCT_ACTIONS for a in legal)


def test_R010_initial_builds_only_on_lowest_open_tier() -> None:
    """R-010/R-016: with nothing built, only tier-A sites are offered, every type of each."""
    s = _init()
    tier_a = [t for t, site in enumerate(B.sites) if site.tier == "A"]
    expected = sorted(encode_build(t, y) for t in tier_a for y in range(3))
    assert s.legal_actions() == expected


def test_R016_non_build_actions_not_legal_in_setup_buildings() -> None:
    """ACTIONS §3.1: in SB, SELECT / DROP_OUT / EXTEND / CLOCK ids are not legal, and a build on a
    tier-B site is not legal while tier A has empty sites."""
    s = _init()
    legal = set(s.legal_actions())
    assert not legal & set(range(0, BUILD_BASE))
    tier_b = [t for t, site in enumerate(B.sites) if site.tier == "B"]
    assert not legal & {encode_build(t, y) for t in tier_b for y in range(3)}


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R073_initial_returns_are_zero(n: int) -> None:
    """R-073/OPENSPIEL_NOTES: returns() is defined (zeros) before the game ends."""
    s = _init(n)
    assert list(s.returns()) == [0.0] * n
    assert list(s.rewards()) == [0.0] * n


def test_R073_no_chance_after_a_move() -> None:
    """R-073: no chance node anywhere; the next node is a decision node too."""
    s = _init()
    s.apply_action(s.legal_actions()[0])
    assert not s.is_chance_node()
    assert s.current_player() == 0  # seat 0 places both setup buildings in a row (R-016)
    assert s.move_number() == 1


def test_R073_observation_is_available_for_every_seat() -> None:
    """R-073: perfect information; every seat can read the public state (observation string)."""
    s = _init(3)
    strs = {s.observation_string(p) for p in range(3)}
    assert all(strs)
    assert s.get_game().observation_tensor_size() > 0


def test_clone_is_independent() -> None:
    """OPENSPIEL_NOTES: clone() copies state; applying an action to one leaves the other alone."""
    s = _init()
    before = str(s)
    c = s.clone()
    assert str(c) == before
    c.apply_action(c.legal_actions()[0])
    assert str(s) == before
    assert s.move_number() == 0
    assert c.move_number() == 1
    assert str(c) != before


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_serialize_roundtrip_initial_state(n: int) -> None:
    """OPENSPIEL_NOTES: serialize()/deserialize_state() round-trips the initial state."""
    g = _game(n)
    s = g.new_initial_state()
    r = g.deserialize_state(s.serialize())
    assert str(r) == str(s)
    assert r.current_player() == s.current_player()
    assert r.legal_actions() == s.legal_actions()
    assert list(r.history()) == list(s.history())
    assert r.stones_on_clock == s.stones_on_clock
    assert r.passenger_supply == s.passenger_supply


def test_serialize_roundtrip_after_moves_keeps_playing() -> None:
    """OPENSPIEL_NOTES: a deserialized mid-setup state plays on identically."""
    g = _game(4)
    s = g.new_initial_state()
    for _ in range(3):
        s.apply_action(s.legal_actions()[0])
    r = g.deserialize_state(s.serialize())
    assert str(r) == str(s)
    assert r.current_player() == s.current_player()
    a = s.legal_actions()
    assert r.legal_actions() == a
    s.apply_action(a[-1])
    r.apply_action(a[-1])
    assert str(r) == str(s)


def test_str_is_nonempty_and_distinguishes_player_counts() -> None:
    """CLAUDE.md: str(state) delegates to the observer string; different games print differently."""
    strings = [str(_init(n)) for n in PLAYER_COUNTS]
    assert all(strings)
    assert len(set(strings)) == 3


def test_str_changes_after_an_action() -> None:
    """CLAUDE.md: distinct states never print alike."""
    s = _init()
    before = str(s)
    s.apply_action(s.legal_actions()[0])
    assert str(s) != before


def test_str_matches_observer_string_of_a_seat() -> None:
    """CLAUDE.md: str(state) is derived from the observer, so it contains the public phase."""
    s = _init()
    assert "SB" in str(s)


@pytest.mark.parametrize("seat", [0, 1, 2])
def test_action_to_string_build_in_setup(seat: int) -> None:
    """ACTIONS §4: BUILD renders as 'P<seat>: build <type> at site <t> (<tier>, n<node>)'."""
    s = _init(3)
    t = next(i for i, site in enumerate(B.sites) if site.tier == "A")
    for y, word in enumerate(C.BUILDING_TYPES):
        node = _site_node(t)
        expected = f"P{seat}: build {word} at site {t} ({B.sites[t].tier}, n{node})"
        assert s.action_to_string(seat, encode_build(t, y)) == expected


def test_action_to_string_single_arg_uses_current_player() -> None:
    """ACTIONS §4 / pyspiel: action_to_string(action) labels with the acting seat (P0 here)."""
    s = _init(4)
    a = s.legal_actions()[0]
    assert s.action_to_string(a).startswith("P0: build ")
    assert s.action_to_string(a) == s.action_to_string(0, a)


def test_legal_actions_have_strings() -> None:
    """ACTIONS §4: every legal action renders as a non-empty '<seat>: <verb>' string."""
    s = _init()
    for a in s.legal_actions():
        assert s.action_to_string(0, a).startswith("P0: build ")
