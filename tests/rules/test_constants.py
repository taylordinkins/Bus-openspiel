"""Constants from RULES_SPEC section 1-2 (R-001..R-008), expected in the module ``bus.constants``.

Expected names: MIN_PLAYERS, MAX_PLAYERS, DEFAULT_PLAYERS, RETURNS_MODES, DEFAULT_RETURNS,
ACTION_MARKERS_PER_PLAYER, LINE_MARKERS_PER_PLAYER, BUSES_PER_PLAYER, INITIAL_BUSES,
TOTAL_PASSENGERS, INITIAL_BOARD_PASSENGERS, TIME_STONES_BY_PLAYERS (dict), BUILDING_TYPES,
and function ``time_stones(players) -> int``.
"""

import pytest


def _c():  # noqa: ANN202
    from bus import constants

    return constants


def test_R001_player_range_and_default() -> None:
    """R-001: players in {3,4,5}, default 4."""
    c = _c()
    assert (c.MIN_PLAYERS, c.MAX_PLAYERS, c.DEFAULT_PLAYERS) == (3, 5, 4)


def test_R001_two_and_six_players_not_supported() -> None:
    """R-001: two-player and solo variants are out of scope; range excludes 1, 2 and 6."""
    c = _c()
    valid = list(range(c.MIN_PLAYERS, c.MAX_PLAYERS + 1))
    assert valid == [3, 4, 5]


def test_R002_returns_modes() -> None:
    """R-002: returns in {win_loss, margin}, default win_loss."""
    c = _c()
    assert set(c.RETURNS_MODES) == {"win_loss", "margin"}
    assert c.DEFAULT_RETURNS == "win_loss"


def test_R004_per_player_components() -> None:
    """R-004: 20 Action markers, 25 Bus Line markers, 5 buses, 1 bus placed at setup."""
    c = _c()
    assert c.ACTION_MARKERS_PER_PLAYER == 20
    assert c.LINE_MARKERS_PER_PLAYER == 25
    assert c.BUSES_PER_PLAYER == 5
    assert c.INITIAL_BUSES == 1


def test_R005_passengers() -> None:
    """R-005: 15 passengers, 4 placed at setup and 11 in the supply."""
    c = _c()
    assert c.TOTAL_PASSENGERS == 15
    assert c.INITIAL_BOARD_PASSENGERS == 4
    assert c.TOTAL_PASSENGERS - c.INITIAL_BOARD_PASSENGERS == 11


@pytest.mark.parametrize("players,stones", [(3, 4), (4, 5), (5, 5)])
def test_R006_time_stones_by_player_count(players: int, stones: int) -> None:
    """R-006: 5 time stones, or 4 when players = 3."""
    c = _c()
    assert c.time_stones(players) == stones
    assert c.TIME_STONES_BY_PLAYERS[players] == stones


@pytest.mark.parametrize("players", [1, 2, 6])
def test_R006_time_stones_rejects_unsupported_counts(players: int) -> None:
    """R-001/R-006: an unsupported player count is not silently given a stone count."""
    c = _c()
    with pytest.raises((ValueError, KeyError)):
        c.time_stones(players)


def test_R007_building_types_in_clock_order() -> None:
    """R-007 / ACTIONS type y: house, office, pub, in that order."""
    c = _c()
    assert tuple(c.BUILDING_TYPES) == ("house", "office", "pub")
