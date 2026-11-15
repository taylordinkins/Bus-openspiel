"""OpenSpiel registration of ``python_bus`` (PLAN.md §5.2, ACTIONS.md §2/§6, OPENSPIEL_NOTES.md).

Expected API: ``import bus.game`` registers "python_bus"; ``pyspiel.load_game("python_bus",
{"players": n, "returns": r})``; invalid ``players`` / ``returns`` raise.
"""

from typing import Any

import pyspiel
import pytest

PLAYER_COUNTS = [3, 4, 5]


def _load(**params: Any) -> Any:
    import bus.game  # noqa: F401  (import registers the game)

    return pyspiel.load_game("python_bus", params) if params else pyspiel.load_game("python_bus")


def test_registration_game_type() -> None:
    """PLAN §5.2: short_name python_bus, sequential, deterministic, perfect info, terminal."""
    t = _load().get_type()
    assert t.short_name == "python_bus"
    assert t.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert t.chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC
    assert t.information == pyspiel.GameType.Information.PERFECT_INFORMATION
    assert t.reward_model == pyspiel.GameType.RewardModel.TERMINAL
    assert t.min_num_players == 3
    assert t.max_num_players == 5


def test_registration_defaults_are_four_players_win_loss() -> None:
    """R-001, R-002: defaults are players=4, returns=win_loss."""
    g = _load()
    assert g.num_players() == 4
    assert g.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    params = g.get_parameters()
    assert params.get("players", 4) == 4
    assert params.get("returns", "win_loss") == "win_loss"


def test_registration_utility_by_returns_mode() -> None:
    """PLAN §5.2: ZERO_SUM for win_loss, GENERAL_SUM for margin."""
    assert _load(returns="win_loss").get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    assert _load(returns="margin").get_type().utility == pyspiel.GameType.Utility.GENERAL_SUM


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_registration_game_info(n: int) -> None:
    """ACTIONS §2/§6, R-073: 376 actions, no chance outcomes, max_game_length = 234 * n."""
    g = _load(players=n)
    assert g.num_players() == n
    assert g.num_distinct_actions() == 376
    assert g.max_chance_outcomes() == 0
    assert g.max_game_length() == 234 * n


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_registration_win_loss_utility_bounds(n: int) -> None:
    """R-071: a sole winner gets +1, losers -1/(n-1); the sum is zero."""
    g = _load(players=n, returns="win_loss")
    assert g.max_utility() == pytest.approx(1.0)
    assert g.min_utility() == pytest.approx(-1.0 / (n - 1))
    assert g.utility_sum() == pytest.approx(0.0)


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_registration_margin_utility_bounds(n: int) -> None:
    """R-072: margin returns are clipped to [-1, 1]; bounds must cover that range."""
    g = _load(players=n, returns="margin")
    assert g.max_utility() >= 1.0
    assert g.min_utility() <= -1.0


@pytest.mark.parametrize("n", [0, 1, 2, 6, 7])
def test_registration_rejects_unsupported_player_counts(n: int) -> None:
    """R-001: only 3, 4, 5 players; other counts must raise on load."""
    with pytest.raises((ValueError, pyspiel.SpielError)):
        _load(players=n)


@pytest.mark.parametrize("r", ["", "WIN_LOSS", "score", "zero_sum"])
def test_registration_rejects_unknown_returns_mode(r: str) -> None:
    """R-002: returns must be win_loss or margin; anything else raises."""
    with pytest.raises((ValueError, pyspiel.SpielError)):
        _load(returns=r)


def test_registration_rejects_unknown_parameter() -> None:
    """OPENSPIEL_NOTES: unknown params raise SpielError."""
    with pytest.raises(pyspiel.SpielError):
        _load(start_player=2)


def test_registration_string_form_loads() -> None:
    """The game-string form with parameters works once the module is imported."""
    _load()
    g = pyspiel.load_game("python_bus(players=5,returns=margin)")
    assert g.num_players() == 5
    assert g.get_type().utility == pyspiel.GameType.Utility.GENERAL_SUM


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_R073_initial_state_is_a_decision_node_not_chance(n: int) -> None:
    """R-073: no chance nodes; the initial state is a seat-0 decision node."""
    s = _load(players=n).new_initial_state()
    assert not s.is_chance_node()
    assert not s.is_terminal()
    assert s.current_player() == 0
