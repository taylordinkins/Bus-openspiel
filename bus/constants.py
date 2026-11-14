"""Rule constants (docs/RULES_SPEC.md R-001..R-008). Piece counts live here, nowhere else."""

from __future__ import annotations

MIN_PLAYERS = 3  # R-001
MAX_PLAYERS = 5
DEFAULT_PLAYERS = 4

RETURNS_MODES = ("win_loss", "margin")  # R-002
DEFAULT_RETURNS = "win_loss"

ACTION_MARKERS_PER_PLAYER = 20  # R-004
LINE_MARKERS_PER_PLAYER = 25
BUSES_PER_PLAYER = 5
INITIAL_BUSES = 1

TOTAL_PASSENGERS = 15  # R-005
INITIAL_BOARD_PASSENGERS = 4

TIME_STONES_BY_PLAYERS: dict[int, int] = {3: 4, 4: 5, 5: 5}  # R-006

# Spaces per action, in resolution order (R-022): Lines, Buses, Passengers, Buildings, Clock,
# Vrrooomm!, Starting Player.
ACTION_SPACES = (6, 1, 6, 6, 1, 6, 1)
MANDATORY_PASSES = 2  # R-023

BUILDING_TYPES = ("house", "office", "pub")  # R-007, clock-cycle order


def time_stones(players: int) -> int:
    """Time stones for a player count (R-006)."""
    if players not in TIME_STONES_BY_PLAYERS:
        raise ValueError(f"unsupported player count {players}")
    return TIME_STONES_BY_PLAYERS[players]
