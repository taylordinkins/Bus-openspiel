"""Board immutability and negative parser tests (PLAN.md section 9 Phase 0 backlog).

Expected parse entry point: ``parse_board(raw: dict) -> Board`` in the board module (raw is the
dict loaded from data/board.yaml). Bad input raises ValueError. Only the current rule of exactly
20 bus-grid cells is tested (a row-per-seat change is under review for Phase 2).
"""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from bus.board import load_board

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "board.yaml"


def _raw() -> dict[str, Any]:
    return deepcopy(yaml.safe_load(RAW_PATH.read_text()))


def _parse(raw: dict[str, Any]) -> Any:
    from bus import board

    return board.parse_board(raw)


def test_board_is_hashable_and_cached() -> None:
    """BOARD-BACKLOG: hash(load_board()) works so the board can key caches."""
    b = load_board()
    assert isinstance(hash(b), int)
    assert hash(b) == hash(load_board())


def test_layout_is_immutable() -> None:
    """BOARD-BACKLOG: layout holds no plain mutable dict or list."""
    lay = load_board().layout
    assert not isinstance(lay.actions, dict)
    assert not isinstance(lay.bus_grid, list)
    assert not isinstance(lay.clock_angles, dict)
    with pytest.raises(TypeError):
        lay.actions["lines"] = {}  # type: ignore[index]


def test_parse_board_accepts_the_shipped_data() -> None:
    """BOARD-BACKLOG: parse_board on the unmodified raw dict equals load_board()."""
    assert _parse(_raw()) == load_board()


def _point(r: dict[str, Any], value: Any) -> None:
    r["layout"]["stones"][0] = value


def _frame_equal(r: dict[str, Any]) -> None:
    _point(r, [1991, 10])


def _frame_negative(r: dict[str, Any]) -> None:
    _point(r, [-1, 10])


def _frame_over(r: dict[str, Any]) -> None:
    r["layout"]["score_track"][0] = [10, 5000]


def _bad_letter(r: dict[str, Any]) -> None:
    r["layout"]["actions"]["lines"]["g"] = [10, 10]


def _missing_letter(r: dict[str, Any]) -> None:
    del r["layout"]["actions"]["lines"]["a"]


def _extra_letter_single(r: dict[str, Any]) -> None:
    r["layout"]["actions"]["buses"]["b"] = [10, 10]


def _bus_cells_19(r: dict[str, Any]) -> None:
    r["layout"]["bus_grid"].pop()


def _bus_cells_21(r: dict[str, Any]) -> None:
    r["layout"]["bus_grid"].append([10, 10])


def _no_clock_segment(r: dict[str, Any]) -> None:
    del r["layout"]["clock"]["segments"]["pub"]


def _unknown_action(r: dict[str, Any]) -> None:
    r["layout"]["actions"]["teleport"] = {"a": [10, 10]}


def _hand_zero(r: dict[str, Any]) -> None:
    r["layout"]["clock"]["hand"] = 0


def _radius_negative(r: dict[str, Any]) -> None:
    r["layout"]["clock"]["radius"] = -5


def _point_three(r: dict[str, Any]) -> None:
    _point(r, [1, 2, 3])


def _point_float(r: dict[str, Any]) -> None:
    _point(r, [1.5, 2])


MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "frame_equal_1991": _frame_equal,
    "frame_negative": _frame_negative,
    "frame_over": _frame_over,
    "letter_g": _bad_letter,
    "missing_letter": _missing_letter,
    "extra_letter_on_single": _extra_letter_single,
    "bus_cells_19": _bus_cells_19,
    "bus_cells_21": _bus_cells_21,
    "no_clock_segment": _no_clock_segment,
    "unknown_action_key": _unknown_action,
    "clock_hand_zero": _hand_zero,
    "clock_radius_negative": _radius_negative,
    "point_three_elements": _point_three,
    "point_float": _point_float,
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_parse_board_rejects_bad_layout(name: str) -> None:
    """BOARD-BACKLOG: malformed layout data raises ValueError (frame is 0..1990)."""
    raw = _raw()
    MUTATIONS[name](raw)
    with pytest.raises(ValueError):
        _parse(raw)
