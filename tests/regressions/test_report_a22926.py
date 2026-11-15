"""Playtest report 20261115T190834-a22926 (category engine, 3 players).

Reporter's note: the time clock should advance automatically when no stone is taken, during the
Clock resolution.  Same session and same 26-action history as report 2b0b32: nobody selected
Clock in round 1, yet the clock stayed at house through Vrrooomm!.  D22 (R-050, R-051): the Clock
step runs every round and advances automatically when there is no Clock marker.
"""

from __future__ import annotations

from typing import Any

from tests import harness as H

REPORT = "20261115T190834-a22926"
HISTORY = [
    154,
    149,
    173,
    178,
    166,
    152,
    31,
    47,
    41,
    22,
    122,
    21,
    5,
    5,
    5,
    2,
    0,
    1,
    7,
    7,
    7,
    38,
    289,
    289,
    298,
    331,
]
VRROOOMM_START = 24


def scenario(history: list[int]) -> dict[str, Any]:
    return {
        "format": 1,
        "parameters": {"players": 3, "returns": "win_loss"},
        "history": list(history),
    }


def at_vrrooomm() -> Any:
    return H.load_doc_state(scenario(HISTORY[:VRROOOMM_START]))


def test_report_a22926_clock_advanced_automatically_without_a_marker() -> None:
    """R-050, R-051: round 1, no Clock marker: the hand moved house -> office by the time
    Vrrooomm! starts, with no Clock decision having been offered (history has none)."""
    s = at_vrrooomm()
    assert s.clock == "office"


def test_report_a22926_automatic_advance_takes_no_stone() -> None:
    """R-051: the automatic advance takes no time stone (3 players: 4 on the clock, none held)."""
    s = at_vrrooomm()
    assert s.clock == "office"  # the step has run
    assert s.stones_on_clock == 4
    assert [s.seat(p).stones for p in range(3)] == [0, 0, 0]
    assert not s.is_terminal()


def test_report_a22926_passengers_moved_at_the_step_only_into_offices() -> None:
    """R-054, R-055: no office stands at n13, n17, n18, n21 or at the station n5, so after the
    step nobody is inside a building and all six passengers wait where they were."""
    s = at_vrrooomm()
    assert s.clock == "office"
    assert not any(s.is_occupied(t) for t in range(47))
    assert {n: s.waiting_at(n) for n in range(36) if s.waiting_at(n)} == {
        5: 2,
        13: 1,
        17: 1,
        18: 1,
        21: 1,
    }
