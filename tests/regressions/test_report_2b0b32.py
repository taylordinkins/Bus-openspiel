"""Playtest report 20261115T190746-2b0b32 (category engine, 3 players).

Reporter's note: before Vrrooomm! the passengers at intersections are considered to occupy the
appropriate buildings there.  History of the 26 actions (from the report's export): setup (12),
round-1 selection (P0 Vrrooomm!+Passengers, P1 Vrrooomm!+Lines, P2 Vrrooomm!+Buses, three
drop-outs), P1's Lines extension, P0 places two passengers at station n5, then P0's Vrrooomm!:
pick at n5 (id 298), deliver to site 2 = the house at n13 (id 331).  Nobody selected Clock, so the
clock stayed at house and P0 carried a passenger into the house at n13 while another passenger
waited there.  Under D22 (R-050, R-056, R-059) the Clock step runs in round 1 with no marker: the
clock reads office at Vrrooomm!, and P0 cannot deliver to a house.
"""

from __future__ import annotations

from typing import Any

import pytest

from bus.actions import DELIVER_TO_BASE
from tests import harness as H

REPORT = "20261115T190746-2b0b32"
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
PICK_N5 = 298
DELIVER_SITE_2 = 331  # the house at n13
VRROOOMM_START = 24  # actions before P0's first Vrrooomm! decision


def scenario(history: list[int]) -> dict[str, Any]:
    return {
        "format": 1,
        "parameters": {"players": 3, "returns": "win_loss"},
        "history": list(history),
    }


def test_report_2b0b32_clock_reads_office_at_vrrooomm_round_1() -> None:
    """R-050, R-056: with no Clock marker the round-1 Clock step advances house -> office before
    Vrrooomm!, so P0's first Vrrooomm! decision is taken with the clock at office."""
    s = H.load_doc_state(scenario(HISTORY[:VRROOOMM_START]))
    assert s.phase == "VP" and s.current_player() == 0
    assert s.clock == "office"


def test_report_2b0b32_p0_cannot_deliver_to_the_house_at_n13() -> None:
    """R-056, R-059: at the clock's office, P0 picks the n5 passenger and the house at n13 (site 2)
    is not a destination; the only empty office on P0's line n5-n10-n13 is s0 at n10."""
    s = H.load_doc_state(scenario(HISTORY[:VRROOOMM_START]))
    assert PICK_N5 in s.legal_actions()
    s.apply_action(PICK_N5)
    assert s.phase == "VD"
    assert DELIVER_SITE_2 not in s.legal_actions()
    assert s.legal_actions() == [DELIVER_TO_BASE + 0]  # s0, the office at n10


def test_report_2b0b32_reported_full_history_is_no_longer_legal() -> None:
    """R-059: replaying the reported 26 actions fails at the last one (deliver to site 2)."""
    with pytest.raises(H.scenario_module().ScenarioError) as e:
        H.load_doc_state(scenario(HISTORY))
    assert e.value.code == "S-HISTORY"
