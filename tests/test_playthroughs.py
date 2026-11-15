"""Golden playthroughs (PLAN §6): replay recorded random games; a diff names the first move."""

import json
from typing import Any

import pytest

from tools import playthrough


def _load(players: int, seed: int) -> dict[str, Any]:
    return json.loads(playthrough.path_for(players, seed).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("players", "seed"), playthrough.SEEDS)
def test_golden_playthrough(players: int, seed: int) -> None:
    diff = playthrough.check(_load(players, seed))
    assert diff is None, diff


@pytest.mark.parametrize(("players", "seed"), playthrough.SEEDS)
def test_golden_matches_fresh_recording(players: int, seed: int) -> None:
    assert playthrough.record(players, seed) == _load(players, seed)


def test_check_names_first_differing_move() -> None:
    doc = _load(*playthrough.SEEDS[0])
    doc["moves"][4]["label"] = "tampered"
    diff = playthrough.check(doc)
    assert diff is not None and diff.startswith("move 5:")
