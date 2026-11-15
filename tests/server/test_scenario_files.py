"""Named lobby scenarios live in data/scenarios/ as exact copies of tests/scenarios/ fixtures.

UI_DESIGN 6.9: the server reads only `data/`; this guards the copies against drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from bus.scenario import load_scenario
from server.adapters.bus import SCENARIOS_MANIFEST

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "scenarios"
TESTS = ROOT / "tests" / "scenarios"
MANIFEST: dict[str, Any] = yaml.safe_load(SCENARIOS_MANIFEST.read_text())
ENTRIES: list[dict[str, Any]] = MANIFEST["scenarios"]


def test_manifest_points_at_data_scenarios() -> None:
    assert (ROOT / MANIFEST["dir"]).resolve() == DATA.resolve()


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["id"] for e in ENTRIES])
def test_named_scenario_file_exists_loads_and_matches_original(entry: dict[str, Any]) -> None:
    copy = DATA / entry["file"]
    original = TESTS / entry["file"]
    assert copy.is_file(), f"{copy} missing"
    assert original.is_file(), f"{original} missing"
    assert copy.read_bytes() == original.read_bytes(), f"{copy} differs from {original}"
    doc = json.loads(copy.read_text())
    doc.pop("expect", None)
    load_scenario(doc, require_legal=True)
