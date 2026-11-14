"""Public test harness for rule tests (.claude/rules/engine.md).

Rule tests load board positions from JSON fixtures in ``tests/scenarios/`` and assert on the
fixture's ``expect`` block or on the public state API; they never build positions in test code.

The scenario module (``bus.scenario``) is imported lazily so test modules still collect before the
loader exists.

API
  scenario_module()            the scenario module (load_scenario, dump_scenario,
                               ScenarioError, check_expect)
  fixture_names()              sorted names of the ``*.json`` fixtures that carry a ``format`` key
  fixture_path(name)           Path of ``tests/scenarios/<name>.json``
  load_doc(name)               the raw JSON document (fresh copy) of a fixture
  load_fixture(name, game=None)  state at the fixture position, ``history`` applied
  load_doc_state(doc, game=None)  same for an in-memory document
  run_fixture(name)            load_fixture then check the fixture's ``expect``; returns the state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCENARIO_DIR = Path(__file__).parent / "scenarios"


def scenario_module() -> Any:
    import bus.scenario

    import bus.game  # noqa: F401  (registers python_bus)

    return bus.scenario


def fixture_path(name: str) -> Path:
    return SCENARIO_DIR / f"{name}.json"


def fixture_names() -> list[str]:
    names = []
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        doc = json.loads(p.read_text())
        if isinstance(doc, dict) and "format" in doc:
            names.append(p.stem)
    return names


def load_doc(name: str) -> dict[str, Any]:
    return json.loads(fixture_path(name).read_text())


def load_doc_state(doc: dict[str, Any], game: Any = None) -> Any:
    return scenario_module().load_scenario(doc, game)


def load_fixture(name: str, game: Any = None) -> Any:
    return load_doc_state(load_doc(name), game)


def run_fixture(name: str) -> Any:
    doc = load_doc(name)
    state = load_doc_state(doc)
    scenario_module().check_expect(state, doc.get("expect", {}))
    return state
