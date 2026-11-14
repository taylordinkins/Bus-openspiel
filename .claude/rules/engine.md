---
paths:
  - "bus/**/*.py"
  - "tests/**/*.py"
  - "tools/**/*.py"
---

# Engine conventions

- `bus/rules.py` is pure: no pyspiel import, no I/O, no randomness. `game.py` only adapts it
  to the pyspiel `Game`/`State` API. Bus has no chance nodes (D5); never add hidden randomness.
- Keep state small and cheap to copy (tuples, small ints, frozen dataclasses). MCTS clones states
  constantly; avoid per-state dicts of lists and deep object graphs.
- Board topology comes only from `data/board.yaml` via `bus/board.py`; never hard-code node,
  street or site ids. Rule constants (piece counts, stone counts) live in `bus/constants.py`.
- Use the named constants in `bus/actions.py` for action IDs; never write numeric literals.
- Every function that implements a rule names its rule ID in the docstring (e.g. "Implements R-021").
- Rule tests target the public harness (`tests/harness.py`) and scenario JSON in `tests/scenarios/`,
  not engine internals, so they survive refactors. Name them `test_R###_<behaviour>`.
- Invariants (PLAN.md §6: piece conservation, one passenger per building, line validity, score
  monotonic, action markers never recovered) are checked in the random-simulation tests; add a new
  invariant whenever a bug slips past the existing ones.
- Players (seats) are numbered from 0 everywhere: state fields, observation strings, spec examples,
  test names and log text. Never introduce a 1-based "Player 1".
- `State.__str__` returns the observer's string, so two distinct states never print alike. Don't
  hand-write a separate summary.
- Bot and UI needs shape the engine API from the start (D9): legal actions carry enough metadata
  (`action_meta`) for click-to-act, and nothing a bot needs is only reachable through private fields.
- Fixing a bug from a playtest report: first add a failing test in `tests/regressions/`, then fix.
