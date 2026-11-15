# Bus — OpenSpiel

Python OpenSpiel game (`bus/`) of *Bus* (Splotter, Complete Edition rules) on the reworked map
(`assets/map/`), with a FastAPI server (`server/`) and React/TS web UI (`web/`) ported from
`../lignum` in Phase 2. Plan and phases: `PLAN.md` (read only the current phase and §3–§4 as
needed). Rules contract: `docs/RULES_SPEC.md` (from Phase 0). OpenSpiel gotchas:
`docs/OPENSPIEL_NOTES.md`. Default branch is `master`; work happens on `phase-N/<slug>` branches
merged by PR.

## Commands
- Install: `uv sync`
- Fast tests: `uv run pytest -q` · one rule: `uv run pytest -q -k R012` · slow (hooks, sims): `uv run pytest -q -m slow`
- Lint/format/types: `uv run ruff check .` · `uv run ruff format .` · `uv run pyright`
- From Phase 1: `uv run python -m tools.play_text --players 4 --seed 7 [--human 0]`,
  `uv run python -m tools.replay <file.json> [--steps]`
- From Phase 2: `tools/dev.sh` starts API (:8000) and web (:5173) together, Ctrl-C stops both.
  Separately: server `uv run uvicorn server.app:app --reload`; web `cd web && npm ci && npm run dev|test|typecheck|build|e2e`

## Model and agent policy
The main session orchestrates; it delegates rather than writing large changes itself.
- Workhorse (Sonnet): `implementer` writes code, `spec-test-writer` writes tests from the spec.
- Cheap (Haiku): `Explore` for searching, `sim-runner` for long test/sim runs and their output.
- QA (Opus): `rules-auditor` checks spec and board data conformance, `code-reviewer` checks for bugs.
  Use them at phase close, and as spot audits for units touching line expansion, resolution order
  and action power, Vrrooomm! scoring, or game end.
- Workflow skills: `/start-phase N`, `/next-task`, `/close-phase`, `/triage-report <file>`.

## Rules
- IMPORTANT: no rule behaviour without a `docs/RULES_SPEC.md` entry. Ambiguity → stop, add it to
  PLAN.md §9, ask the user. Never resolve a rules question by guessing.
- Rules sources, in priority order: the rulebook `third_party/rules/bus_rulebook.pdf` (local only,
  git-ignored; a fresh clone lacks it), then PLAN.md §3–§4. Paraphrase it; never paste it.
- Once the spec's Status is APPROVED it is locked (a hook enforces this). Propose amendments to the user.
- Board topology lives only in `data/board.yaml` (loaded by `bus/board.py`); code never hard-codes
  node, street or site ids.
- Players (seats) are numbered from 0 everywhere: engine, tests, spec examples, logs, server, UI
  labels (`P0`…`P4`). Never show "Player 1" for seat 0.
- `State.__str__` delegates to the observer's string, so distinct states never print alike.
- Tests written ahead of code carry `@pytest.mark.pending("R-012")` (rules) or `@pytest.mark.regression("<report id>")` (playtest bugs, strict xfail); remove the marker when the rule or fix lands.
- Never edit a test to make it pass, and never regenerate `tests/playthroughs/` golden files without
  saying why in the commit message.
- Action IDs are frozen once Phase 1 closes: replays and playtest reports depend on them.
- The Stop-hook check is off (too slow every reply). Before committing run `uv run ruff check .`,
  `uv run pyright` and `uv run pytest -q` (or `python3 .claude/hooks/check.py`); don't commit red.
- UI quality matters (D9): the board follows the reworked map's layout and palette, with our own
  house/office/pub icons and generic pieces. Every legal action must be reachable by clicking the board.
- Commit at each green step with a message that names the rule IDs or task it covers.
