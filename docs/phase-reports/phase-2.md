# Phase 2 report — Server + polished board UI

Branch `phase-2/server-ui` · closed 2026-11-15 · ✋ checkpoint: playtest #1 done (hotseat game and a game against random bots; two reports filed and triaged, user happy with the result). The user merges the PR.

## Exit criteria

| Criterion | Evidence |
|---|---|
| Server, Bus adapter, board UI, interaction layer, animation, lobby scenarios | Units 1–10, see PLAN.md Phase 2 list. Every legal action is reachable by clicking the board (unit 8; e2e finishes a 3-player game by mouse only). |
| Playwright smoke, layout, full game by clicking, CK take-stone and last-stone confirm | `cd web && npm run e2e`: **16 passed**, 6 skipped (opt-in screenshot baselines, `npm run e2e:shots`: 6 passed; advisory in CI). |
| CI web + e2e jobs | `.github/workflows/ci.yml` `web` and `e2e` jobs; reviewed by `code-reviewer`. |
| Playtest #1 | Two reports (2b0b32, a22926) triaged as a spec gap → user ruling D22 (Clock step every round), spec amended, regression tests `tests/regressions/test_report_*.py`. |
| `rules-auditor` + `code-reviewer` | code-reviewer **APPROVE** (no blockers). rules-auditor **FAIL → fixed → re-check PASS** (one must-fix: last-stone end events; 400+300 biased games, 0 event-replay mismatches). |

## Full run (before the last-stone event fix; fast suite re-run after it)

| Command | Result |
|---|---|
| `uv run ruff check .` / `ruff format --check .` / `uv run pyright` | clean / 115 files formatted / 0 errors |
| `uv run pytest -q -rxX` | **1670 passed**, 2 xfailed (marker self-tests); after the fix **1673 passed** |
| `uv run pytest -q -m slow -rxX` | **238 passed** (7:20) |
| `npm run typecheck` / `npm test` / `npm run build` / `npm run check:api` | clean / **446 passed** (22 files) / built / no API drift |
| `npm run e2e` | 16 passed, 6 skipped |

## Decisions this phase
- D22 (2026-11-15): Clock step every round; R-061 retired; `S-CLOCKSTEP` loader check; `clock_advanced` event. Spec amended (AMENDING → APPROVED). Golden `4p_seed2` regenerated: its sites-full round 5 had no Clock marker, so the automatic step now advances pub → house and moves passengers before END.
- D23: last-stone end events (see PLAN §3).

## Findings
- **Fixed:** rules-auditor must-fix (R-053/R-067) — after a last-stone pause the event stream and log showed later Vrrooomm! markers "resolving"; `round_ended` named a next starting player at game end.
- **Deferred to the PLAN.md §9 backlog (should-fix / minor):** Back works at END; `seat` validated after mutation; playback/polling edge cases; prompt-chip wording; score track cap; action-box hover order; stale `VP.lone.json`; 262 `act()` warnings; two Phase 1 items marked *before Phase 3* (`state.clone()` deep copy, R-042 lookahead memoisation).

## For the reviewer
Play one more hotseat and one bot game on the merged branch if you want a final check; the Back-at-END bug (backlog) is the one visible rough edge.
