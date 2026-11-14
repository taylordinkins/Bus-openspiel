# Phase 0 report — Scaffold, board data, rules spec, board look

Branch `phase-0/rules-spec` · closed 2026-11-14 · ✋ checkpoint passed: the user approved the spec and the preview.

## Exit criteria

| Criterion | Evidence |
|---|---|
| Spec APPROVED | `docs/RULES_SPEC.md` R-001–R-073, Status APPROVED by the user (78694cf). No open questions remain; PLAN §9 Q1–Q3, Q-A, Q-B and spec O-1–O-4 are all resolved by the user. |
| Board data confirmed | `data/board.yaml` has 36 nodes, 70 streets, 47 sites (A 12, B 11, C 9, D 15), 2 stations and 4 roundabouts. The rules-auditor board audit **PASS**: every street, site letter/owner and node coordinate was checked against the map (within 1 px). The 28–35 bend was corrected (0fa41db). |
| Preview look agreed | `docs/board-preview.svg` from `tools/render_board.py`: pieces drawn over the user's map art (D15), with player panels themed after the printed board. The user signed off. Output is reproducible (byte-identical re-render). |

## Full run (final HEAD)

| Command | Result |
|---|---|
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 42 files already formatted |
| `uv run pyright` | 0 errors, 0 warnings |
| `uv run pytest -q -rxX` | 43 passed, 2 xfailed (the marker self-tests R-000 and report-000, which fail on purpose) |
| `uv run pytest -q -m slow -rxX` | 228 passed |
| `tools.render_board` re-render vs committed SVG | identical |

## Units

1. Scaffold (5968fc4) · 2. Board digitized (5968fc4, 3f435ca) · 3. Rules spec (f8b36d6) · 4. `ACTIONS.md` (376 action ids) and `SCENARIOS.md` (4ed5d3c) · 5. Spec audit (FAIL, fixed in 4ed05b6) and board audit (PASS, 0fa41db) · 6. Board preview (968184e, then reworked onto the user's art in 9df4b5c and 1d2123e) · 7. User sign-off (78694cf).

## Reviews at close

- **rules-auditor (Opus), whole spec vs rulebook, plus ACTIONS/SCENARIOS/layout vs spec: PASS.** No blockers or must-fix findings and no open rules questions. It found 2 should-fix and 4 minor items, all deferred.
- **code-reviewer (Opus), `git diff master...HEAD`: PASS.** No blockers or must-fix findings. It found 2 should-fix and 6 minor items, all deferred. Output is deterministic, no ids are hard-coded in code, and no "Player 1" labels appear.

## Findings fixed this phase

From the unit-5 spec audit:
- R-071 tie-returns example corrected.
- R-033 order (+1 before the ≤0 check) and R-040/R-042 maximal Lines resolution were made explicit, and the user decided both.
- R-039 now lists every case where lines may overlap: the mid-line counter-example and the end of another player's loop.
- R-046 wording, R-066 end types (now a set), R-019/R-073 start player, and the R-022 paraphrase.

From the board audit: the 28–35 `via` point moved and frame/bend tests were added.

## Findings deferred

Twelve findings are recorded in PLAN.md §9 Backlog, tagged Phase 0. Two need attention before or during Phase 1:
- `Board` is unhashable now that `Layout` holds dicts.
- Scenario move numbers need an origin so the R-070 tiebreak works for loaded scenarios.

Two spec wording items are queued for the next AMENDING window: "every site has a building" (R-049, R-066) and a missing full stop (R-046).

## Decisions this phase

- D15 (2026-11-14): the UI uses the user's map art as the background for both the map and the placement section. Only pieces are drawn on top, positioned from `data/board.yaml`'s `layout` block.
- User rules answers (2026-11-14):
  - Building supply is unlimited per type.
  - Multi-street Lines are evaluated one street at a time and must be maximal.
  - With 5 players the +1 is applied before the ≤0 check.
  - Players who run out of markers during the mandatory passes are skipped.
  - C4/C5 readings confirmed.
  - The C13 tiebreak uses the move number; ties that survive are shared.
- UI note for Phase 2: the Buses cell is one row per seat.
