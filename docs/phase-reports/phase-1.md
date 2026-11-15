# Phase 1 report — Engine: the full game, 3–5 players

Branch `phase-1/engine` · closed 2026-11-14 · no ✋ checkpoint (the next one is the Phase 2 playtest).

## Exit criteria

| Criterion | Evidence |
|---|---|
| Rule tests green | `uv run pytest -q -rxX`: **1334 passed**, 2 xfailed (the marker self-tests R-000 / report-000, which fail on purpose). No `pending` marker remains on any rule test. Every rule R-001–R-073 is exercised by at least one test (rules-auditor coverage table). |
| `random_sim_test` passes for 3, 4, 5 players | `tests/test_conformance.py`, serialize=True, both `returns` modes; 2 sims each in the fast suite, 25 each under `-m slow`. All pass. |
| 10,000 random games, zero invariant violations | `BUS_SIM_GAMES=10000 uv run pytest -q -m slow -rxX`: **238 passed** in 7 min 3 s. `check_invariants` after every action; dump/load and string round trips every 25th action and at terminal. **0 violations.** |
| `sim_stats` report | `uv run python -m tools.sim_stats --games 1000 --players 3,4,5 --seed 1` — summary below. |
| `rules-auditor` + `code-reviewer` pass | rules-auditor **PASS**, code-reviewer **APPROVE** (below). |

## Full run (final HEAD)

| Command | Result |
|---|---|
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 81 files already formatted |
| `uv run pyright` | 0 errors, 0 warnings |
| `uv run pytest -q -rxX` | 1334 passed, 2 xfailed |
| `BUS_SIM_GAMES=10000 uv run pytest -q -m slow -rxX` | 238 passed (7:03) |
| `uv run python -m tools.play_text --players 4 --seed 7` | 159 moves, END (markers), P0 wins 2–1–0–(−1) |
| golden playthroughs (`tests/test_playthroughs.py`) | 3p seed 1, 4p seed 2, 5p seed 3 replay identically |

### `sim_stats` (1,000 uniform-random games per count, seed 1)

| | 3 players | 4 players | 5 players |
|---|---|---|---|
| End: markers / last stone / sites full (incl. with markers) | 95.0% / 5.0% / 0% | 98.4% / 1.2% / 0.4% | 95.1% / 3.0% / 1.9% |
| Rounds mean (max) | 4.0 (7) | 4.6 (7) | 5.1 (7) |
| Moves mean / max (cap) | 118 / 171 (702) | 158 / 213 (936) | 210 / 263 (1170) |
| Winner score mode | 0 | 0 | 2 |
| Seat win rate range | 0.32–0.34 | 0.24–0.26 | 0.18–0.22 |
| Sites built (incl. setup) | 20.1 | 26.0 | 30.4 |
| Passengers delivered | 1.3 | 2.4 | 4.7 |
| Mean legal actions | 9.2 | 8.8 | 8.2 |
| Time per game | 0.01 s | 0.01 s | 0.02 s |

Random play rarely reaches sites-full or the 25-marker cap; the bots in Phase 3 should. No seat-order bias beyond noise.

Engine numbers: observation tensor 1178 / 1474 / 1770 floats (3/4/5 players), ~100 µs per call; worst Lines legal-move search 1.2 ms.

## Units

1. Constants, action IDs, board fixes (f120d63) · 2. State model, registration (d31c138) · 3. Scenario loader, harness, invariants (f7518e0) · 4. Line geometry ⚑ (15928d1) · 5. Setup phases (32daa01) · 6. Action selection (b8974bd) · 7. Resolution driver, power, Buses ⚑ (01894c8) · 8. Lines maximal resolution ⚑ (a0731f0) · 9. Passengers, Buildings (cdfdab3) + spec amendment (9560e63) · 10. Clock ⚑ (cf40779) · 11. Vrrooomm! ⚑ (946d38a) · 12. Round/game end, returns ⚑ (7a233f2) · 13. Observer, action_meta, forced_action, game_result (ed67198, 13e3d61) · 14. Conformance and sims (67c915b) · 15. play_text, replay (e86cf4a) · 16. sim_stats, golden playthroughs (e25b292).

Every engine unit ran spec-test-writer before implementer. Six ⚑ units had an Opus spot audit; all PASS. One found a real bug before commit (unit 7: the round hand-off offered selections to a seat with no Action markers, R-027), fixed in the unit.

## Reviews at close

- **rules-auditor (Opus), R-001–R-073 vs spec, plus ACTIONS and SCENARIOS §9: PASS.** An independent whole-game checker ran 4,000 games under four policies (~175k decisions) with 0 mismatches against the spec; dump/load round trip at every node of 300 games (56,938 nodes) identical. No blockers, must-fix or should-fix; 2 minor (deferred). No rules questions.
- **code-reviewer (Opus), `git diff master...HEAD`: APPROVE.** No blockers or must-fix. Clone safety, serialization, action-id round trip, move-number bookkeeping, terminal handling, determinism and observer buffers checked sound. 2 should-fix (performance) and 7 minor (deferred).

## Findings fixed this phase (outside the unit work)

- Margin game declared `utility_sum` although GENERAL_SUM; `random_sim_test` rejected it (found by unit 14 tests; fixed in 67c915b).
- `sim_stats` counted setup buildings only (seat field) instead of built sites (e25b292).
- BD apply incremented a seat's building count although buildings have no owner (cdfdab3).
- Last-stone END did not round-trip through the loader (cf40779).
- An E501 lint error left by 13e3d61 (fixed in 67c915b).

## Findings deferred

Appended to PLAN.md §9 Backlog, tagged Phase 1: clone cost (deep copy of the frozen core), repeated Lines lookahead per step, unbounded observer cache, loader acceptance of dead non-terminal nodes before Phase 2 server input, tool and test minors, missing rule-ID tags (R-003, R-009, R-011, R-013), stale docstrings, `replay` not reading golden playthrough files.

## New decisions

- **D16** Scenario format clarifications (SCENARIOS §9.1–9.10): error codes for unnamed cases, `waiting` required from round 1, `move_offset`, `units_left` = full power on entry, last-stone END keeps the action board, canonical `end.types` order, R-064 applied at a round-end END.
- **D17** Spec amendment (user-approved): R-049 and R-066 say "every site has a building"; R-046 punctuation.
- **D18** Encoding (ACTIONS §7): absolute observation tensor with a seat-to-act one-hot, counts scaled to [0, 1], points scaled by 10 and capped (user); `game_result` None before terminal; current seat 0 at END.
