# Phase 3 — Bots: strength evaluation (unit 7)

Baseline runs of `tools/tournament.py`, 4 players, win_loss returns, seats rotated, seeds 101–104.
No weight tuning was needed: both exit criteria were met by the first implementations.
Run with `uv run python -m tools.tournament --seats <list> --games N --seed S` (the module form; `python tools/tournament.py` cannot import `bots`).

| Run | Seats | Games | Subject win% | Wilson 95% CI | Notes |
|---|---|---|---|---|---|
| A | heuristic + 3 random | 200 | 72.6 | 65.9–78.2 | random seats 9.1% each; 0.13 ms/decision; 1.1 s wall |
| B | mcts:200 + 3 random | 200 | 95.0 | 91.0–97.3 | random seats 1.7% each; ~1.25 s/decision; 21 min wall (8 workers) |
| C | mcts:200 + 3 heuristic | 100 | 34.0 | 25.5–43.7 | heuristic seats 22.0% each; informational, parity is 25% |
| D | 2 heuristic + 2 random | 100 | 49.0 (per heuristic seat) | 42.2–55.9 | random seats 1.0%; informational |

Exit criteria (PLAN §7 Phase 3): MCTS beats random (B: lower bound 91.0% > 25% parity and > 50%) and the heuristic beats random (A: lower bound 65.9%), over ≥ 200 four-player games each. Met.

Observations
- Games are short: mean 4.5–5.2 rounds, almost all ending on the markers condition (last_stone 1–2 per 200, sites_full rare).
  Unit 3 flagged the heuristic's `drop_out` and `sel_*` weights as a place to look; heuristic-vs-heuristic games (run C) run 5.2 rounds.
- MCTS vs the heuristic (C) is only weakly above parity (lower bound 25.5% against 25%): 200 random-rollout simulations
  beat random play decisively but barely beat the heuristic. `mcts:200-heuristic` or more simulations are untested here; candidate follow-up before playtest #2 if wanted.
- Seat position shows no bias beyond noise (all positions 23–27% in A and B).
- MCTS decision time is ~1.2–1.4 s wall at 200 simulations while 8 games run in parallel; the server default think-time cap is 5 s.

---

# Phase 3 close report

Status: units 1-7 done; unit 8 (✋ playtest #2) is pending, so Phase 3 stays unticked in PLAN.md until the user has played and reports are triaged.

## Exit criteria
| Criterion | Evidence |
|---|---|
| Heuristic seat beats random over a few hundred 4-player games | 200 games: 72.6% [65.9, 78.2] vs three random seats |
| MCTS seat beats random | 200 games: `mcts:200` 95.0% [91.0, 97.3] |
| User plays against the bots | **Pending (unit 8)** |

## Full run (sim-runner)
ruff check, ruff format --check, pyright clean; pytest 1738 passed + 2 expected xfails; slow 240 passed (4m32s); vitest 451; `check:api`, typecheck, build clean; e2e 17 passed, 6 opt-in screenshot specs skipped.

## Reviews
- rules-auditor (scope: R-035–R-042 speed refactor, bot legality): **PASS**. Differential check against master's `bus/lines.py` over 1,500 random games (28.7k Lines decisions) and 15k random line positions: 0 mismatches. One minor wording item (mcts docstring).
- code-reviewer (`git diff master...HEAD`): **APPROVE**, no blockers or must-fix; 1 should-fix and 5 minor items, all deferred to the PLAN §9 backlog.

## Findings fixed / deferred
Fixed: none needed. Deferred: 7 items tagged "Phase 3" in PLAN §9 Backlog.

## New decisions
D24 (bot seat types, think-time cap and heuristic fallback). Speed: `state.clone()` 95 → 29 µs, random rollouts 120 → 155 games/s.
