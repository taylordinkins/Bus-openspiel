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
