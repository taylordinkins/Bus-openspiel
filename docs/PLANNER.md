# Model-based planner bots (`planner`, `tplanner`)

Phase 4 bot work, 2026-11-19, branch `phase-4/planner`. Bots only: no engine, server or UI
changes. Seats are tool-only (registry, tournament, `tools/bot_eval.py`); not offered in the lobby.

## Idea

Bus is deterministic with perfect information, and most opponents play like one of a few known
policies. Instead of training a stronger net, the planner searches the real game with explicit
opponent models.

1. **Opponent identification** (`_Tracker`): every observed non-forced opponent decision updates
   a posterior over models: `ppo-long` greedy, with 10%/30% softmax deviations, at T = 0.5 and
   T = 1, mixed 50% with uniform; `ppo-smoke` greedy and T = 1; the heuristic; uniform random. It
   also tracks argmax agreement for the greedy models. A seat that looks random is planned
   against as a competent but noisy `ppo-long` instead, because MCTS bots look random to every
   net but play well.
2. **Batched lock-step rollouts** on `rules.Core` (no pyspiel states): every rollout advances one
   decision at a time and all states that need the same net share one forward pass.
3. **`tplanner` tree search** (the main bot): root sequential halving over at most 10
   candidates (ppo-long top-9 + the heuristic's pick), with early stop when a leader is clear by
   more than 2/√n. Below the root:
   - own nodes and **unpredictable opponents** (anything not ≥ 97% argmax-predictable by a greedy
     model) use max-n PUCT with a flattened ppo-long prior;
   - near-perfectly predictable opponents (greedy nets) collapse to their model's move;
   - sampled-net opponents are sampled from their model.

   Leaves get one model rollout each. The value is the win share + 0.05 × margin to the best
   opponent (clipped ±10) − 0.1 per own wasted marker.
4. **Wasted markers**: `surely_wasted` drops selections that cannot have an effect (Buses at 5
   buses; Lines/Passengers/Buildings at a letter whose power stays ≤ 0 even if this round's Buses
   raises M; empty supply; no sites). Rollouts and tree paths count the remaining waste (e.g.
   Vrrooomm! with nothing to carry) with the same definition as `tools/bot_eval.py`, which was
   checked to match the engine exactly.

`planner:R[-K][e]` is the flat version (sequential halving over candidates, R rollouts each).

## Seats

- `tplanner[:SIMS[-C]]`: tree planner, default 384 simulations, c_puct 1.5. About 15 s per
  decision on one core (~60 decisions per game), roughly halved at 192.
- `planner[:R[-K][e]]`: flat planner (≈1 s per decision at R = 8).
- `net_sample:<model>`: a manifest net sampling from its policy (evaluation opponent).
- `gumbel_sup:<R|R_leader|R_rank>` (registered by `tools/bot_eval.py` only): the experiment A2
  search player with the supervised value net from `runs/value_sup/a1/`.

Environment knobs (experiments only): `PLANNER_WASTE`, `PLANNER_RANDOM_AS`, `PLANNER_MAXN`,
`PLANNER_DET_AGREE`, `PLANNER_SAMPLE_MODE`, `PLANNER_TREE_EARLY`, `PLANNER_BATCH`.

## Evaluation

`uv run python -m tools.bot_eval --cand tplanner --opps a,b[,c] --games 12 --workers 12`
puts the candidate in every seat in turn (game g at seat g mod n). Results are in `runs/planner/*.txt`.
Samples are small (12 games per cell, as asked), so one cell's 95% interval is about ±0.2.

Win share of the candidate (12 games per row, rotated over seats), its mean wasted markers per
game, and the opponents' waste for comparison:

| players | opponents | bot (sims) | win | waste (opp) |
|---|---|---|---|---|
| 3 | heuristic ×2 | tplanner:96 | 1.00 | 0.33 (9.4) |
| 3 | ppo-smoke ×2 | tplanner:96 | 1.00 | 1.83 (5.6) |
| 3 | random ×2 | tplanner:96 | 0.96 | 1.33 (9.8) |
| 3 | ppo-long + heuristic | tplanner:96 | 1.00 | 0.83 (9.6) |
| 3 | ppo-long + ppo-smoke | tplanner:96 | 1.00 | 1.33 (7.8) |
| 3 | ppo-long ×2 | tplanner:96 | 1.00 | 1.42 (7.3) |
| 3 | sampled ppo-long ×2 | tplanner:96 | 0.83 | 1.17 (7.4) |
| 3 | mcts:100 ×2 | tplanner:192 | 0.92 | 0.75 (8.6) |
| 3 | R_leader (A2) ×2 | tplanner:192 | 0.75 | 1.42 (6.5) |
| 3 | gumbel:32 ×2 | **tplanner (384)** | 0.92 | 0.75 (7.0) |
| 3 | mcts:200 ×2 | **tplanner (384)** | 0.75 | 1.00 (6.4) |
| 4 | ppo-long ×3 | tplanner:192 | 1.00 | 0.83 (6.9) |
| 4 | ppo-long + heuristic + ppo-smoke | tplanner:192 | 1.00 | 0.58 (7.3) |
| 4 | sampled ppo-long ×3 | tplanner:192 | 1.00 | 0.92 (6.9) |
| 4 | gumbel:32 ×3 | tplanner:192 → **384** | 0.42 → 0.92 | 0.75 (7.1) |
| 4 | R_leader (A2) ×3 | tplanner:192 → **384** | 0.42 → 0.83 | 0.58 (6.4) |
| 4 | mcts:100 ×3 | tplanner:192 → **384** | 0.75 → 1.00 | 0.42 (7.9) |

Baselines for scale: greedy `ppo-long` wastes ~5 markers per game and the heuristic ~9; greedy
`ppo-long` wins 0.50 of 3p games against two mcts:100, the heuristic 0.00.

What helped, in order: exploiting deterministic greedy opponents (flat planner, 3p vs ppo-long
≈ 0.96); sequential halving with common random numbers against stochastic opponents; not
modelling MCTS as random; the max-n tree for unpredictable opponents (vs gumbel 3p 0.50 → 0.75
at equal seeds); simulations (192 → 384 doubles the 4p win rate against the gumbel family).
What did not help: greedy `ppo-long` rollouts for MCTS-like seats; larger simulation batches
(engine cost per state dominates, not the net).

Open: 3p vs mcts:200 (0.75) and vs R_leader at 192 sims (0.75; not rerun at 384); mcts:1000 and
5 players are untested (too slow for this budget); 12-game cells are noisy.
