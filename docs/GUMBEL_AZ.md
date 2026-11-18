# Gumbel AlphaZero for Bus: design and throughput spike (Phase 4 unit 8f)

Status: design for review, 2026-11-17. Nothing here is built yet beyond the spike
`tools/bench_search.py`. Open decisions for the user are collected in PLAN.md §9 (Q-AZ1–Q-AZ7).

References: Danihelka, Guez, Schrittwieser, Silver, "Policy improvement by planning with Gumbel"
(ICLR 2022), and its reference implementation `mctx` (`gumbel_muzero_policy`,
`qtransform_completed_by_mix_value`); Silver et al. AlphaZero (2018); Schrittwieser et al. MuZero
(2020) and MuZero Reanalyse; Petosa & Balch, "Multiplayer AlphaZero" (2019, vector values);
Sturtevant (max^n), Sturtevant & Korf (paranoid), Schadd & Winands (best-reply search),
Nijssen & Winands, "Search policies in multi-player games" (ICGA J. 2013).

## 1. Where we are and why search

- `ppo-long` (runs/long best_1320) plateaued: value explained variance (EV) ~0.25–0.30, and an
  exploiter warm-started from it beat 3 frozen copies 0.73/0.71/0.35 at 3/4/5 players (fair
  0.33/0.25/0.20). The dense relative-score reward (unit 8e) gained ~+0.05–0.1 at 3–4p and left EV
  where it was (early-game quartile ~0.12–0.18).
- Bus is deterministic, perfect information, 3–5 players, no chance nodes: the setting where
  expert iteration (search as a policy-improvement operator, net distils the search) is the
  standard answer to a policy-gradient plateau. Gumbel AlphaZero is the variant that still
  improves the policy with a handful of simulations, which is what Python can afford.

### 1.1 Why unit 7's net-guided MCTS did not help

Measured facts (spike and `runs/eval_search`):

1. **The prior is nearly deterministic.** On decision states of `ppo-long` self-play the top
   action has mean probability 0.92 (3p) / 0.95 (4p) / 0.93 (5p); 75–85% of decisions have a top
   action above 0.9; policy entropy 0.14–0.21 nats. With PUCT (`c` = 1.5) and 50 simulations
   the low-prior actions get almost no visits, so "most visited" almost always equals the
   prior's argmax, and where it differs it is driven by a noisy Q (point 3).
2. **Flattening the prior is what helped.** `mcts:50` on `ppo-long` vs 3 sampled `ppo-long`:
   +0.01/+0.015/+0.045 over chance at 3/4/5p; `mcts:200` +0.07 at 4p; but prior temperature 10
   (`probe_temp`) gave 0.36 at 4p (+0.11). Forcing the search to look at alternatives pays; PUCT
   with a sharp prior does not. Gumbel's sequential halving visits every considered action by
   construction (m = min(16, legal); legal actions per decision: mean 7–8, median 5–6, more than
   16 in only 6–9% of decisions), so it does this without a tuned temperature.
3. **The value is weak early.** EV ~0.12 in the first quartile means Q noise is larger than the
   real gaps between early moves. Any search inherits this; Gumbel does not fix it, it only
   makes the improvement step better behaved with few simulations (§2.3, risk R1).
4. **Plumbing costs.** OpenSpiel's `MCTSBot` spends a simulation on every in-tree forced move
   (13–15% of decisions have at most 2 legal actions and forced moves are only skipped at the
   root), uses an unvisited-child value of 0, and mixes scales (terminal = win_loss returns, net
   value = 0.7 win_loss + 0.3 margin). It also costs ~3 ms per simulation (pyspiel clones and
   Python node objects) versus ~0.2–0.25 ms in the spike.
5. The smoke net (25% vs 3 plain-net seats) had all of the above plus a 4p-only, 20-minute value.

## 2. Algorithm

### 2.1 Search (per decision, one tree)

Gumbel MuZero's search with the true simulator (so "Gumbel AlphaZero"), as in `mctx`:

- **Root:** sample g ~ Gumbel(0, 1) per legal action (scale 1 in self-play, 0 in evaluation).
  Consider the top m = min(16, #legal) actions by g + logits. Sequential halving over them with
  the `mctx` visit schedule (`considered_visits(m, n)`); after the budget, play
  argmax over the surviving actions of g + logits + σ(q̂). That action is both the move played
  in self-play (Gumbel-top-k is the exploration; no Dirichlet noise, no temperature schedule) and
  the evaluation move (g = 0).
- **Interior nodes:** deterministic selection argmax_a [π'(a) − N(a) / (1 + Σ_b N(b))] with
  π' = softmax(logits + σ(completed q̂)), as in the paper (no PUCT constant to tune).
- **Completed Q:** visited actions use their mean backed-up value for the node's mover;
  unvisited actions use the mixed value v_mix = (v̂ + Σ_b N(b) · Σ_visited π(b) q(b) / Σ_visited π(b))
  / (1 + Σ_b N(b)) (`mctx` `use_mixed_value=True`).
- **σ transform:** σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂ with q̂ min–max rescaled to
  [0, 1] over the node's completed Q values (`mctx` defaults c_visit = 50, c_scale = 0.1; the
  rescaling makes the value scale irrelevant, so no global value normalisation is needed).
- **Forced moves** are applied while expanding an edge (one simulation = one evaluated decision
  node or terminal), as in `train/env.py`; they never consume budget.
- **Tree nodes hold the immutable engine `Core`** (`bus.rules.apply` returns a new one), so search
  never clones a pyspiel state. No transposition table and no subtree reuse between moves (as in
  `mctx`; both are optional later).
- **Batched across games:** an actor runs B concurrent games; each wave every open search
  selects one leaf, one forward evaluates all leaves, then all back up.

### 2.2 Multiplayer adaptation

- **Value representation:** the net keeps its 5-wide ego-ordered value head (index 0 = seat to
  move, absent seats masked). On expansion the vector is rotated to absolute seats; the tree
  stores and backs up the full per-seat vector unchanged (no sign flips, no rewards in transit,
  discount 1). Terminal nodes use the same per-seat target quantity as training (§3.2), not
  `state.returns()` (fixes point 4 above).
- **Which seat drives selection:** at every node the mover reads its own entry of the vector
  (completed Q, v_mix and σ are computed for the node's mover only). This is the max^n reading
  with mean backups (as in Petosa & Balch's multiplayer AlphaZero and the "MP-MCTS" of Nijssen &
  Winands), not paranoid (assumes a coalition against the root player; too pessimistic in 3–5p
  and wrong about how the opponents actually play) and not best-reply search (collapses
  opponents into one move; changes the game tree and the policy targets of opponent nodes).
- **Policy improvement:** the Gumbel guarantee is per node and per mover: given the q estimates
  of the mover, sequential halving + completed-Q softmax is an improvement over the prior for that
  mover. With N players that is exactly what we need for the root (the only node whose target we
  train on); interior nodes model each opponent as improving its own value.
- **Deviations from the paper** (all deliberate): per-seat value vectors instead of a scalar;
  forced edges collapsed; value targets are a relative-score blend, not a win probability
  (§3.2); synchronous generations (§5); no reanalyse in the first version.

### 2.3 Hyperparameters (first experiment)

| Knob | Value | Why |
|---|---|---|
| simulations n | 32 (try 16 and 64) | Gumbel improves reliably from n ≥ m; m ≈ legal count here |
| considered m | 16 | covers all actions in > 90% of decisions |
| c_visit, c_scale | 50, 0.1 | `mctx` defaults; sweep c_scale 0.03 / 0.1 / 0.3 in unit 8g |
| Gumbel scale | 1 self-play, 0 eval | paper |

## 3. Training targets and losses

### 3.1 Policy

Target = the completed-Q improved policy softmax(logits + σ(completed q̂)) over the root's legal
actions, cross-entropy against the net's masked policy. Not visit counts: with n ≈ 2m visits are
a coarse and biased target (Danihelka et al.: visit-count targets need many simulations; the
improved policy is valid at any n). Forced decisions produce no sample. No entropy bonus.

### 3.2 Value

- **Quantity:** per-seat R = mix · win_loss + (1 − mix) · margin, with margin the clipped R-072
  relative margin (own score − mean opponent score) / 10 and score = points − stones. This is the
  objective the 8e dense run optimised (mix 0.5 held constant, user decision 2026-11-17). It is
  **relative**, so taking a time stone (−1 for the taker, denies the others) is valued correctly;
  own-score-only shaping is ruled out. Evaluation stays pure win_loss (D7). Whether to keep the
  margin term or move to win_loss only is Q-AZ2.
- **Target:** the final outcome z (Monte Carlo, per seat, ego-rotated, seat mask as in PPO) for the
  first experiment, with an option `--value-target z|zq` where zq = (z + q_root) / 2 averages in
  the root's search value vector ("Lessons from AlphaZero" style). zq lowers early-game target
  variance, which is exactly where our EV is weak, but bootstraps from a weak value; try it only
  if z leaves early EV < 0.2. n-step bootstrapping is not needed: there are no intermediate
  rewards in the search formulation and games are short (150/205/280 decisions at 3/4/5p).
- **Dense reward is not reused.** Potential-based shaping matters for PPO's credit assignment;
  in AZ the search consumes the value directly and the target is the final R. (A later variant:
  predict the residual R − (1 − mix) · Φ(s), with Φ the current relative margin, as an
  easier-to-learn baseline. Not in the first experiment.)

### 3.3 Aux heads

Keep the unit 8e training-only aux heads (per-seat final score, rank, winner) with a small weight
(0.25–0.5): they are cheap, never exported, and their targets come for free from the finished
game. They can be dropped if they do not raise value EV in the first experiment.

Loss = CE(π', π_θ) + c_v · MSE(R, v_θ) (present seats only) + c_aux · aux + weight decay 1e-4.

## 4. Network, warm start, export

- Reuse `BusNet` (512×512 tanh, separate value trunk, 5-wide value head) and observation v2
  (`FEATURES_VERSION` 1). The exported `.npz` is the same format 2, so a trained AZ net is
  immediately a `net:<name>` seat, a `mcts_net:` seat and a fixed league/eval opponent. A larger
  trunk is possible later (forward is 7–30% of actor time) but would not change the format.
- **Warm start (recommended, Q-AZ1):** from `runs/long/best_1320.pt` (= `ppo-long`). Its policy is
  strong (93% vs 3 heuristic at 4p), so 32-simulation search starts from good priors and the run
  measures whether search adds anything. The value head was trained on 0.7 wl + 0.3 margin; keep
  it (close to the new quantity) with a short value-only warm-up (first ~2 generations train the
  value and aux heads only, policy frozen) so the policy is not pulled by a mis-scaled value.
  `runs/dense` checkpoints predict return-to-go, not the final value, so they need a conversion
  (v_final = v_rtg + (1 − mix) Φ(s)) and are a worse starting point.
- Fresh start is the AlphaZero orthodoxy and avoids inheriting PPO's quirks, but at our
  throughput (§6) it would spend most of a session relearning what `ppo-long` knows.

## 5. Self-play loop

- **Synchronous generations** (like `tools/train_ppo.py`): each generation the actors play G games
  with the current weights, the learner trains on the replay buffer, exports `.npz`, repeat.
  Simple, resumable and reproducible; the learner is so cheap (below) that overlap buys little.
  A later version can let actors run one generation behind.
- **Actors:** `--workers` CPU processes (default 6, configurable; ~80 MB peak RSS each in the
  spike), each with B = 32 concurrent games and the numpy forward in-process with one BLAS thread
  (same as the PPO actors). No CUDA in actors: a torch/CUDA process is ~960 MB RSS, so 8 of them
  would not fit in 7.9 GB.
- **Batched GPU inference server (not in v1):** it would remove the 20–35% forward share from the
  actors (≤ 1.4× at the same worker count) for an IPC path of ~2–9 KB per evaluation. Worth it
  only after the Python search itself is faster (§6.3).
- **Samples** (one per non-forced decision of every seat; all seats are the current net in pure
  self-play): sparse observation (4–5% non-zero, ≤ 36 distinct values: int16 index + uint8 value
  code, ~350 B), legal mask (bit-packed), improved policy over legal actions, seat count, z per
  seat, root q vector, aux targets, game stage. ~0.6 KB per sample; returned through `/dev/shm`
  spill files (`train/shm.py`), as in PPO.
- **Replay buffer:** FIFO window of the last 500k samples (~15–20 minutes of play at 32
  simulations on 6–8 workers; ~0.3 GB), uniform sampling, densified on the GPU per minibatch.
- **Learner:** batch 1024, Adam lr 3e-4 → 1e-4 cosine, target sample reuse ~4 (samples trained
  per sample generated). At ~1.5M samples/h that is ~6k steps/h, under a minute of GPU time per
  hour, so the learner never limits; reuse can be raised if the held-out value loss allows.
- **Reanalyse:** optional, off. Actors are the bottleneck and a reanalyse search costs the same as
  a fresh one, so it only pays for fresher targets on old samples.
- **Opponent mix:** pure self-play for v1 (all seats current net with search). Adding a share of
  frozen seats (`ppo-long`, hall of fame) is Q-AZ5; they would yield no targets and blur the
  search's opponent model.
- **Checkpoint/resume:** reuse `train/runstate.py` (`save_pt`, `write_latest`,
  `apply_retention`, `merge_resume_args`, `truncate_csv`, best shortlist) and `train/atomic.py`
  for model, optimizer, generation counter, RNG states and the eval/best records. The replay
  buffer is not persisted (a resume refills it for ~2 generations before training; logged).
  SIGINT/SIGTERM handling and shm cleanup as in unit 8a.
- **Memory budget:** learner with torch/CUDA ~1.0 GB + buffer 0.3 GB + 8 actors × ~0.1 GB + eval
  workers ~0.5 GB → ~2.6 GB, under the 4 GB cap. The buffer size is the knob if needed.

## 6. Throughput spike (measured 2026-11-17, this machine)

`uv run python -m tools.bench_search` (RTX 3080, 12 logical cores, WSL2; net = `ppo-long`,
4 players unless stated; self-play starts from states of `ppo-long` self-play so all game phases
are represented; rates are steady state: decisions/s = simulations/s ÷ n).

### 6.1 Components (per process)

| Item | Cost |
|---|---|
| `rules.legal_actions` | 2.6 µs (memoised) |
| `rules.apply` | 27 µs on random-game states, ~80 µs mid-game in search (profile) |
| `encode` (obs v2) | 29–30 µs |
| pyspiel `state.clone` (not used by the search) | 28 µs |
| numpy forward, 1 BLAS thread, B = 1 / 16 / 64 / 256 | 312 / 101 / 58 / 48 µs per obs |
| torch CUDA forward incl. copies, B = 1 / 16 / 64 / 256 | 384 / 26 / 6.9 / 2.4 µs per obs |
| decisions per game (net self-play) | 150 / 205 / 283 at 3 / 4 / 5p |

### 6.2 Self-play under Gumbel search

One process (4p):

| Backend | B | n=16 sims/s | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|
| numpy | 1 | 1720 | 1580 | 1540 | 1560 |
| numpy | 32 | 4330 | 3990 | 3610 | 3170 |
| torch CUDA | 64 | 5190 | 4710 | 4400 | 3970 |
| torch CUDA | 256 | 6440 | 5720 | 5050 | 4420 |

3p and 5p are within ±10% per simulation (numpy B=32 n=16: 4420 / 4150; n=64: 3760 / 3510).

Worker processes, numpy B = 32 each (4p):

| Workers | n | sims/s | decisions/s | samples/h | games/h (4p) |
|---|---|---|---|---|---|
| 4 | 16 | 14.2k | 887 | 3.2M | 15.6k |
| 4 | 64 | 10.6k | 165 | 0.6M | 2.9k |
| 8 | 16 | 19.4k | 1215 | 4.4M | 21.4k |
| 8 | 64 | 15.1k–18.3k | 235–286 | 0.85–1.0M | 4.1k–5.0k |
| 11 | 16 | 21.2k | 1322 | 4.8M | 23.3k |
| 11 | 64 | 17.4k | 271 | 1.0M | 4.8k |

(n = 32 at 8 workers interpolates to ~17k sims/s, ~530 decisions/s, ~1.9M samples/h, ~9k
games/h at 4p.) Scaling is sub-linear (8 workers ≈ 4.5×, 11 ≈ 4.9×): 6 physical cores under WSL.
Peak RSS per numpy actor ~80 MB.

Profile (cProfile, torch B = 64, n = 64): search bookkeeping (completed Q, softmax, selection,
backup) ~43%, engine `apply`/`legal_actions` ~33% (`dataclasses.replace` alone ~13%), `encode`
~12%, forward and copies ~7%. Compared with PPO's ~17k decisions/s, search self-play produces
~30× (n = 16) to ~65× (n = 32) fewer decisions per second.

### 6.3 C++ port: recommendation

**Do not port now.** The numbers behind it:

- The engine plus `encode` is ~45% of actor time; porting only those caps the gain at ~1.8×.
  The other ~45% is the search's own Python bookkeeping, which a C++ engine does not touch.
- A full port (engine, features, search, batched GPU server) could reach the GPU limit
  (~400k evaluations/s at B = 256), ~20× today, but it is weeks of work with bit-exact
  equivalence tests against `bus/rules.py`, `train/features.py` and the action ids.
- Cheaper first, each measurable with `tools/bench_search.py`: (a) a leaner search (scalar
  Python for nodes with ≤ 8 actions, cached σ inputs, no per-call numpy temporaries; or numba
  for the selection math), est. 1.3–1.5×; (b) a faster `Core` copy path than
  `dataclasses.replace` (unit 3b already found it at 55% of a plain env step), est. ~1.15×;
  (c) the GPU inference server, ≤ 1.4×. Together ~2×, days not weeks.
- The current throughput (~1.9M samples/h at n = 32 on 8 workers) is enough for the first
  experiment. Revisit the port with a new trigger: the first experiment shows Gumbel AZ beating
  `ppo-long` on held-out opponents **and** curves still rising when the run is data-bound
  (learner waiting on actors) after the cheap speedups. That trigger replaces unit 9's PPO-era
  one (Q-AZ4).

## 7. Evaluation

- **Held-out opponents** (never in the AZ training mix): exploiter best
  (`runs/exploit_long/best.npz`), `mcts:200`, older `runs/long` checkpoints (`ckpt_250`, `ckpt_750`),
  `ppo-smoke`, `runs/dense/best_410`, plus `ppo-long` and `heuristic`. Tables: candidate vs 3
  copies of one opponent at 3/4/5p, and mixed tables (e.g. candidate + `ppo-long` + exploiter +
  heuristic at 4p) with seats rotated and common seeds (`tools/eval_search.py` already does
  rotation, common random numbers and 1/k shared wins).
- **Two candidates per checkpoint:** raw policy (greedy and sampled; what the lobby `net:` seat
  plays) and policy + search (Gumbel, n = 32, g = 0). The raw policy is the distillation test; the
  search player is the strength test.
- **Exploitability benchmark:** rerun the unit 8d exploiter test against the final AZ net
  (warm-start PPO exploiter vs 3 frozen raw AZ copies, same budget as `runs/exploit_long`,
  ~300 updates) and compare with 0.73/0.71/0.35 against `ppo-long`. Lower is better.
- **Eval-baseline offset:** checked with the trainer's own harness (`train.rollout.play_games`,
  sampled, seats rotated) with the learner identical to the reference (`ppo-long` vs 3
  `ppo-long`), 600 games per count: 0.307 / 0.237 / 0.213 vs fair 0.333 / 0.25 / 0.20, i.e. no
  harness bias. The ~0.43–0.50 at 3p in `runs/exploit_long` and `runs/dense` is the first eval
  row, which is written at u10, after 10 updates of a learner that adapts fast to a fixed
  opponent, plus noise: a 100-game cell has a Wilson half-width of ~0.09 at p = 0.33. Also, the
  trainer counts a shared win as a win (`wl > 0`) where tournaments count 1/k; no shared wins
  occurred in those 1,800 games, so it is harmless, but the AZ trainer should use 1/k. Fixes for
  the AZ trainer: evaluate at generation 0 before any training, ≥ 300 games per 3p cell for
  head-to-heads, common seeds across checkpoints.

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Weak early value makes completed-Q targets noisy; distillation chases noise | Unit 8g search-only check first; c_scale sweep; zq target option; value warm-up |
| R2 | Sharp PPO prior → near-one-hot targets, little to learn | Gumbel visits all considered actions; σ shifts logits by up to ~5–10; monitor target entropy and KL(π' ‖ π) |
| R3 | Throughput: ~10k games/h at n = 32 is small for AZ | Warm start; small n; cheap speedups (§6.3) before C++ |
| R4 | Pure self-play cycles or collapse in 3–5p | Held-out eval tables, exploiter test; frozen-seat share (Q-AZ5) if it shows |
| R5 | Memory (WSL OOM at 11 PPO workers) | Sparse samples, ≤ 0.3 GB buffer, numpy-only actors, worker count flag, RSS logged |
| R6 | Mixed 3/4/5p dilutes a 1–2 h signal | Option for a 4p-only first run (Q-AZ6) |

## 9. Minimum viable first experiment

Two steps, ~2 hours of wall time in total:

1. **Search-only check (unit 8g, ~15 minutes):** Gumbel search on the frozen `ppo-long` (no
   training) as one seat vs 3 sampled `ppo-long` seats, 400 games per cell at 4p, n = 16/32/64,
   c_scale 0.03/0.1. If search does not beat the raw policy here, distilling it cannot help;
   stop and fix the value (zq targets, value-only fine-tune on R) before building the trainer.
2. **AZ run (unit 8i, 90 minutes):** warm start from `ppo-long`, mixed 3/4/5p (0.3/0.4/0.3) or
   4p-only, n = 32, m = 16, 6–8 workers × 32 games, generations of ~256 games, z targets, mix
   0.5, aux 0.25. Expected volume: ~2.5–3M samples, ~13k games. Eval every ~15 minutes on the
   raw policy (cheap) vs 3 `ppo-long`, vs 3 heuristic and vs the exploiter, 300 games per cell at
   3/4/5p; search-player eval at the end only.

**Success (go on):** the raw AZ policy beats 3 `ppo-long` at 4p with a Wilson lower bound above
0.25 over ≥ 400 games, without losing more than 0.05 vs 3 heuristic or vs the exploiter, and
value EV in the first game quartile above 0.2. **Failure (stop and rethink):** no significant
gain over the 8e dense run's +0.05–0.1 within 90 minutes.

## 10. Interaction-aware value and search (after attempts 1–2, 2026-11-17)

Status: built and tested (PLAN units 8k–8n); the experiments in §10.6 are designed, not run.

### 10.1 Diagnosis

- Every method stalls at a value EV of ~0.25–0.30. The noise ceiling (4p, 1600 positions × 32
  `ppo-long` continuations at τ = 1, `tools/noise_ceiling.py`) is 0.25 / 0.38 / 0.63 / 0.91 by
  game quartile (0.54 overall) against single outcomes; the best net reaches 0.14 / 0.16 / 0.37 /
  0.55 (0.30), about 0.56 R² against the K-mean. Under heuristic continuations the ceiling is
  0.75 overall.
- In AZ, once the policy moves, the value cannot rank actions: search targets drift
  (search-differs 15% → 36%) and play against the heuristic collapses (PLAN 8i, attempts 1–2).
- The user's diagnosis (taken as given, not re-tested): Bus is highly interactive. Board state and
  marker placement matter a lot, players easily disrupt each other through marker placement and
  resolution choices, so outcomes are driven by the other players' choices. Bus is deterministic
  with perfect information (R-073), so all outcome spread comes from those choices; a value that
  does not model the interaction cannot predict it.

Three directions follow (pool diversification comes later): model opponents in the search
(§10.2), give the net interaction features (§10.3), and make the value target leader-aware
(§10.4). §10.5 is a cheap supervised path to test the value directions without self-play.

### 10.2 Opponent modelling in the search

**Problem.** Gumbel explores widely only at the root. At a non-root node the selection
argmax_a [π'(a) − N(a) / (1 + ΣN)] is a deterministic quasi-sampler: it keeps the node's visit
distribution close to π' = softmax(logits + σ(q̂)) (Danihelka et al. 2022, §5). The prior is
therefore the search's *model of how the seat at that node plays*. `ppo-long`'s prior is nearly
one-hot (top probability 0.92–0.95, §1.1), and with 1–10 visits per opponent node σ(q̂) cannot
move it: the first visit always takes the argmax, and with all-equal Q the off-argmax visit share
at opponent nodes is 0 (`tests/bots/test_gumbel_interaction.py`). The search assumes every
opponent plays its single likeliest move and never sees the disruptive replies that decide Bus
games.

**Method chosen: a tempered / ε-mixed opponent prior at non-root nodes.** With `opp_temp` = T,
`opp_mix` = ε, the prior of every node whose mover is not the root's mover (`opp_scope =
"opponents"`; `"nonroot"` also covers the root player's own later nodes) is

    p_opp = (1 − ε) · softmax(logits / T) + ε / k    (k = legal actions at that node)

and its log replaces the logits in π', v_mix and the selection rule (`bots.gumbel.opponent_prior`;
the root is untouched, so Gumbel-top-k, sequential halving, the improved-policy target and its
guarantee are unchanged). The defaults T = 1, ε = 0 leave the search bit-identical to unit 8g
(pinned by a reference test on recorded root outputs).

Why this and not the alternatives:

- **It is what Gumbel MuZero's non-root rule already is**, with a different behaviour model.
  Because selection tracks π', flattening the prior turns opponent nodes into a deterministic,
  low-variance expectation over plausible replies (an expectimax-like average under a Boltzmann
  opponent model) while σ(q̂) still lets each opponent prefer replies that are good *for it*
  (the max^n reading of §2.2 is kept). One knob per idea, no new node type.
- **Boltzmann (quantal-response) opponents** are the standard bounded-rationality model
  (McKelvey & Palfrey 1995): a temperature keeps the ranking of the net's replies and gives a
  plausible second move more weight than a blunder. The uniform share ε is the complementary
  "any legal move" floor: it reaches low-prior blocking placements the PPO net almost never makes
  (R-026 allows ineffective blocking markers), at the price of visits on absurd moves. Both are
  exposed; the experiment picks.
- **max^n with a sharp model is brittle; paranoid and best-reply are wrong models here.** max^n
  (Luckhardt & Irani 1986; Sturtevant 2003) assumes each opponent maximises its own value under
  the model; with a near-deterministic model that collapses to "opponents play the PPO argmax".
  Paranoid search (Sturtevant & Korf 2000) assumes a coalition against the root and is too
  pessimistic at 3–5 players; best-reply search (Schadd & Winands 2011) lets only one opponent
  move and changes the game tree. A softened max^n sits between "opponents play their argmax" and
  "opponents do anything".
- **Not sequential halving or Gumbel sampling at opponent nodes.** Sequential halving needs
  ≥ m·log2 m visits to mean anything and its guarantee is about the root's simple regret;
  opponent nodes get 1–10 visits at n = 64. Danihelka et al. use it only at the root for that
  reason. Gumbel noise at interior nodes would add variance to a 64-simulation search.
- **Not sampled opponent replies (chance-node style, as Stochastic MuZero treats chance).**
  The deterministic selection already allocates visits ∝ π' without sampling noise and keeps the
  search reproducible for a fixed seed.

Effects to expect: root Q values become expectations against a spread of replies, so actions that
only work if every opponent plays its argmax lose value and robust or disruptive actions gain;
search-differs rises a little. In AZ self-play only root targets are trained, and every seat's
target comes from its own root search, so the opponent model never trains the opponents'
policies directly.

Exposed as `GumbelConfig(opp_temp, opp_mix, opp_scope)`; env `BUS_GUMBEL_OPP_TEMP`,
`BUS_GUMBEL_OPP_MIX`, `BUS_GUMBEL_OPP_SCOPE` for registry seats; `tools.eval_search` candidates
`gumbel:<n>,t=<T>,e=<ε>,scope=<…>`; `tools.train_az --opp-temp --opp-mix --opp-scope`. Cost:
none measurable (`gumbel:64` 67–73 ms per decision with or without it, 4p smoke).

### 10.3 Observation v3: per-seat and interaction features

`train.features.encode_v3` (`FEATURES_V3` = 2, 3219 inputs) is the v2 vector unchanged (same
blocks and offsets) followed by 905 inputs, all from public state through the rule helpers and
board data (no ids in code; the module docstring lists every block with its rule IDs):

| Group | Blocks |
|---|---|
| v2's ego-only derived features for every seat | `seat_pickable` (N, 5), `seat_free_sites` (N, 5), `seat_vroom` (5, 3) |
| What each seat can score now | `seat_vroom_now` (5): the R-060 move count (= `rules.vrrooomm_moves`); `seat_targets` (5, 3): unoccupied buildings per type on the line (R-059 destinations per clock type) |
| Who can reach what next | `seat_reach` (N, 5): nodes one legal extension would add (R-036..R-039, R-041) |
| Contest between seats | `contested_node` (N, 3): waiting passengers and free buildings on the ego line that an opponent's line also covers, and opponents one extension away from the ego's nodes; `pair_overlap` (5, 4): per opponent, shared nodes / waiting / buildings / streets; `frontier_contested` (S): opponents whose legal extensions include the ego's extension streets |
| Who acts before whom | `res_before`, `res_after` (7, 5): per action, each seat's markers resolving before / after the ego's first marker there (R-029, R-030); `res_pending` (7, 5): markers still to resolve this round |
| What a marker placed now gets | `place_now` (7, 2): power with the current M (R-025, R-031..R-033; M is re-read at resolution, R-032) and whether it would resolve first (R-030) |
| Round and score state | `sel_state` (5, 2): still selecting (R-024, R-027), markers placed; `score_gap` (3): ego score minus best / mean opponent (clipped /10) and the rank term |

Remaining action and line markers per seat were already in v2. No feature needed a rule reading
the spec does not cover (no §9 entry): reach and frontier reuse v2's `legal_extensions` reading
(no R-042 lookahead, as v2), and `place_now` says explicitly that it uses the current M.

**Versions side by side.** `FEATURES_VERSION` (the default for new PPO runs, `encode`,
`OBS_SIZE`) stays v2, so `ppo-long`, `ppo-smoke` and `runs/az*` load and play unchanged.
`NetWeights` reads the features version from the stored layout hash (the v2 hash
`4bb7af3e33b12f0e` is pinned in a test) and exposes its encoder; every seat type (`net:`,
`gumbel:`, `mcts_net`), the AZ actors (learner and frozen seats separately), the PPO/AZ eval
rollouts and the noise-ceiling tool encode with the net's own version, so mixed v2/v3 tables work.

**Warm start.** `NetWeights.widened(2)` (numpy) and `train.model.widen_model` (torch) copy the v2
first-layer weights of both trunks into the matching columns (`train.features.input_columns`)
and zero the 905 new columns; everything else is copied. The widened net equals `ppo-long` on
v3 observations (logits and values within 2e-5; identical Gumbel decisions and visit counts in
tests). `tools.train_az --features v3 --init-from <v2 .pt>` does this automatically.

**Cost.** `encode` v2 ≈ 19–20 µs per state on this machine; v3 ≈ 25 µs when the board/order
blocks hit their caches (typical inside a search, where many states share lines and passengers),
≈ 38–46 µs on a stream of random-play states, ≈ 89 µs worst case (every cache missing). At the
8f profile's ~12% encode share of actor time this is roughly +5–10% actor time. The forward
grows too: v3 has ~246 non-zero inputs per state vs ~153 for v2, and the sparse first layer
(`bots.gumbel.sparse_forward`, one BLAS thread) costs 152 vs 114 µs per observation at B = 1 and
57 vs 47 µs at B = 32 (`ppo-long` widened, 400 labelled positions). At the 8f forward share of
7–30% that is another +2–7% actor time.

### 10.4 Leader-aware value targets

R stays `mix` · win_loss (R-071) + (1 − `mix`) · relative term, mix 0.5 (Q-AZ2), final scores
s = points − stones (R-068). `bots.gumbel.relative_term`:

| `target` | relative term for seat p | range |
|---|---|---|
| `mean` (default, unchanged) | R-072 margin: (s_p − mean_{q≠p} s_q) / 10, clipped | [−1, 1] |
| `leader` | (s_p − max_{q≠p} s_q) / 10, clipped like R-072 | [−1, 1] |
| `rank` | 2 · (#opponents beaten + ties / 2) / (n − 1) − 1 | [−1, 1] |

- `leader`: for the leader it is the lead over the runner-up; for everyone else the gap to the
  leader; tied leaders get 0. It removes the mean-margin incentive to farm weak seats: in 4p,
  scoring 10 against 4, 4 and a leader's 14 is +0.27 on the mean margin but −0.4 on the leader
  margin. It is still relative (a stone costs its taker against the leader too), never
  own-score-only.
- `rank`: scale-free placement, robust to blowouts; ties count half, tiebreaks (R-069, R-070) are
  left to the win_loss half. Sums to 0 over seats.
- Same /10 scale and clip as R-072, so values stay in the tanh head's range and σ(q̂)'s min-max
  rescaling makes the scale irrelevant to the search anyway. Note the leader term is ≤ 0 for all
  but one seat, so its mean is negative (−0.14 in 4p over the 800-position smoke labels),
  unlike the zero-mean margin; the value head's bias absorbs it.
- Plumbed through `terminal_values(core, mix, target)` (default unchanged), `GumbelConfig.target`
  (terminal search values), the AZ trainer's z (`--value-quantity`) and `tools.noise_ceiling`,
  which now labels R, R_leader, R_rank and wl in one pass. Hand tests cover ties at the top,
  clipping and all-way ties.

### 10.5 Supervised value path

`tools.noise_ceiling --save-obs` stores both observations of every labelled position;
`tools.value_sup` trains only the value trunk and head (policy trunk and head frozen; checked) on
the K-mean of a chosen quantity, starting from `ppo-long` (widened for v3, so both start equal),
with early stopping on a held-out set, and reports per quartile: EV against single outcomes, R²
against the K-mean and the ceiling EV_max. Measured: labelling runs at ~190 continuations/s at 6
workers (800 positions × K = 8 in 41 s); training a cell takes seconds on the GPU. RAM: actors
≤ ~230 MB each; a 16k-position label file holds ~350 MB of dense observations in memory.

A smoke on 800 positions (640 train, 160 held out, 10 epochs; far too small to decide anything)
already moved held-out R² against the K-mean from 0.63 (`ppo-long`) to 0.73 (v2) / 0.73 (v3)
on R: the K-mean target alone is a much cleaner signal than single outcomes.

### 10.6 Initial experiments (designed, not run; cheapest decisive first)

All 4p, 6 workers, numbers from this machine. "R²" is held-out R² against the K-mean, "EV" is
against single outcomes, both ego seat (§10.5).

**A. Value-only supervised: features × target (unit 8l, ~25 min).**
- Hypothesis: interaction features close part of the gap to the ceiling where interaction
  matters most (q3/q4: ceiling 0.63 / 0.91 vs 0.37 / 0.55 reached), and the leader target is
  at least as learnable as the mean one.
- Setup: train labels `noise_ceiling --positions 16000 --k 8 --save-obs --nets "" --seed 201`
  (~12 min); held-out labels `--positions 4000 --k 16 --seed 202` (~6 min; a separate seed, so
  no game is shared); `value_sup --features v2,v3 --targets R,R_leader,R_rank` (6 cells, ~2 min
  on the GPU), then the best cell again with `--seed 1` to check training noise.
- Success: v3 beats v2 on the same target by ≥ 0.05 R² overall and ≥ 0.05 in q3 or q4 (on R
  and on R_leader), with the seed repeat agreeing in sign. For the target: compare R² as a share
  of each quantity's own ceiling, and carry the nets to A2 (predictability is not usefulness).
- Decision: v3 for attempt 3 if it wins; otherwise keep v2 (cheaper, no conversion).
- **Drop direction 2** if v3 − v2 < 0.02 R² overall and in both q3 and q4 on both targets.
- Side result that changes attempt 3 by itself: if supervised v2 on R already reaches R² ≥ 0.75,
  the value gap is mostly target noise, which argues for K-mean / zq-style targets in AZ.

**A2. Do better values make better search? (unit 8l, ~30 min, after A).**
- Hypothesis: the value-trained nets from A (policy identical to `ppo-long`, only the value
  differs) make `gumbel:64,c=0.1` stronger, and leader beats mean in play.
- Setup: `eval_search --candidates "gumbel:64,c=0.1" --baselines ppo_sample --players 4
  --games 800 --seed 1 --net runs/value_sup/a1/<cell>.npz --baseline-net models/ppo-long.npz`
  for the best v2 cell, the best v3 cell and v3 R_leader (3 × ~8 min); pair against
  `runs/gumbel_check/confirm` (`ppo-long`, 0.372 [0.340, 0.407], same seeds: join the two
  `games.jsonl` on (baseline, players, game) and apply `tools.eval_search.paired`).
- Success: a cell beats the confirm reference by a paired diff whose 95% interval excludes 0.
- Decision: the value init and target for attempt 3. **Drop direction 3** (keep mean, Q-AZ2) if
  R_leader neither beats R in A (share of ceiling) nor in A2 (paired diff ≤ 0).

**B. Search-only opponent modelling (unit 8m, ~40 min; independent of A, can run first if
preferred).**
- Hypothesis: a softened opponent model makes the frozen-`ppo-long` Gumbel player stronger
  against 3 sampled `ppo-long` (their actual replies are spread, τ = 1) without losing to the
  heuristic.
- Setup: `eval_search --candidates "gumbel:64,c=0.1;gumbel:64,c=0.1,t=2;gumbel:64,c=0.1,t=4;
  gumbel:64,c=0.1,e=0.25;gumbel:64,c=0.1,t=2,scope=nonroot" --baselines ppo_sample --players 4
  --games 800 --seed 1 --net models/ppo-long.npz --out runs/gumbel_check/opp
  --paired-ref "gumbel:64,c=0.1"` after `cp -r runs/gumbel_check/confirm runs/gumbel_check/opp`
  (a rerun rewrites the summary files, so never point it at the confirm directory itself); the
  copied `games.jsonl` makes it skip the 800 reference games and pair on the same seeds (4 new
  cells × ~8 min ≈ 32 min). Then the best setting and the reference vs `--baselines
  heuristic`, 400 games each (~6 min).
- Success: a setting with paired diff > 0 and 95% interval excluding 0 (at 800 games the
  half-width is ~0.035–0.045, so ≳ +0.04), Wilson lower bound > 0.34, and vs heuristic within
  0.03 of the reference. A borderline best (interval touching 0) gets 800 more games on fresh
  seeds (`--seed 2`) before deciding.
- Decision: the opponent setting for attempt 3 (or none). **Drop direction 1** if no setting is
  above the reference by more than +0.01 and the best's interval includes 0 after the extension.

**C. AZ attempt 3 (unit 8n, 90 minutes, only after A/A2/B).**
- Hypothesis: with a value that models interaction (v3 and/or leader target, initialised from the
  A2 winner) and a search that expects disruption, the value keeps ranking actions after the
  policy moves, so search targets stop drifting and the heuristic collapse does not recur.
- Setup: attempt 2's command (`docs/TRAINING.md`: n = 64, c_scale 0.1, 10 value-only warm-up
  generations, 128 games/gen, replay 300k, `--heuristic-share 0.15 --frozen-share 0.2
  --value-target zq`, lr 1e-4 constant) plus the winners: `--features v3` (widened warm start),
  `--value-quantity <winner>`, `--opp-temp/--opp-mix <winner>`. If A2 picked a value_sup net,
  attempt 3 needs it as the warm start: today `--init-from` takes a trainer `.pt` and
  `value_sup` writes `.npz`, so 8n first adds an npz → `.pt` conversion (small;
  `tools.value_sup.weights_to_net` already builds the `BusNet`), or it warm-starts from
  `best_1320.pt` and relies on the warm-up.
- Stop rule (kept from attempt 2): stop when vs 3 heuristic < 0.80 after warm-up. Resume
  pitfall: `--stop-after` is saved and carries over on `--resume`, so resume with
  `--stop-after 0` (attempt 2 lost a session to this).
- Success (go on): raw policy vs 3 `ppo-long` Wilson lower bound > 0.25 at ≥ 400 games, vs
  heuristic within 0.05 of g0, and the failure signature absent: fresh-sample value EV after the
  policy starts moving stays within 0.03 of the warm-up level (attempts 1–2 fell 0.05–0.13) and
  search-differs stays ≤ 25%. Failure: the heuristic stop rule fires again, or EV falls as before.
- Decision: go on to the held-out protocol (§7, 8i) and the exploiter test; or, on failure,
  pool diversification (the next approved direction) before more AZ.

| Order | Experiment | Cost | Success criterion | Feeds | Drop if |
|---|---|---|---|---|---|
| 1 | A: value-only, v2/v3 × mean/leader/rank | ~25 min | v3 − v2 ≥ 0.05 R² overall and in q3 or q4 | features for C; size of the target-noise effect | v3 − v2 < 0.02 everywhere → drop direction 2 |
| 2 | A2: search with A's value nets | ~30 min | paired diff vs confirm ref > 0, CI excludes 0 | value init/target for C | leader ≤ mean in A and A2 → drop direction 3 |
| 3 | B: opponent model, frozen `ppo-long` | ~40 min | paired diff ≳ +0.04 (CI excludes 0), LB > 0.34, heuristic within 0.03 | search setting for C | best ≤ +0.01 or CI includes 0 after 800 more games → drop direction 1 |
| 4 | C: AZ attempt 3 | 90 min | raw vs 3 `ppo-long` LB > 0.25, heuristic −0.05 max, EV holds | go to held-out protocol | heuristic stop rule fires or EV falls as before |
