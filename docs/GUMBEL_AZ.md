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
