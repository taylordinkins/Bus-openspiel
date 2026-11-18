# Training runs (PPO, Phase 4)

Everything here is about `tools/train_ppo.py`. Install the training group first:
`uv sync --group train`.

## Start, stop, resume

```
uv run python -m tools.train_ppo --name long --players 4 --updates 400 --workers 8 \
    --ckpt-every 10 --eval-every 10
```

Long run (stage A + stage B): `--updates` must be the total horizon of the whole run, set in stage
A; stage B (resume) changes only `--games-per-update` (and other whitelisted, schedule-neutral
options):

```
uv run python -m tools.train_ppo --name long --players 3,4,5 --updates 2000 --workers 8 \
    --games-per-update 64 --lr-decay --ent-coef 0.01 --ent-coef-final 0.002 --best-keep 5
uv run python -m tools.train_ppo --resume runs/long --games-per-update 256   # stage B
```

- `--updates` is the fixed horizon. The lr decay (`--lr-decay`), the entropy schedule
  (`--ent-coef-final`) and the reward-mix anneal are pure functions of the update counter and
  `--updates`, so stopping early (Ctrl-C, SIGTERM, `--max-minutes`, `--stop-after N`) never
  changes them.
- Stop with Ctrl-C or `kill -TERM <pid>`. The trainer finishes the current update (or abandons the
  rollout if that takes more than 60 s after the signal), saves a checkpoint and `latest.json`,
  removes its `/dev/shm/bus_rollout_<pid>_*` files, shuts the workers down, prints the resume
  command and exits 0. A second signal aborts at once (shm is still cleaned, no final checkpoint).
- `--max-minutes M` is a per-session wall-clock limit (also saves a checkpoint);
  `--stop-after N` stops after N updates of this session.
- Resume with `uv run python -m tools.train_ppo --resume runs/long`. The next update behaves as if
  the run had never stopped: model, optimizer, update counter, total games, torch (CPU, CUDA and
  the learner's generator), numpy and python RNG states, the league pool and the best-model record
  are restored. Rollout and evaluation seeds depend only on `(seed, update)`, so they need no
  state. With `--workers 0` on CPU a stop/resume run is bit-identical to a straight run
  (tested). Resuming a finished run does nothing.
- Starting a fresh run in a directory that already has checkpoints is refused (use `--resume` or
  a new `--name`).

## Resume overrides

`args.json` of the run is loaded; command-line options given next to `--resume` override it only
for this whitelist: `--max-minutes --workers --eval-every --eval-games --eval-mcts/--no-eval-mcts
--eval-greedy --ckpt-every --keep-every --lr --lr-decay/--no-lr-decay --ent-coef
--ent-coef-final --players --player-weights --league --snapshot-every --pool-size
--device --stop-after --task-timeout --max-retries --games-per-update --ratings-every
--ratings-games --aux-coef` (`--aux-coef` only between positive values; 0 <-> >0 adds or removes
model heads and is rejected, as is any `--reward` change). Anything else (`--hidden`, `--seed`, `--name`, `--out-dir`, `--value-trunk`, ...) that differs from the run is rejected with an error (exit code 2); so
is a checkpoint whose `features_version`, observation layout hash or hidden sizes do not match the
code. Changing `--lr`, `--lr-decay` or `--ent-coef*` changes the schedules from the resume point
on by design. `--updates` is the horizon of the cosine/linear LR and the entropy schedule, so
extending it would make the LR jump up: a changed `--updates` is rejected on resume unless
`--allow-horizon-change` is also given (logged as `horizon_change` in `resume_log.json`).

Every resume appends an entry (time, update, checkpoint, old/new value of each override, argv) to
`resume_log.json`; `args.json` is updated and the very first one is kept as `args.original.json`.

## What is saved where (`runs/<name>/`)

| File | Content |
|---|---|
| `log.csv`, `eval.csv` | one row per update / per evaluated opponent; append only. On resume rows with `update` greater than the checkpoint's are dropped first, so there are no duplicates or gaps. `log.csv` has `rss_mb` (parent RSS). |
| `ckpt_<u>.pt` | model, optimizer, config and run state (RNGs, total games, elapsed time, best record) |
| `ckpt_<u>.npz` | numpy export for `bots.net_forward` / the server |
| `latest.json` | newest complete checkpoint; if missing or corrupt, resume scans for the newest `ckpt_*.pt` that loads |
| `best.pt`, `best.npz`, `best.json` | copy of the top shortlist entry (below) |
| `best_<u>.pt/.npz`, `best_list.json` | top `--best-keep` (5) checkpoints by composite; the list holds update, composite, formula tag, parts and eval stats |
| `ratings.csv` | Elo ratings from the periodic round-robin (below) |
| `pool/snap_<id>.npz`, `pool/hof_<name>.npz`, `pool/pool.json` | recent snapshots, hall of fame, and their index (ids, hof entries, win-rate EMAs) |
| `args.json`, `args.original.json`, `resume_log.json` | configuration history |

All of these are written atomically (tmp file, fsync, `os.replace`), so a crash or kill leaves
either the old or the new file, never a partial one.

## Retention

After each checkpoint the trainer keeps the last 3 `ckpt_*` pairs, every update that is a multiple
of `--keep-every` (default 250, 0 = none), and `best*` files; older pairs are deleted once the new one
and `latest.json` are fully written. League snapshots follow `--pool-size` (oldest evicted).

## Workers and shared memory

Rollouts run in a spawn `ProcessPoolExecutor`; workers ignore Ctrl-C/SIGTERM (the trainer decides).
If a worker dies or a batch exceeds `--task-timeout` seconds (default 1800), the trainer warns,
rebuilds the pool and redoes the batch (up to `--max-retries`, default 3), then saves a checkpoint
and exits with code 1.

Workers spill samples to `bus_rollout_<trainer pid>_<random>` directories in `/dev/shm` (or the
system temp dir if it is small). A run removes its own directories when it ends and at startup
removes directories whose pid is dead (pid-less names from older versions after an hour). Directories
of live pids are never touched, so concurrent runs are safe.

## Seeds

Training game seeds are `seed * 1_000_003 + update * 1009 + league_seed * 7_919_011 + worker`;
evaluation seeds start at `2**62` plus a mixed-radix index of `(seed, update, opponent, chunk)`, so
the two spaces cannot overlap for any update count.

## League (unit 8b)

Games draw a type from `--league self,snapshots,bots` (default `0.4,0.42,0.18`). Bot games are
about 5/6 heuristic and 1/6 random, so heuristic is permanently ~15% and random ~3% of all games.
Snapshot seats are split `--hof-share` (0.5) from the hall of fame and the rest from the recent
pool (`--pool-size`, oldest evicted every `--snapshot-every` updates).

- **Hall of fame.** Every `--hof-every` (100) updates the current weights are stored as `hof:m<u>`.
  Entries listed in `--hof-seed` (default `ppo-smoke`, from `models/manifest.yaml`) are added at the
  start of a fresh run as permanent entries (`hof:ppo-smoke`); the old-format file is used as is.
  Over `--hof-cap` (20) entries, milestones are thinned: permanent entries, the first and the newest
  milestone stay, then the interior milestone whose neighbours are closest together is dropped
  first, so survivors stay spread over the run.
- **PFSP.** Within a source, a snapshot is drawn with weight `max((1 - w)^2, --pfsp-floor)`, where
  `w` is an EMA (`--pfsp-ema`, 0.2 per update) of the learner's training win rate in games that
  contained that label (unseen: 0.5). `--no-pfsp` gives uniform (or `recency`) weights.
- Worker caches hold every opponent net they have sampled (~6.6 MB each at 512x512), so
  recent + hof (up to 30) cost up to ~200 MB per worker.

## Network and schedules

- `--value-trunk separate` (default) gives the value head its own trunk (export format 2, npz meta
  `format_version`, `n_v_layers`, arrays `vw<i>`/`vb<i>`); `shared` is the original layout. Format-1
  files (`ppo-smoke.npz`) load unchanged in `bots.net_forward` and need no conversion. Default
  `--vf-coef` is 1.0 and `--gae-lambda` (alias `--lam`) 0.99.
- `--lr-decay` enables the LR schedule `--lr-schedule cosine` (default; or `linear`) from `--lr` to
  `--lr-final-frac` (0.1) times it over `--updates`. The entropy coefficient falls linearly from
  `--ent-coef` to the constant floor `--ent-coef-final` at `--ent-floor-at` (0.5) of `--updates`
  and stays there. Both are pure functions of the update counter and the horizon.
- `--players 3,4,5` samples player counts with `--player-weights` (default 0.3,0.4,0.3 for exactly
  3,4,5; otherwise uniform).
- Suggested long run: see the stage A / stage B commands at the top (`--updates` = whole run).

## Evaluation, ratings, best model

- `eval.csv` has one row per (opponent, player count): `--eval-games` games for each, so cost grows
  with the number of counts. Opponents: random, heuristic, optionally `mcts:50`, oldest/latest
  recent snapshot, and `hof_ref:<name>` (the oldest hall-of-fame entry, ppo-smoke by default).
  The `eval_win/lo/hi` columns of `log.csv` are the random rows pooled over counts.
- `log.csv` also has `ev_q1..ev_q4` (explained variance of the value head, by quarter of each
  seat's own decision sequence, first to last), `hof_size`, `eval_composite`, `rating_latest`.
- **Best model shortlist.** Each evaluation's composite may enter a shortlist of the top
  `--best-keep` (default 5) records; the lowest is evicted (its `best_<u>.*` deleted) and ties keep
  the older. The shortlist is in the checkpoint state and restored on resume (files of records made
  after the checkpoint are removed). Each record carries a `formula` tag: `h+hof` (reference net
  present) or `h-only`. Their scales differ, so composites are only compared within a formula and
  any `h+hof` record outranks any `h-only` one (a run normally uses one formula throughout).
- **Best model composite** (computed at each evaluation):
  `0.8 * H + 0.2 * R`, where `H` is the mean over the evaluated player counts of the win rate vs 3
  heuristic seats and `R` the same vs 3 copies of the hall-of-fame reference; without a reference,
  `H` alone. `best.json` stores the parts and the criterion text. (The rating round is too
  infrequent and noisy to pick checkpoints, so ratings are not part of it.)
- **ratings.csv** (every `--ratings-every` = 50 updates, `--ratings-games` = 300 games): entrants are
  the latest net, all hall-of-fame entries, heuristic and random. Each game draws a player count
  from `--players` and distinct random entrants for the seats. Seat `i` beats seat `j` when its
  final score is higher (tie = half each); the pairwise results are fitted with Bradley-Terry
  (MM iteration, prior of half a win each way per pair) and reported as `400 * log10(strength)`
  with heuristic = 1000. Columns: update, entrant, kind, rating, games (seats), score_rate.
  Ratings are per round, not accumulated, and noisy at 300 games.

## Exploiter test

Measures how exploitable a trained net (here `ppo-long`) is: warm-start a learner from it and
train only against frozen copies of it. `--init-from` loads the model weights of a trainer `.pt`
(fresh run only; fresh optimizer, update 0; exit code 2 on a hidden-size, value-trunk,
features-version or observation-layout mismatch; not combinable with `--resume`, and ignored when
a run resumes). `--opp-seats all` makes every non-self game give the opponents all n-1 seats, so
the learner holds exactly one randomly placed seat (default `random` keeps the 1..n-1 sampling;
run-defining, so it cannot change on resume). `runs/long` used the default `--hidden 512,512` and
`--value-trunk separate`, so no extra flags are needed.

```
uv run python -m tools.train_ppo --name exploit_long --players 3,4,5 --init-from runs/long/best_1320.pt \
  --league 0,1,0 --opp-seats all --hof-seed ppo-long --snapshot-every 0 --hof-every 0 --no-pfsp \
  --mix 1.0 --mix-anneal-updates 0 --lr 1e-4 --ent-coef 0.002 --updates 600 --max-minutes 90 \
  --ratings-every 0 --eval-every 10 --eval-games 200
```

With an empty snapshot pool and the single permanent hall-of-fame seed, every training game is one
learner against n-1 `ppo-long` seats, and `eval.csv` carries `hof_ref:ppo-long` at each player
count; its win rate rising well above the fair share (1/n) means the net is exploitable.

## Dense reward and aux heads (unit 8e)

`ppo-long` plateaued with critic EV ~0.25-0.30 on the single terminal return. Two opt-in changes
give the learner more signal; defaults are unchanged (`--reward terminal --aux-coef 0` is
bit-identical to before).

- `--reward dense` (run-defining; rejected on resume if changed; old runs resume as `terminal`).
  The margin part of the shaped return is paid out as potential differences. Per seat,
  `Phi = (own score - mean opponent score) / 10`, score = points - stones (the R-068 formula, used
  only as a training signal). At each of a seat's own decisions the rollout records `Phi` of all
  seats; reward `r_t = (1-mix) (Phi(s_{t+1}) - Phi(s_t))` where `s_{t+1}` is the state at the
  seat's next decision (so opponents' moves between decisions count), and the last reward adds
  `mix * win_loss` and closes on the terminal state. The sum telescopes exactly to
  `mix * win_loss + (1-mix) * (unclipped margin)`. Shaping is relative on purpose: passenger
  deliveries are the points, and taking a time stone costs the taker 1 to deny others points, so
  own-score-only shaping would wrongly penalise the stone.
- Value targets in dense mode are return-to-go per ego entry,
  `mix * wl_j + (1-mix) * (Phi_j(T) - Phi_j(t))`, clipped to [-1, 1] (tanh head); the clipped
  fraction is `vclip` in `log.csv`. Entry 0 equals the lambda=1 GAE return.
- `mix` now weights win/loss against the dense margin. Use `--mix 0.5 --mix-anneal-updates 0`
  (constant): annealing mix to 1.0 would switch the dense shaping off (user decision 2026-11-17).
- Use `--gae-lambda 0.95` with dense rewards (the default stays 0.99; gamma stays 1.0).
- `--aux-coef C` (> 0 builds training-only heads on the value trunk output: final score / 20 per
  ego seat (MSE), rank per ego seat (CE over the present rank classes, ties share the best rank),
  winner (masked softmax CE against a uniform distribution over the R-069/R-070 winner set)). The
  loss `C * (score + rank + win)` is added in `ppo_update`; terms are logged as `aux_score`,
  `aux_rank`, `aux_win`. Rollout workers never evaluate the heads; targets travel in the samples.
  `.pt` checkpoints include the heads; the npz export and `bots.net_forward` ignore them.
  `--init-from` a checkpoint without heads leaves the heads freshly initialised (all other keys
  still strict).
- Diagnostic: `stones_cur` / `stones_oth` in `log.csv` = mean `CLOCK_PAUSE` (time stone) takes per
  current / other seat per game.

Recommended dense run (exploiter-style warm start from the long run):

```
uv run python -m tools.train_ppo --name dense --init-from runs/long/best_1320.pt \
    --reward dense --gae-lambda 0.95 --mix 0.5 --mix-anneal-updates 0 --aux-coef 0.5 \
    --players 3,4,5 --updates 600 --workers 11 --lr 1e-4 --lr-decay
```

## Gumbel AlphaZero trainer (unit 8h)

`tools/train_az.py` (CLI; the trainer is `train/az_trainer.py`, actors `train/az_selfplay.py`,
replay `train/az_replay.py`, learner `train/az_learn.py`). Design: `docs/GUMBEL_AZ.md`; user
decisions Q-AZ1..Q-AZ7 in PLAN.md §9.

- Synchronous generations: `--workers` (default 6) numpy-only actor processes play
  `--games-per-gen` games with the current weights under Gumbel search (`bots.gumbel`,
  `--sims` 32, `--c-scale` 0.3, `--c-visit` 50, `--m` 16, `--gumbel-scale` 1), `--concurrent`
  games in flight per task and one batched (sparse first-layer) forward per search wave. Samples
  return through `/dev/shm` spill files. The CLI module imports no torch, so spawned actors stay
  at ~150 MB RSS each (the learner process with CUDA is ~1.6 GB).
- Targets: completed-Q improved policy at the root (policy CE); per-seat final
  R = `--mix` win_loss + (1 − `--mix`) relative margin (stones counted), ego order, present seats
  only (value MSE); `--value-target zq` uses (z + root search value) / 2 (off by default); aux
  heads at `--aux-coef` 0.25 (training only, not exported); AdamW with `--weight-decay` 1e-4.
  The first `--warmup-gens` (2) generations train the value side only (policy loss off).
- Learner: FIFO replay of `--replay-size` (500k) samples, `ceil(--reuse * new / --batch)` steps
  per generation (reuse 4, batch 1024), lr cosine `--lr` 3e-4 → `--lr-final` 1e-4 over
  `--generations`. `--min-replay` holds training until the buffer has that many samples.
- Tables: `--frozen-share` (0.2) of games seat a frozen net in 1..n−1 random seats: the
  `--frozen-nets` manifest names (`ppo-long`) plus the learner's own snapshot every
  `--snapshot-every` (10) generations (last `--snapshot-keep` 4 kept in `pool/`). Frozen seats play
  their raw policy sampled; only learner seats yield samples. 4p only by default (`--players`).
  `--heuristic-share` (default 0) of games, disjoint from the frozen ones, seat `bots.registry`
  heuristic bots in 1..n−1 random seats (learner in the rest; no search, no samples for the
  heuristic seats). One uniform draw per table picks heuristic / frozen / self, so the shares are
  shares of all tables (their sum must be ≤ 1). Both shares are run-defining: changing either on
  `--resume` is rejected.
- Evaluation every `--eval-every` (10) generations, generation 0 first (before training): raw
  policy, greedy (`--no-eval-greedy` to sample), no search, one seat vs 3 sampled copies of each
  `--eval-nets` entry (`ppo-long`, exploiter `runs/exploit_long/best.npz`) and vs 3 heuristic,
  seats rotated, `--eval-games` (200) per cell, shared wins credited 1/k, Wilson 95% intervals and
  mean score margin in `eval.csv`. `best.*` tracks the best win rate vs the first eval net.
- `log.csv` per generation: losses, value EV (own seat, overall and by game quartile, on the
  fresh samples before training), policy CE to the targets, target entropy, KL(π′‖prior),
  `search_differs` (share of moves where search overrode the prior's argmax: the R2 diagnostic),
  per table kind (self / frozen / heuristic) the share of games (`tables_*`) and the learner
  seats' mean win credit (`win_*`, 1/k for a k-way shared win), sims/s, games/h, RSS (parent and max actor),
  replay MB, timings.
- Stop/resume: `--max-minutes`, `--stop-after`, SIGINT/SIGTERM finish the generation, checkpoint
  and exit 0 (a second signal aborts). `--resume <run dir>` restores model, optimizer, generation,
  snapshot pool and best record; the replay is not saved and refills from the resume point (noted
  in `resume_log.json`). Overrides allowed on resume: `--max-minutes --workers --eval-every
  --eval-games --ckpt-every --keep-every --stop-after --task-timeout --max-retries --device
  --generations --games-per-gen --concurrent --lr --lr-final --reuse`.
- Exports (`ckpt_<gen>.npz`, `best.npz`) are the usual format-2 npz, usable as `net:`, `gumbel:`
  and `mcts_net:` seats once copied into `models/` and the manifest.

Recommended 8i run (90 minutes, warm start from `ppo-long`, 4p, n = 32, c_scale 0.3, the 8g
confirmation's choice):

```
uv run --group train python -m tools.train_az --name az1 --init-from runs/long/best_1320.pt \
    --players 4 --sims 32 --c-scale 0.3 --workers 6 --games-per-gen 256 --eval-every 10 \
    --eval-games 200 --max-minutes 90
```

Attempt 2 (after `runs/az1` was stopped at g18 for losing ground vs 3 heuristic, PLAN 8i):
heuristic tables, zq value targets, a lower constant lr and a heavier value loss:

```
uv run --group train python -m tools.train_az --name az2 --init-from runs/long/best_1320.pt \
    --players 4 --sims 64 --c-scale 0.1 --warmup-gens 10 --replay-size 300000 --workers 6 \
    --games-per-gen 128 --eval-every 4 --eval-games 200 --max-minutes 90 \
    --heuristic-share 0.15 --frozen-share 0.2 --value-target zq --lr 1e-4 --lr-final 1e-4 \
    --vf-coef 2.0
```

Interaction options (after attempt 2; design and experiments in `docs/GUMBEL_AZ.md` §10). All
are run-defining (a change on resume is rejected) and default to the attempt-2 behaviour:

- `--features v2|v3` (default v2): observation version (`train.features`). `--features v3` with
  a v2 `--init-from` checkpoint widens the first layers with zero columns for the new inputs, so
  generation 0 plays exactly like the checkpoint (`train.model.widen_model`). Frozen and eval
  nets keep their own observation version.
- `--value-quantity mean|leader|rank` (default mean): the relative term of R for z and for
  terminal search values (`bots.gumbel.relative_term`).
- `--opp-temp T`, `--opp-mix e`, `--opp-scope opponents|nonroot` (defaults 1, 0, opponents): the
  search's opponent model at non-root nodes (`bots.gumbel.opponent_prior`).
- Resume pitfall (attempt 2): `--stop-after` is saved in `args.json` and carries over on
  `--resume`; pass `--stop-after 0` explicitly when resuming a session that used it.

Supervised value path (no self-play): `tools.noise_ceiling --save-obs` labels positions with K
continuations and stores both observation versions; `tools.value_sup` trains the value trunk
only (policy frozen) on the K-mean of a target and reports held-out EV / R² against the ceiling:

```
uv run python -m tools.noise_ceiling --positions 16000 --k 8 --save-obs --nets "" \
    --bootstrap 20 --seed 201 --out runs/value_labels/train
uv run --group train python -m tools.value_sup --labels runs/value_labels/train/positions.npz \
    --holdout runs/value_labels/heldout/positions.npz --features v2,v3 --targets R,R_leader \
    --out runs/value_sup/a1
```

## Playing against the bots from the command line (`tools/agent_play.py`)

One decision per command, so an LLM agent (or a person) can play a full game against bot seats.
The game lives in a JSON file (config, action history, notes); every call rebuilds the state by
replaying the history, and bots are only queried for new moves (seeded from `--seed` and the
history length). Forced moves (one legal action) are applied automatically for every seat.

```bash
uv run python -m tools.agent_play new --players 4 --agent 0 --bots ppo-long --seed 7 --out /tmp/g.json
uv run python -m tools.agent_play show --game /tmp/g.json
uv run python -m tools.agent_play act 150 --game /tmp/g.json --note "why I chose it"
uv run python -m tools.agent_play notes --game /tmp/g.json
```

`--bots` takes a registry seat type (`heuristic`, `net:ppo-long`) or a bare model name
(`ppo-long` means `net:ppo-long`), either one for all bot seats or a comma list per bot seat.
`--agent` may list several seats. At game end the output shows the result and a replay file
`<game>.replay.json` is written, loadable by `tools.replay` and the web import.

### Map legend and `--summary` (for LLM players)

The state text uses bare node ids, so read the map once per game:

```bash
uv run python -m tools.agent_play legend      # or: uv run python -m tools.board_legend
uv run python -m tools.agent_play show --game /tmp/g.json --summary
uv run python -m tools.agent_play act 150 --game /tmp/g.json --note "why" --summary
```

`legend` lists every node (position, station/roundabout role, neighbours with street ids, the
building sites there), every street, and a key to the state fields (`line`, `waiting`, `sites`,
`occupied`, `board`, `am`, `lm`, `st`, ...). `--summary` (off by default, appended after the legal
actions) gives per seat: the line as a node path with its ends, buildings on it, passengers waiting
on and next to it, the legal extensions now (`bus.lines.legal_extensions`) and the Vrrooomm! moves
now (`bus.rules.vrrooomm_moves`). It also lists each seat's action markers left (`am`, already
reduced by markers placed this round, R-065) with the markers end condition from
`bus.rules.round_end_types` (R-066), and the Vrrooomm! targets if the clock advances (an estimate:
`advance_clock` + `move_passengers` on the current state, ignoring this round's other actions).
`show|act --board-compact` adds each action space with its letters in resolution order, occupants
and power (`marker_power`). Rejected actions print a `REJECTED: ...` line on stdout (exit 2, file
unchanged). `legend` accepts and ignores `--game`. All helpers are used read-only; no rule logic.

Recommended workflow: read `legend` once, then each turn call `show --summary` (or `act ... --summary`,
which already shows the next decision), pick an action id, and record why with `--note`.
