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
--ratings-games`. Anything else (`--hidden`, `--seed`, `--name`, `--out-dir`, `--value-trunk`, ...) that differs from the run is rejected with an error (exit code 2); so
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
