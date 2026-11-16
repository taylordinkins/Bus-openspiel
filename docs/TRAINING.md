# Training runs (PPO, Phase 4)

Everything here is about `tools/train_ppo.py`. Install the training group first:
`uv sync --group train`.

## Start, stop, resume

```
uv run python -m tools.train_ppo --name long --players 4 --updates 400 --workers 8 \
    --ckpt-every 10 --eval-every 10
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
--ent-coef-final --players --league --snapshot-every --pool-size --updates --device --stop-after
--task-timeout --max-retries`. Anything else (`--hidden`, `--seed`, `--name`, `--out-dir`,
`--games-per-update`, ...) that differs from the run is rejected with an error (exit code 2); so
is a checkpoint whose `features_version`, observation layout hash or hidden sizes do not match the
code. Changing `--lr`, `--lr-decay`, `--ent-coef*` or `--updates` changes the schedules from the
resume point on by design. Extending `--updates` is allowed and recorded.

Every resume appends an entry (time, update, checkpoint, old/new value of each override, argv) to
`resume_log.json`; `args.json` is updated and the very first one is kept as `args.original.json`.

## What is saved where (`runs/<name>/`)

| File | Content |
|---|---|
| `log.csv`, `eval.csv` | one row per update / per evaluated opponent; append only. On resume rows with `update` greater than the checkpoint's are dropped first, so there are no duplicates or gaps. `log.csv` has `rss_mb` (parent RSS). |
| `ckpt_<u>.pt` | model, optimizer, config and run state (RNGs, total games, elapsed time, best record) |
| `ckpt_<u>.npz` | numpy export for `bots.net_forward` / the server |
| `latest.json` | newest complete checkpoint; if missing or corrupt, resume scans for the newest `ckpt_*.pt` that loads |
| `best.pt`, `best.npz`, `best.json` | best model so far: highest Wilson lower bound of the win rate vs 3 heuristic seats (ties: win rate), taken from the periodic eval. Noisy with small `--eval-games`; unit 8b may replace the criterion. |
| `pool/snap_<id>.npz`, `pool/pool.json` | league snapshots and their index |
| `args.json`, `args.original.json`, `resume_log.json` | configuration history |

All of these are written atomically (tmp file, fsync, `os.replace`), so a crash or kill leaves
either the old or the new file, never a partial one.

## Retention

After each checkpoint the trainer keeps the last 3 `ckpt_*` pairs, every update that is a multiple
of `--keep-every` (default 250, 0 = none), and `best.*`; older pairs are deleted once the new one
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
