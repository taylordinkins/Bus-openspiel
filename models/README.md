# Trained models

`manifest.yaml` is the allow-list of nets that can be used as `net:<name>` seats. A seat type is
resolved only through it; paths are never accepted. `file` is relative to this directory and
`features_version` must match `train.features.FEATURES_VERSION`.

## ppo-long

The long-run PPO model (replaces the smoke model as the playable seat).

- Source: `runs/long/best.npz` (= `best_1320.npz`, update 1320, composite 0.793 by the run's
  best-model criterion), trained with `tools/train_ppo.py` using league/PFSP self-play, a separate
  value trunk (export format 2), hidden 512,512, mixed 3/4/5 players; arguments in
  `runs/long/args.json`. The run was stopped around update 1360 of 1700.
- Strength (in-run eval, sampled play): about 0.98 vs 3 random; vs 3 heuristic about 0.97 / 0.85 /
  0.80 at 3 / 4 / 5 players (update 1360); about chance against its own earlier snapshots.
- `mcts_net:<sims>-ppo-long` seats work but play almost the same moves as the plain net (see
  `runs/eval_search/`), at about 160 ms per move for 50 sims.

## ppo-smoke

The 20-minute smoke model (`runs/smoke_gate/ckpt_426.npz`). Superseded by ppo-long but still listed in the manifest, because the training league uses it as its permanent hall-of-fame reference
(`--hof-seed ppo-smoke`) and tests load it directly.
