# Trained models

`manifest.yaml` is the allow-list of nets that can be used as `net:<name>` seats. A seat type is
resolved only through it; paths are never accepted. `file` is relative to this directory and
`features_version` must match `train.features.FEATURES_VERSION`.

## ppo-smoke

A 20-minute smoke model, to be replaced after the long run (Phase 4 unit 8).

- Source: `runs/smoke_gate/ckpt_426.npz` (update 426), trained with `tools/train_ppo.py`; the run
  arguments are in `runs/smoke_gate/args.json`: players 4, hidden 512,512, lr 3e-4, 128 games per
  update, 8 workers, mix 0.7 annealed over 300 updates, league 0.5/0.3/0.2, seed 0, max 20 minutes.
- Strength: beat 3 heuristic seats 0.93 [0.86, 0.97] over 100 games at 4 players.
- Trained at 4 players only; 3 and 5 players are legal but untuned.
