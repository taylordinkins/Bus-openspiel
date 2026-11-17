"""Env throughput: ``uv run python -m tools.bench_env [--games N] [--players P] [--seed S]``.

Random legal-action self-play through ``train.env.BusEnv``; reports decisions/s (step + obs +
mask), forced steps/s, games/s and the time split between engine step, obs and mask.
"""

from __future__ import annotations

import argparse
import time

from train.env import BusEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = BusEnv(a.players, seed=a.seed)
    t_step = t_obs = t_mask = 0.0
    decisions = forced = 0
    clock = time.perf_counter
    start = clock()
    for g in range(a.games):
        t0 = clock()
        env.reset(seed=a.seed + g)
        t_step += clock() - t0
        while not env.done:
            t0 = clock()
            env.obs()
            t1 = clock()
            mask = env.legal_mask()
            t2 = clock()
            action = env.rng.choice(env.legal_actions())
            env.step(action)
            t3 = clock()
            t_obs += t1 - t0
            t_mask += t2 - t1
            t_step += t3 - t2
            decisions += 1
            assert mask.any()
        forced += env.forced_steps
    wall = clock() - start
    print(f"players={a.players} games={a.games} wall={wall:.2f}s")
    print(
        f"decisions={decisions} ({decisions / wall:.0f}/s)  forced={forced} ({forced / wall:.0f}/s)"
    )
    print(f"games/s={a.games / wall:.2f}  decisions/game={decisions / a.games:.0f}")
    tot = t_step + t_obs + t_mask
    for name, t in (("engine step (incl. forced)", t_step), ("obs", t_obs), ("mask", t_mask)):
        print(f"  {name}: {t:.2f}s ({100 * t / tot:.0f}%)")


if __name__ == "__main__":
    main()
