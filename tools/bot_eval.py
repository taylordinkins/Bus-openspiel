# pyright: basic
# ruff: noqa: E402
"""Candidate-vs-field evaluation with wasted-marker counts (Phase 4 bot work).

``uv run python -m tools.bot_eval --cand planner --opps net:ppo-long,net:ppo-long --games 30``

The candidate sits in every seat position in turn (game g puts it at seat g % players); the other
seats take the opponent list in order. A *wasted* marker is one that resolves with no effect for its
owner: Lines/Passengers/Buildings with power <= 0 or nothing to do, Vrrooomm! with 0 moves, Buses
with 5 buses already. Counted by wrapping the engine's resolution hooks in the worker process only
(the engine is not modified).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json
import multiprocessing as mp
import random
import time
from typing import Any

import pyspiel

import bus.game  # noqa: F401  # pyright: ignore[reportUnusedImport]
from bots.registry import make_bot, register
from bus import constants as C
from bus import rules


def _gumbel_sup_factory(param: str | None) -> Any:
    """Eval-only seat ``gumbel_sup:<R|R_leader|R_rank>``: the experiment A2 search player
    (``gumbel:64,c=0.1``, policy = ppo-long, value = the supervised K-mean value net)."""
    from pathlib import Path

    from bots.gumbel import GumbelBot, GumbelConfig
    from bots.net_forward import NetWeights

    if param not in ("R", "R_leader", "R_rank"):
        raise ValueError("gumbel_sup takes R, R_leader or R_rank")
    path = Path(__file__).resolve().parent.parent / f"runs/value_sup/a1/v2_{param}.npz"
    return GumbelBot(NetWeights.load(path), 64, GumbelConfig(c_scale=0.1), name=f"sup-{param}")


register("gumbel_sup", _gumbel_sup_factory)

_WASTE: list[int] = []
_COUNT = False


def _install_hooks(players: int) -> None:
    orig_enter = rules._enter  # pyright: ignore[reportPrivateUsage]
    orig_buses = rules._resolve_buses  # pyright: ignore[reportPrivateUsage]

    def enter(core: Any, action: int, letter: int) -> Any:
        node = orig_enter(core, action, letter)
        if _COUNT and node is None and action != rules.ACT_CLOCK:
            _WASTE[core.action_board[action][letter]] += 1
        return node

    def buses(core: Any, owner: int) -> Any:
        if _COUNT and core.seats[owner].buses >= C.BUSES_PER_PLAYER:
            _WASTE[owner] += 1
        return orig_buses(core, owner)

    rules._enter = enter  # pyright: ignore[reportPrivateUsage]
    rules._resolve_buses = buses  # pyright: ignore[reportPrivateUsage]


_HOOKED = False


def play(job: tuple[int, int, list[str]]) -> dict[str, Any]:
    global _HOOKED, _WASTE, _COUNT
    g, seed, types = job
    n = len(types)
    if not _HOOKED:
        _install_hooks(n)
        _HOOKED = True
    _WASTE = [0] * n
    rng = random.Random(seed * 1_000_003 + g)
    state = pyspiel.load_game("python_bus", {"players": n}).new_initial_state()
    bots = [make_bot(t) for t in types]
    for b in bots:
        if hasattr(b, "_tracker"):
            b._tracker = None  # pyright: ignore
    t0 = time.time()
    while not state.is_terminal():
        p = state.current_player()
        state.apply_action(bots[p].choose(state, rng))
    core = state.core  # type: ignore[attr-defined]
    # Replay the real game with counting on (bots' internal rollouts are not counted).
    _COUNT = True
    c = rules.initial_core(n)
    for act in state.history():
        c = rules.apply(c, act)
    _COUNT = False
    return {
        "g": g, "types": types, "scores": list(rules.final_scores(core)),
        "winners": list(rules.winners(core)), "waste": list(_WASTE), "secs": time.time() - t0,
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--opps", required=True, help="comma list, players-1 seat types")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    opps = a.opps.split(",")
    n = len(opps) + 1
    jobs = []
    for g in range(a.games):
        pos = g % n
        types = opps[:pos] + [a.cand] + opps[pos:]
        jobs.append((g, a.seed, types))
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(play, jobs, chunksize=1)
    win = 0.0
    waste = 0
    margin = 0.0
    per_seat = [[0.0, 0] for _ in range(n)]
    for r in res:
        pos = r["g"] % n
        w = (1 / len(r["winners"])) if pos in r["winners"] else 0.0
        win += w
        per_seat[pos][0] += w
        per_seat[pos][1] += 1
        waste += r["waste"][pos]
        sc = r["scores"]
        margin += sc[pos] - max(s for i, s in enumerate(sc) if i != pos)
    G = len(res)
    secs = sum(r["secs"] for r in res) / G
    seat_s = " ".join(f"P{i}:{w / max(c, 1):.2f}" for i, (w, c) in enumerate(per_seat))
    opp_waste = sum(sum(r["waste"]) - r["waste"][r["g"] % n] for r in res) / G / (n - 1)
    print(
        f"{a.cand} vs {a.opps}: win {win / G:.3f} ({win:.1f}/{G}) [{seat_s}] "
        f"margin {margin / G:+.2f} waste {waste / G:.2f} (opp {opp_waste:.2f}) {secs:.1f}s/game"
    )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
