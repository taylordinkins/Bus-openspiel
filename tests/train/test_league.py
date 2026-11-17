# pyright: basic, reportMissingImports=false
"""Phase 4 unit 4: opponent league sampling, snapshot pool, mixed-seat rollouts, eval."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bots.net_forward import NetWeights
from train.league import CURRENT, League, LeagueSpec, game_type, parse_league, sample_assignment
from train.rollout import RolloutConfig, play_games


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _weights(seed: int = 0, hidden: tuple[int, ...] = (16,)) -> NetWeights:
    torch = pytest.importorskip("torch")
    from train.model import BusNet
    from train.ppo import net_to_weights

    torch.manual_seed(seed)
    return net_to_weights(BusNet(hidden=hidden))


def test_parse_league() -> None:
    assert parse_league("1,0,0") == (1.0, 0.0, 0.0)
    assert parse_league("5,3,2") == pytest.approx((0.5, 0.3, 0.2))
    for bad in ("1,0", "-1,1,1", "0,0,0"):
        with pytest.raises(ValueError):
            parse_league(bad)


def test_proportions_and_current_seat() -> None:
    spec = LeagueSpec((0.5, 0.3, 0.2), (3, 6, 9), "uniform")
    rng = np.random.default_rng(1)
    types: Counter[str] = Counter()
    n_games = 6000
    for i in range(n_games):
        n = 3 + i % 3
        labels = sample_assignment(spec, n, rng)
        assert len(labels) == n and CURRENT in labels
        types[game_type(labels)] += 1
        for x in labels:
            assert x in (CURRENT, "heuristic", "random", "snap:3", "snap:6", "snap:9")
    assert types["self"] / n_games == pytest.approx(0.5, abs=0.03)
    assert types["snap"] / n_games == pytest.approx(0.3, abs=0.03)
    assert (types["heuristic"] + types["random"]) / n_games == pytest.approx(0.2, abs=0.03)
    assert types["heuristic"] > 0 and types["random"] > 0


def test_opponent_seats_randomised_and_empty_pool_fallback() -> None:
    rng = np.random.default_rng(0)
    spec = LeagueSpec((0.0, 0.0, 1.0), (), "uniform")
    seats = Counter()
    for _ in range(400):
        labels = sample_assignment(spec, 4, rng)
        assert game_type(labels) in ("heuristic", "random")
        seats.update(i for i, x in enumerate(labels) if x == CURRENT)
    assert all(seats[i] > 100 for i in range(4))  # the learner sits everywhere
    # snapshots-only league with an empty pool degrades to self-play
    assert sample_assignment(LeagueSpec((0.0, 1.0, 0.0), ()), 4, rng) == [CURRENT] * 4


def test_recency_sampling_prefers_newer() -> None:
    rng = np.random.default_rng(0)
    c: Counter[str] = Counter()
    for sampling in ("recency",):
        spec = LeagueSpec((0.0, 1.0, 0.0), (1, 2, 3, 4), sampling)
        for _ in range(2000):
            c.update(x for x in sample_assignment(spec, 4, rng) if x != CURRENT)
    assert c["snap:4"] > c["snap:1"] * 2


def test_assignment_deterministic() -> None:
    spec = LeagueSpec((0.4, 0.4, 0.2), (2, 4), "uniform")
    a = [sample_assignment(spec, 4, np.random.default_rng(5)) for _ in range(1)]
    b = [sample_assignment(spec, 4, np.random.default_rng(5)) for _ in range(1)]
    assert a == b
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    assert [sample_assignment(spec, 5, r1) for _ in range(50)] == [
        sample_assignment(spec, 5, r2) for _ in range(50)
    ]


def test_pool_eviction_and_files(tmp_path: Path) -> None:
    lg = League((0.5, 0.3, 0.2), pool_size=3, directory=tmp_path / "pool")
    w = _weights()
    for i in (10, 20, 30, 40, 50):
        evicted = lg.add(i, w)
        assert evicted == ([10] if i == 40 else [20] if i == 50 else [])
    assert lg.ids == [30, 40, 50] and len(lg) == 3
    files = sorted(p.name for p in (tmp_path / "pool").glob("snap_*.npz"))
    assert files == ["snap_30.npz", "snap_40.npz", "snap_50.npz"]
    loaded = NetWeights.load(tmp_path / "pool" / "snap_30.npz")
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(loaded.layers, w.layers, strict=True))
    assert set(lg.opponents()) == {"snap:30", "snap:40", "snap:50"}
    assert lg.spec().snap_ids == (30, 40, 50)
    with pytest.raises(ValueError):
        lg.add(30, w)


@pytest.mark.parametrize("via_path", [False, True])
def test_mixed_seat_rollout(tmp_path: Path, via_path: bool) -> None:
    w, snap = _weights(0), _weights(1)
    lg = League((0.0, 0.5, 0.5), 4, directory=tmp_path / "pool" if via_path else None)
    lg.add(7, snap)
    cfg = RolloutConfig(
        players=(4,), mix=0.7, lam=1.0, gamma=1.0, envs_per_worker=4, league=lg.spec()
    )
    res = play_games(w, 24, 11, cfg, opponents=lg.opponents())
    st, s = res["stats"], res["samples"]
    assert st["games"] == 24 and st["illegal"] == 0
    bt = st["by_type"]
    assert set(bt) <= {"snap", "heuristic", "random"} and len(bt) >= 2
    assert sum(v[0] for v in bt.values()) == 24
    seats = sum(v[1] for v in bt.values())
    assert seats == len(st["ego_returns"]) < 24 * 4  # opponent seats do not count
    assert s is not None and (s["mask"][np.arange(len(s["action"])), s["action"]]).all()
    # Monte Carlo return of each trained trajectory equals its terminal shaped return
    assert np.allclose(s["adv"] + s["value"], s["vtarget"][:, 0], atol=1e-4)
    # trajectories exist only for current seats: fewer samples than total decisions
    assert len(s["action"]) < st["decisions"]


@pytest.mark.parametrize("opp", ["random", "heuristic", "snap:7"])
def test_eval_opponents(opp: str) -> None:
    w, snap = _weights(0), _weights(1)
    cfg = RolloutConfig(players=(4,), mix=1.0, collect=False, eval_opponent=opp)
    res = play_games(w, 4, 3, cfg, egos=[0, 1, 2, 3], opponents={"snap:7": snap})
    assert res["samples"] is None and res["stats"]["illegal"] == 0
    assert len(res["stats"]["ego_returns"]) == 4


def test_league_smoke_run(tmp_path: Any) -> None:
    pytest.importorskip("torch")
    from tools.train_ppo import COLUMNS, EVAL_COLUMNS, main

    rc = main(
        [
            "--name", "lg", "--out-dir", str(tmp_path), "--updates", "3", "--games-per-update",
            "6", "--workers", "2", "--hidden", "32,32", "--device", "cpu", "--eval-every", "2",
            "--eval-games", "4", "--ckpt-every", "3", "--seed", "1", "--envs-per-worker", "3",
            "--snapshot-every", "1", "--pool-size", "2", "--league", "0.2,0.5,0.3",
        ]
    )  # fmt: skip
    assert rc == 0
    run = tmp_path / "lg"
    rows = _read(run / "log.csv")
    assert list(rows[0].keys()) == COLUMNS and len(rows) == 3
    assert all(int(r["illegal"]) == 0 for r in rows)
    for c in ("frac_self", "frac_snap", "frac_heuristic", "frac_random", "win_snap", "ret_random"):
        assert c in COLUMNS
    assert rows[2]["pool_size"] == "2"
    ev = _read(run / "eval.csv")
    assert list(ev[0].keys()) == EVAL_COLUMNS
    assert [e["opponent"] for e in ev if e["update"] == "2"][:2] == ["random", "heuristic"]
    assert any(e["opponent"].startswith("snap_") for e in ev)
    assert all(int(e["games"]) == 4 for e in ev)
    assert sorted(p.name for p in (run / "pool").glob("snap_*.npz")) == [
        "snap_2.npz", "snap_3.npz",
    ]  # fmt: skip
    w = NetWeights.load(run / "ckpt_3.npz")
    assert w.meta["extra"]["pool_ids"] == [2, 3]


def test_pure_selfplay_flag(tmp_path: Any) -> None:
    pytest.importorskip("torch")
    from tools.train_ppo import main

    rc = main(
        [
            "--name", "sp", "--out-dir", str(tmp_path), "--updates", "1", "--games-per-update",
            "4", "--workers", "0", "--hidden", "16", "--device", "cpu", "--eval-every", "0",
            "--league", "1,0,0", "--envs-per-worker", "2",
        ]
    )  # fmt: skip
    assert rc == 0
    rows = _read(tmp_path / "sp" / "log.csv")
    assert float(rows[0]["frac_self"]) == 1.0
    assert not (tmp_path / "sp" / "pool").exists()
