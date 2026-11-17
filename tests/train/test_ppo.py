# pyright: basic, reportMissingImports=false
"""Phase 4 unit 3: GAE, rollouts, schedules, tiny training run."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bots.net_forward import NetWeights
from train.features import FEATURES_VERSION, MAX_SEATS
from train.rollout import END_TYPES, RolloutConfig, gae, play_games


def test_gae_hand_computed() -> None:
    # gamma=0.5, lam=0.5; rewards 0,0,1; values .1,.2,.3; V(after last)=0
    adv, ret = gae(np.array([0, 0, 1.0]), np.array([0.1, 0.2, 0.3]), 0.5, 0.5)
    d2 = 1.0 - 0.3
    d1 = 0.0 + 0.5 * 0.3 - 0.2
    d0 = 0.0 + 0.5 * 0.2 - 0.1
    a2 = d2
    a1 = d1 + 0.25 * a2
    a0 = d0 + 0.25 * a1
    assert np.allclose(adv, [a0, a1, a2], atol=1e-6)
    assert np.allclose(ret, adv + np.array([0.1, 0.2, 0.3]), atol=1e-6)


def test_gae_lambda_one_is_monte_carlo() -> None:
    v = np.array([0.3, -0.2, 0.5, 0.1])
    adv, ret = gae(np.array([0, 0, 0, 0.7]), v, 1.0, 1.0)
    assert np.allclose(ret, 0.7, atol=1e-6)


def test_mix_anneal_schedule() -> None:
    from train.ppo import linear, mix_at

    assert mix_at(0, 0.7, 10) == pytest.approx(0.7)
    assert mix_at(5, 0.7, 10) == pytest.approx(0.85)
    assert mix_at(10, 0.7, 10) == pytest.approx(1.0)
    assert mix_at(99, 0.7, 10) == pytest.approx(1.0)
    assert mix_at(50, 0.7, 0) == 0.7
    assert linear(1.0, 0.0, 5, 10) == pytest.approx(0.5)


def test_features_version_in_layout_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    import train.features as F
    from bots.net_forward import layout_hash

    h = layout_hash()
    monkeypatch.setattr(F, "FEATURES_VERSION", FEATURES_VERSION + 1)
    assert layout_hash() != h


def _random_weights(hidden: tuple[int, ...] = (16,)) -> NetWeights:
    torch = pytest.importorskip("torch")
    from train.model import BusNet
    from train.ppo import net_to_weights

    torch.manual_seed(0)
    return net_to_weights(BusNet(hidden=hidden))


@pytest.mark.parametrize("players", [(4,), (3, 5)])
def test_rollout_consistent(players: tuple[int, ...]) -> None:
    w = _random_weights()
    cfg = RolloutConfig(players=players, mix=0.7, lam=1.0, gamma=1.0, envs_per_worker=3)
    res = play_games(w, 4, 123, cfg)
    st, s = res["stats"], res["samples"]
    assert st["games"] == 4 and st["illegal"] == 0
    assert sum(st["lengths"]) == st["decisions"]
    assert set(st["end"]) == set(END_TYPES)
    n = len(s["action"])
    for k in ("obs", "mask", "logp", "value", "adv", "vtarget", "vmask"):
        assert len(s[k]) == n
    assert s["mask"][np.arange(n), s["action"]].all()
    assert (s["logp"] <= 1e-6).all()
    # One recorded win_loss return per seat per game, from the seat counts used.
    assert all(n_ in players for _, n_ in st["ego_returns"])
    assert 4 * min(players) <= len(st["ego_returns"]) <= 4 * max(players)
    present = s["vmask"].sum(axis=1)
    assert set(np.unique(present)) <= {3.0, 4.0, 5.0}
    assert s["vtarget"].shape == (n, MAX_SEATS)
    assert (s["vtarget"][s["vmask"] == 0] == 0).all()
    # lam=gamma=1: adv + value is the terminal return of that seat (entry 0 of vtarget), so
    # every step of a trajectory shares one return value, in [-1, 1].
    mc = s["adv"] + s["value"]
    assert np.allclose(mc, s["vtarget"][:, 0], atol=1e-4)
    # Win/loss alone: terminal returns of a game sum to zero over its seats.
    assert abs(sum(r for r, _ in st["ego_returns"])) < 1e-4


def test_rollout_deterministic() -> None:
    w = _random_weights()
    cfg = RolloutConfig(players=(4,), envs_per_worker=2)
    a = play_games(w, 2, 7, cfg)
    b = play_games(w, 2, 7, cfg)
    assert np.array_equal(a["samples"]["action"], b["samples"]["action"])
    assert np.array_equal(a["samples"]["adv"], b["samples"]["adv"])


def test_eval_mode_collects_nothing_and_pins_ego() -> None:
    w = _random_weights()
    cfg = RolloutConfig(players=(4,), mix=1.0, collect=False)
    res = play_games(w, 4, 5, cfg, egos=[0, 1, 2, 3])
    assert res["samples"] is None
    assert len(res["stats"]["ego_returns"]) == 4  # one net seat per game


def test_help() -> None:
    pytest.importorskip("torch")
    r = subprocess.run(
        [sys.executable, "-m", "tools.train_ppo", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0 and "--games-per-update" in r.stdout


def _run(tmp: Path, name: str, workers: int) -> list[dict[str, str]]:
    from tools.train_ppo import main

    rc = main(
        [
            "--name", name, "--out-dir", str(tmp), "--updates", "2", "--games-per-update", "4",
            "--workers", str(workers), "--hidden", "32,32", "--device", "cpu", "--eval-every", "2",
            "--eval-games", "4", "--ckpt-every", "2", "--seed", "3", "--envs-per-worker", "2",
        ]
    )  # fmt: skip
    assert rc == 0
    with open(tmp / name / "log.csv") as fh:
        return list(csv.DictReader(fh))


def test_smoke_training_run(tmp_path: Any) -> None:
    pytest.importorskip("torch")
    from tools.train_ppo import COLUMNS

    rows = _run(tmp_path, "a", workers=2)
    assert len(rows) == 2 and list(rows[0].keys()) == COLUMNS
    assert all(int(r["illegal"]) == 0 for r in rows)
    assert rows[1]["eval_games"] == "4" and rows[0]["eval_games"] == ""
    run = tmp_path / "a"
    assert (run / "ckpt_2.pt").exists()
    w = NetWeights.load(run / "ckpt_2.npz")
    assert w.meta["extra"]["update"] == 2 and "mix" in w.meta["extra"]


def test_training_reproducible(tmp_path: Any) -> None:
    pytest.importorskip("torch")
    a = _run(tmp_path, "x", workers=0)
    b = _run(tmp_path, "y", workers=0)
    keys = ["decisions", "samples", "entropy", "approx_kl", "value_loss", "pg_loss", "mean_len"]
    for ra, rb in zip(a, b, strict=True):
        assert {k: ra[k] for k in keys} == {k: rb[k] for k in keys}
