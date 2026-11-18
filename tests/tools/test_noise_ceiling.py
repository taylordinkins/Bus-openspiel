"""Noise-ceiling estimator (``tools.noise_ceiling``): synthetic data with a known ceiling, and a
tiny smoke run of the tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.noise_ceiling import bootstrap, ceiling, net_ev, parse_args, run


def _synthetic(p: int, k: int, sb: float, sw: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, sb, size=p)
    return mu, mu[:, None] + rng.normal(0.0, sw, size=(p, k))


def test_ceiling_recovers_known_value() -> None:
    # Var(mu) = 1, Var(noise) = 3: EV_max = 0.25. With K = 4 the uncorrected between-variance
    # would be 1 + 3/4 (EV ~0.37), so this also checks the finite-K correction.
    _, r = _synthetic(4000, 4, 1.0, 3.0**0.5, 0)
    c = ceiling(r)
    assert abs(c["within"] - 3.0) < 0.1
    assert abs(c["between"] - 1.0) < 0.1
    assert abs(c["ev_max"] - 0.25) < 0.02


def test_net_ev_single_and_conditional_mean() -> None:
    mu, r = _synthetic(4000, 8, 1.0, 3.0**0.5, 1)
    perfect = net_ev(mu, r)
    assert abs(perfect["ev_single"] - 0.25) < 0.02  # the Bayes value reaches the ceiling
    assert abs(perfect["r2_mean"] - 1.0) < 0.05
    const = net_ev(np.zeros_like(mu), r)
    assert abs(const["ev_single"]) < 1e-9 and abs(const["r2_mean"]) < 1e-9
    half = net_ev(mu + np.random.default_rng(2).normal(0.0, 1.0, size=mu.shape), r)
    assert abs(half["r2_mean"]) < 0.08  # MSE to mu = Var(mu): no reachable signal captured
    assert abs(half["ev_single"] - 0.0) < 0.03


def test_bootstrap_covers_truth_and_keeps_clusters() -> None:
    _, r = _synthetic(400, 32, 1.0, 3.0**0.5, 3)
    groups = np.repeat(np.arange(200), 2)
    ci = bootstrap(ceiling, groups, [r], 300, np.random.default_rng(0))["ev_max"]
    assert ci[0] < 0.25 < ci[1]
    assert ci[1] - ci[0] < 0.15


def test_tool_smoke(tmp_path: Path) -> None:
    out = tmp_path / "nc"
    args = parse_args(
        ["--positions", "8", "--k", "3", "--workers", "1", "--bootstrap", "20",
         "--nets", "ppo-long=models/ppo-long.npz", "--out", str(out)]
    )  # fmt: skip
    res = run(args)
    data = np.load(out / "positions.npz")
    assert data["R"].shape == (8, 5, 3) and data["pred_ppo-long"].shape == (8, 5)
    assert sorted(np.bincount(data["quartile"]).tolist()) == [2, 2, 2, 2]
    assert np.all(np.bincount(data["game"]) <= 2)
    # 4p: R and win_loss are zero-sum over the present seats, padding is NaN
    assert np.allclose(np.nansum(data["wl"], axis=1), 0.0)
    assert np.isnan(data["R"][:, 4]).all()
    assert (data["prog"] // 0.25 == data["quartile"]).all()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["summary"]["ego"]["R"]["all"]["units"] == 8
    assert "ppo-long" in res["summary"]["pooled"]["wl"]["q1"]
    assert (out / "summary.md").read_text().startswith("# Noise ceiling")
