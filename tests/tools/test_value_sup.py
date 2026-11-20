# pyright: basic, reportMissingImports=false
"""Supervised value path (Phase 4 interaction): labels with observations, value-only training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.noise_ceiling import net_ev
from tools.noise_ceiling import parse_args as nc_args
from tools.noise_ceiling import run as nc_run
from train import features as F


@pytest.fixture(scope="module")
def labels(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("labels")
    nc_run(nc_args(
        ["--positions", "16", "--k", "3", "--workers", "1", "--bootstrap", "5", "--save-obs",
         "--nets", "ppo-long=models/ppo-long.npz", "--out", str(out)]
    ))  # fmt: skip
    return out / "positions.npz"


def test_labels_carry_both_observation_versions(labels: Path) -> None:
    d = np.load(labels)
    assert d["obs_v2"].shape == (16, F.OBS_SIZE) and d["obs_v3"].shape == (16, F.OBS_SIZE_V3)
    np.testing.assert_array_equal(d["obs_v3"][:, : F.OBS_SIZE], d["obs_v2"])
    for q in ("R", "R_leader", "R_rank", "wl"):
        assert d[q].shape == (16, 5, 3)


def test_value_sup_smoke(labels: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from bots.net_forward import NetWeights
    from tools.value_sup import parse_args, run

    out = tmp_path / "sup"
    res = run(parse_args(
        ["--labels", str(labels), "--holdout", str(labels), "--features", "v2,v3",
         "--targets", "R,R_leader", "--epochs", "2", "--patience", "5", "--batch", "8",
         "--device", "cpu", "--out", str(out)]
    ))  # fmt: skip
    cells = {(c["features"], c["target"]): c for c in res["cells"]}
    assert set(cells) == {("v2", "R"), ("v3", "R"), ("v2", "R_leader"), ("v3", "R_leader")}
    # the start net is ppo-long in both versions (function-preserving widening)
    d = np.load(labels)
    for tgt in ("R", "R_leader"):
        a, b = cells[("v2", tgt)]["start"]["all"], cells[("v3", tgt)]["start"]["all"]
        assert a["ev_single"] == pytest.approx(b["ev_single"], abs=1e-4)
    # and it matches the noise ceiling's own scoring of ppo-long on R (ego seat)
    rows = np.arange(16)
    r = d["R"][rows, d["mover"]]
    want = net_ev(d["pred_ppo-long"][rows, d["mover"]], r)["ev_single"]
    assert cells[("v2", "R")]["start"]["all"]["ev_single"] == pytest.approx(want, abs=1e-4)
    for c in res["cells"]:
        assert len(c["history"]) <= 2 and set(c["best"]) == {"q1", "q2", "q3", "q4", "all"}
    # outputs: trained nets load as their version; the policy side equals ppo-long's
    base = NetWeights.load("models/ppo-long.npz")
    w3 = NetWeights.load(out / "v3_R_leader.npz")
    assert w3.features_version == F.FEATURES_V3
    np.testing.assert_array_equal(w3.pi[0], base.pi[0])
    np.testing.assert_array_equal(w3.layers[1][0], base.layers[1][0])
    np.testing.assert_array_equal(w3.layers[0][0][: F.OBS_SIZE], base.layers[0][0])
    s = json.loads((out / "summary.json").read_text())
    assert s["meta"]["k"] == 3
    assert (out / "summary.md").read_text().startswith("# Supervised value training")
