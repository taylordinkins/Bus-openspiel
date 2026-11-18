"""Noise-ceiling labels for the value-target variants (Phase 4 interaction, direction 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.noise_ceiling import QUANTITIES, parse_args, run


def test_tool_labels_every_target_variant(tmp_path: Path) -> None:
    out = tmp_path / "nc"
    args = parse_args(
        ["--positions", "8", "--k", "2", "--workers", "1", "--bootstrap", "5",
         "--nets", "ppo-long=models/ppo-long.npz", "--out", str(out)]
    )  # fmt: skip
    run(args)
    data = np.load(out / "positions.npz")
    assert set(QUANTITIES) == {"R", "R_leader", "R_rank", "wl"}
    for q in QUANTITIES:
        assert data[q].shape == (8, 5, 2)
        assert np.isnan(data[q][:, 4]).all()
    wl = data["wl"][:, :4]
    for q in ("R", "R_leader", "R_rank"):
        rel = 2.0 * (data[q][:, :4] - 0.5 * wl)  # mix 0.5: the relative term
        assert np.all(np.abs(rel) <= 1.0 + 1e-9)
    # the rank term of 4 seats sums to 0; the leader term is <= 0 for all but at most one seat
    assert np.allclose((2.0 * (data["R_rank"][:, :4] - 0.5 * wl)).sum(axis=1), 0.0)
    lead = 2.0 * (data["R_leader"][:, :4] - 0.5 * wl)
    assert ((lead > 1e-9).sum(axis=1) <= 1).all()
    summary = json.loads((out / "summary.json").read_text())
    assert "R_leader" in summary["summary"]["ego"] and "R_rank" in summary["summary"]["pooled"]
