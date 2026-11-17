# pyright: basic, reportMissingImports=false
"""Phase 4 unit 2: BusNet, npz export, numpy forward."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bots.net_forward import NetWeights, StaleModelError, layout_hash
from bus.actions import NUM_DISTINCT_ACTIONS
from train.features import MAX_SEATS, OBS_SIZE

HIDDEN = (32, 16)


def _write_npz(path: Path, rng: np.random.Generator, **meta_over: Any) -> None:
    """Hand-built file in the export format (no torch needed)."""
    sizes = (OBS_SIZE, *HIDDEN)
    arrays: dict[str, Any] = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        arrays[f"w{i}"] = rng.normal(size=(a, b)).astype(np.float32) * 0.1
        arrays[f"b{i}"] = rng.normal(size=b).astype(np.float32) * 0.1
    arrays["pi_w"] = rng.normal(size=(sizes[-1], NUM_DISTINCT_ACTIONS)).astype(np.float32)
    arrays["pi_b"] = np.zeros(NUM_DISTINCT_ACTIONS, np.float32)
    arrays["v_w"] = rng.normal(size=(sizes[-1], MAX_SEATS)).astype(np.float32)
    arrays["v_b"] = np.zeros(MAX_SEATS, np.float32)
    meta: dict[str, Any] = {
        "extra": {}, "activation": "tanh", "hidden": list(HIDDEN), "n_layers": len(HIDDEN),
        "obs_size": OBS_SIZE, "n_actions": NUM_DISTINCT_ACTIONS, "n_seats": MAX_SEATS,
        "layout_hash": layout_hash(),
    }  # fmt: skip
    meta.update(meta_over)
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def _masks(rng: np.random.Generator, b: int) -> np.ndarray[Any, Any]:
    m = rng.random((b, NUM_DISTINCT_ACTIONS)) < 0.1
    m[:, 0] = True
    return m


def test_load_and_forward_shapes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    _write_npz(tmp_path / "m.npz", rng)
    w = NetWeights.load(tmp_path / "m.npz")
    obs = rng.random((4, OBS_SIZE), dtype=np.float32)
    mask = _masks(rng, 4)
    logits, vals = w.forward(obs, mask)
    assert logits.shape == (4, NUM_DISTINCT_ACTIONS) and vals.shape == (4, MAX_SEATS)
    probs, _ = w.policy(obs, mask)
    assert np.allclose(probs.sum(-1), 1.0, atol=1e-5) and (probs[~mask] == 0).all()
    l1, v1 = w.forward(obs[0], mask[0])  # single observation, no batch dim
    assert l1.shape == (NUM_DISTINCT_ACTIONS,) and np.allclose(v1, vals[0], atol=1e-5)


def test_load_rejects_mismatched_layout(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    _write_npz(tmp_path / "a.npz", rng, obs_size=OBS_SIZE + 1)
    with pytest.raises(StaleModelError, match="obs_size"):
        NetWeights.load(tmp_path / "a.npz")
    _write_npz(tmp_path / "b.npz", rng, layout_hash="deadbeef")
    with pytest.raises(StaleModelError, match="layout hash"):
        NetWeights.load(tmp_path / "b.npz")


def test_net_forward_does_not_import_torch() -> None:
    code = (
        "import sys, bots.net_forward as m; m.layout_hash();"
        "sys.exit(1 if any(k == 'torch' or k.startswith('torch.') for k in sys.modules) else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---- torch-dependent ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_mod() -> Any:
    return pytest.importorskip("torch")


@pytest.mark.parametrize("batch", [1, 3, 32])
@pytest.mark.parametrize("hidden", [(64, 64), (512, 512)])
def test_torch_numpy_agree(torch_mod: Any, tmp_path: Path, batch: int, hidden: tuple[int, ...]):
    from train.export import export_npz
    from train.model import BusNet

    torch_mod.manual_seed(batch)
    net = BusNet(hidden=hidden)
    # Non-trivial heads: default init has tiny policy weights.
    torch_mod.nn.init.normal_(net.pi.weight, std=0.3)
    export_npz(net, tmp_path / "n.npz", {"name": "t"})
    w = NetWeights.load(tmp_path / "n.npz")
    assert w.meta["extra"] == {"name": "t"} and w.meta["hidden"] == list(hidden)
    rng = np.random.default_rng(batch)
    obs = rng.random((batch, OBS_SIZE), dtype=np.float32)
    mask = _masks(rng, batch)
    with torch_mod.no_grad():
        tl, tv = net(torch_mod.from_numpy(obs), torch_mod.from_numpy(mask))
    nl, nv = w.forward(obs, mask)
    legal = mask
    assert np.allclose(tl.numpy()[legal], nl[legal], atol=1e-5)
    assert (nl[~legal] <= -1e7).all() and (tl.numpy()[~legal] <= -1e7).all()
    assert np.allclose(tv.numpy(), nv, atol=1e-5)
    assert np.abs(nv).max() <= 1.0


def test_masked_sampling_never_illegal(torch_mod: Any) -> None:
    from train.model import BusNet

    torch_mod.manual_seed(0)
    net = BusNet(hidden=(64,))
    torch_mod.nn.init.normal_(net.pi.weight, std=1.0)
    rng = np.random.default_rng(1)
    obs = torch_mod.from_numpy(rng.random((256, OBS_SIZE), dtype=np.float32))
    mask_np = _masks(rng, 256)
    mask = torch_mod.from_numpy(mask_np)
    for _ in range(5):
        a, logp, ent, vals = net.act(obs, mask)
        assert mask_np[np.arange(256), a.numpy()].all()
        assert torch_mod.isfinite(logp).all() and (ent >= 0).all() and vals.shape == (256, 5)
    a, _, _, _ = net.act(obs, mask, greedy=True)
    assert mask_np[np.arange(256), a.numpy()].all()
    # Only one legal action: forced, zero entropy.
    one = torch_mod.zeros_like(mask)
    one[:, 7] = True
    a, logp, ent, _ = net.act(obs, one)
    assert (a == 7).all() and torch_mod.allclose(ent, torch_mod.zeros_like(ent), atol=1e-6)
