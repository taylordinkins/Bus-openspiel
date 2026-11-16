# pyright: basic, reportMissingImports=false
"""Export a trained BusNet to the numpy ``.npz`` format read by ``bots.net_forward``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bots.net_forward import layout_hash
from train.atomic import atomic_write
from train.model import BusNet


def export_npz(model: BusNet, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    """Write weights, activation, sizes and the observation layout hash; ``meta`` is extra info."""
    arrays: dict[str, Any] = {}

    def lin(name_w: str, name_b: str, layer: Any) -> None:
        arrays[name_w] = layer.weight.detach().cpu().numpy().T.astype(np.float32).copy()
        arrays[name_b] = layer.bias.detach().cpu().numpy().astype(np.float32)

    for i, layer in enumerate(model.trunk):
        lin(f"w{i}", f"b{i}", layer)
    lin("pi_w", "pi_b", model.pi)
    lin("v_w", "v_b", model.v)
    info = {
        "extra": meta or {},
        "activation": model.ACTIVATION,
        "hidden": list(model.hidden),
        "n_layers": len(model.trunk),
        "obs_size": model.obs_size,
        "n_actions": model.n_actions,
        "n_seats": model.n_seats,
        "layout_hash": layout_hash(),
    }
    atomic_write(Path(path), lambda f: np.savez(f, meta=np.array(json.dumps(info)), **arrays))
