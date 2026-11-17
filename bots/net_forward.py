"""Numpy-only forward pass of a trained BusNet (Phase 4 unit 2).

The server and bots run trained nets through this module, so it must never import torch (directly
or through what it imports). Weights come from an ``.npz`` written by ``train.export.export_npz``.
Layout: ``meta`` (JSON string), then per trunk layer ``w{i}`` (in, out) and ``b{i}``, then
``pi_w``/``pi_b`` (policy head) and ``v_w``/``v_b`` (value head, tanh-bounded).

Format versions (``meta["format_version"]``, absent = 1): 1 has one trunk shared by both heads;
2 (unit 8b) adds a separate value trunk ``vw{i}``/``vb{i}`` (``n_v_layers`` of them, fed by the
observation, same activation) in front of the value head. Version-1 files still load and run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bus.actions import NUM_DISTINCT_ACTIONS

MASK_FILL = -1e8  # same large negative fill as the torch side (train.model.MASK_FILL)
ACTIVATIONS = ("tanh", "relu")
EXPORT_FORMAT = 2
Array = np.ndarray[Any, Any]


class StaleModelError(ValueError):
    """The model file does not match the current observation layout or action space."""


def layout_hash() -> str:
    """Hash of the observation layout (names, shapes), action count and FEATURES_VERSION."""
    from train.features import FEATURES_VERSION, obs_layout  # lazy: no torch involved

    text = json.dumps([[k, list(v)] for k, v in obs_layout().items()])
    text += f"|{NUM_DISTINCT_ACTIONS}|v{FEATURES_VERSION}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class NetWeights:
    layers: tuple[tuple[Array, Array], ...]  # trunk (w (in, out), b)
    pi: tuple[Array, Array]
    v: tuple[Array, Array]
    activation: str
    obs_size: int
    n_actions: int
    n_seats: int
    meta: dict[str, Any]
    v_layers: tuple[tuple[Array, Array], ...] = ()  # separate value trunk (format 2); () = shared

    @classmethod
    def load(cls, path: str | Path, *, check_layout: bool = True) -> NetWeights:
        with np.load(path, allow_pickle=False) as z:
            meta: dict[str, Any] = json.loads(str(z["meta"]))
            n = int(meta["n_layers"])
            layers = tuple((z[f"w{i}"], z[f"b{i}"]) for i in range(n))
            pi = (z["pi_w"], z["pi_b"])
            v = (z["v_w"], z["v_b"])
            nv = int(meta.get("n_v_layers", 0))
            v_layers = tuple((z[f"vw{i}"], z[f"vb{i}"]) for i in range(nv))
        if meta["activation"] not in ACTIVATIONS:
            raise StaleModelError(f"unknown activation {meta['activation']!r}")
        if check_layout:
            from train.features import OBS_SIZE

            if meta["obs_size"] != OBS_SIZE or meta["n_actions"] != NUM_DISTINCT_ACTIONS:
                raise StaleModelError(
                    f"model expects obs_size={meta['obs_size']}, n_actions={meta['n_actions']}; "
                    f"current are {OBS_SIZE}, {NUM_DISTINCT_ACTIONS}: retrain or re-export"
                )
            if meta["layout_hash"] != layout_hash():
                raise StaleModelError(
                    f"model observation layout hash {meta['layout_hash']} != current "
                    f"{layout_hash()}: the features changed since training"
                )
        return cls(
            layers, pi, v, meta["activation"], int(meta["obs_size"]), int(meta["n_actions"]),
            int(meta["n_seats"]), meta, v_layers,
        )  # fmt: skip

    def _trunk(self, x: Array, layers: tuple[tuple[Array, Array], ...]) -> Array:
        for w, b in layers:
            x = x @ w + b
            x = np.tanh(x) if self.activation == "tanh" else np.maximum(x, 0.0)
        return x

    def logits(self, obs: Array, mask: Array) -> Array:
        """Masked policy logits only (skips the value head and any separate value trunk)."""
        x = self._trunk(np.asarray(obs, np.float32), self.layers)
        return np.where(np.asarray(mask, bool), x @ self.pi[0] + self.pi[1], np.float32(MASK_FILL))

    def forward(self, obs: Array, mask: Array) -> tuple[Array, Array]:
        """(masked logits (..., n_actions), values (..., n_seats)); batch dims optional."""
        x0 = np.asarray(obs, np.float32)
        x = self._trunk(x0, self.layers)
        logits = np.where(
            np.asarray(mask, bool), x @ self.pi[0] + self.pi[1], np.float32(MASK_FILL)
        )
        xv = self._trunk(x0, self.v_layers) if self.v_layers else x
        return logits, np.tanh(xv @ self.v[0] + self.v[1])

    def policy(self, obs: Array, mask: Array) -> tuple[Array, Array]:
        """(probabilities over actions, values); illegal actions get exactly 0."""
        logits, values = self.forward(obs, mask)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        e = e * np.asarray(mask, bool)
        return e / e.sum(axis=-1, keepdims=True), values
