"""Numpy-only forward pass of a trained BusNet (Phase 4 unit 2).

The server and bots run trained nets through this module, so it must never import torch (directly
or through what it imports). Weights come from an ``.npz`` written by ``train.export.export_npz``.
Layout: ``meta`` (JSON string), then per trunk layer ``w{i}`` (in, out) and ``b{i}``, then
``pi_w``/``pi_b`` (policy head) and ``v_w``/``v_b`` (value head, tanh-bounded).

Format versions (``meta["format_version"]``, absent = 1): 1 has one trunk shared by both heads;
2 (unit 8b) adds a separate value trunk ``vw{i}``/``vb{i}`` (``n_v_layers`` of them, fed by the
observation, same activation) in front of the value head. Version-1 files still load and run.

Observation versions: the stored ``layout_hash`` names the features version the net was trained
on (``train.features.FEATURE_VERSIONS``: 1 = observation v2, 2 = v3); ``NetWeights.encoder`` is
the matching encoder, so v2 nets (``ppo-long``, ``ppo-smoke``, ``runs/az*``) keep playing on v2
observations next to v3 nets.
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


def layout_hash(version: int | None = None) -> str:
    """Hash of the observation layout (names, shapes), action count and features version
    (default ``train.features.FEATURES_VERSION``, read at call time)."""
    from train import features  # lazy: no torch involved

    v = features.FEATURES_VERSION if version is None else version
    text = json.dumps([[k, list(sh)] for k, sh in features.obs_layout(v).items()])
    text += f"|{NUM_DISTINCT_ACTIONS}|v{v}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def version_for_layout(layout: str) -> int | None:
    """The known features version whose ``layout_hash`` is ``layout``, or None."""
    from train.features import FEATURE_VERSIONS

    return next((v for v in FEATURE_VERSIONS if layout_hash(v) == layout), None)


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
    features_version: int = 1  # observation the net reads (train.features.FEATURE_VERSIONS)

    @property
    def encoder(self) -> Any:
        """``core -> observation`` for this net's features version."""
        from train.features import encoder

        return encoder(self.features_version)

    def encode(self, core: Any) -> Array:
        return self.encoder(core)

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
        from train.features import FEATURES_VERSION, obs_size, version_for_obs_size

        version = version_for_layout(str(meta.get("layout_hash", "")))
        if check_layout:
            if version is None:
                raise StaleModelError(
                    f"model observation layout hash {meta['layout_hash']} matches no known "
                    f"features version (current default {layout_hash()}): the features changed "
                    "since training"
                )
            if meta["obs_size"] != obs_size(version) or meta["n_actions"] != NUM_DISTINCT_ACTIONS:
                raise StaleModelError(
                    f"model expects obs_size={meta['obs_size']}, n_actions={meta['n_actions']}; "
                    f"features v{version} has {obs_size(version)}, {NUM_DISTINCT_ACTIONS}: "
                    "retrain or re-export"
                )
        if version is None:
            version = version_for_obs_size(int(meta["obs_size"])) or FEATURES_VERSION
        return cls(
            layers, pi, v, meta["activation"], int(meta["obs_size"]), int(meta["n_actions"]),
            int(meta["n_seats"]), meta, v_layers, version,
        )  # fmt: skip

    def widened(self, version: int) -> NetWeights:
        """The same net reading features ``version``: first-layer rows copied to the matching
        input columns, zeros for the new ones (function-preserving warm start, numpy side of
        ``train.model.widen_model``)."""
        from dataclasses import replace

        from train.features import input_columns, obs_size

        cols = input_columns(self.features_version, version)
        size = obs_size(version)

        def wide(layers: tuple[tuple[Array, Array], ...]) -> tuple[tuple[Array, Array], ...]:
            if not layers:
                return layers
            (w0, b0), *rest = layers
            w = np.zeros((size, w0.shape[1]), w0.dtype)
            w[cols] = w0
            return ((w, b0), *rest)

        meta = {**self.meta, "obs_size": size, "layout_hash": layout_hash(version)}
        return replace(
            self, layers=wide(self.layers), v_layers=wide(self.v_layers), obs_size=size,
            features_version=version, meta=meta,
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
