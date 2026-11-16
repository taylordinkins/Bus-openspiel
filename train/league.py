# pyright: basic, reportMissingImports=false
"""Opponent league for PPO self-play (Phase 4 unit 4). Numpy only: never imports torch.

Every training game gets one policy label per seat:

- ``"current"``: the net being trained; the only seats that produce training samples.
- ``"snap:<id>"``: a frozen past copy of the net (``id`` is the update it was taken after).
- ``"heuristic"`` / ``"random"``: the Phase 3 bots (``bots.registry`` seat types).

A game's *type* is drawn first: ``self`` (all seats current), ``snap`` (some seats play pool
snapshots, each seat picks its own snapshot) or ``bots`` (some seats play one fixed bot kind,
heuristic or random, chosen once per game). For the last two, the number of opponent seats is
uniform in ``1..n-1`` and the seats are random, so the learner sits at a random position and at
least one seat stays ``"current"``. With an empty pool the snapshot share is redistributed
proportionally over the other types.

``LeagueSpec`` is the small picklable description that workers sample from; ``League`` owns the
pool (bounded, oldest evicted) and writes each snapshot to ``<dir>/snap_<id>.npz`` so workers load
it by path (cached by id on their side) instead of receiving 6.6 MB per task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bots.net_forward import NetWeights, layout_hash

CURRENT = "current"
RANDOM = "random"
HEURISTIC = "heuristic"
SNAP_PREFIX = "snap:"
FIXED_BOTS = (HEURISTIC, RANDOM)
GAME_TYPES = ("self", "snap", "heuristic", "random")  # as reported by ``game_type``


def parse_league(text: str) -> tuple[float, float, float]:
    """``"0.5,0.3,0.2"`` -> normalised (self, snapshots, bots) probabilities."""
    parts = [float(x) for x in text.split(",") if x.strip()]
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError("league must be three non-negative numbers, e.g. 0.5,0.3,0.2")
    s = sum(parts)
    return (parts[0] / s, parts[1] / s, parts[2] / s)


@dataclass(frozen=True)
class LeagueSpec:
    probs: tuple[float, float, float] = (0.5, 0.3, 0.2)  # self, snapshots, fixed bots
    snap_ids: tuple[int, ...] = ()  # pool ids, oldest first
    sampling: str = "uniform"  # or "recency" (weight proportional to age rank + 1)


def sample_assignment(spec: LeagueSpec, n: int, rng: np.random.Generator) -> list[str]:
    """Per-seat labels for one game of ``n`` seats. Deterministic given ``rng``."""
    p = np.array(spec.probs, float)
    if not spec.snap_ids:
        p[1] = 0.0
    if p.sum() <= 0:  # e.g. league 0,1,0 with an empty pool: fall back to self-play
        return [CURRENT] * n
    kind = int(rng.choice(3, p=p / p.sum()))
    if kind == 0 or n < 2:
        return [CURRENT] * n
    k = int(rng.integers(1, n))  # opponent seats: 1..n-1, so a current seat always remains
    seats = rng.choice(n, size=k, replace=False)
    labels = [CURRENT] * n
    if kind == 1:
        w = np.ones(len(spec.snap_ids))
        if spec.sampling == "recency":
            w = np.arange(1, len(spec.snap_ids) + 1, dtype=float)
        picks = rng.choice(len(spec.snap_ids), size=k, p=w / w.sum())
        for s, j in zip(seats, picks, strict=True):
            labels[int(s)] = f"{SNAP_PREFIX}{spec.snap_ids[int(j)]}"
    else:
        bot = FIXED_BOTS[int(rng.integers(len(FIXED_BOTS)))]
        for s in seats:
            labels[int(s)] = bot
    return labels


def game_type(labels: list[str]) -> str:
    """``self``, ``snap``, ``heuristic`` or ``random`` (other labels count as their own name)."""
    others = [x for x in labels if x != CURRENT]
    if not others:
        return "self"
    if all(x.startswith(SNAP_PREFIX) for x in others):
        return "snap"
    return others[0]


def save_weights(w: NetWeights, path: Path, extra: dict[str, Any] | None = None) -> None:
    """Write ``w`` in the ``export_npz`` layout (loadable by ``NetWeights.load``)."""
    arrays: dict[str, Any] = {}
    for i, (wi, bi) in enumerate(w.layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = wi, bi
    arrays["pi_w"], arrays["pi_b"] = w.pi
    arrays["v_w"], arrays["v_b"] = w.v
    info = {
        "extra": extra or {}, "activation": w.activation,
        "hidden": [int(x[0].shape[1]) for x in w.layers], "n_layers": len(w.layers),
        "obs_size": w.obs_size, "n_actions": w.n_actions, "n_seats": w.n_seats,
        "layout_hash": layout_hash(),
    }  # fmt: skip
    np.savez(path, meta=np.array(json.dumps(info)), **arrays)


class League:
    """Bounded pool of frozen snapshots plus the sampling configuration."""

    def __init__(
        self,
        probs: tuple[float, float, float] = (0.5, 0.3, 0.2),
        pool_size: int = 10,
        sampling: str = "uniform",
        directory: Path | None = None,
    ) -> None:
        if sampling not in ("uniform", "recency"):
            raise ValueError("sampling must be uniform or recency")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.probs = probs
        self.pool_size = pool_size
        self.sampling = sampling
        self.directory = directory.resolve() if directory is not None else None
        self._ids: list[int] = []
        self._weights: dict[int, NetWeights] = {}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def ids(self) -> list[int]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def spec(self) -> LeagueSpec:
        return LeagueSpec(self.probs, tuple(self._ids), self.sampling)

    def sample_assignment(self, n: int, rng: np.random.Generator) -> list[str]:
        return sample_assignment(self.spec(), n, rng)

    def add(self, snap_id: int, weights: NetWeights) -> list[int]:
        """Freeze ``weights`` as ``snap_id``; returns the ids evicted (oldest first)."""
        if snap_id in self._weights:
            raise ValueError(f"snapshot {snap_id} already in the pool")
        self._ids.append(snap_id)
        self._weights[snap_id] = weights
        if self.directory is not None:
            save_weights(weights, self.directory / f"snap_{snap_id}.npz", {"snapshot": snap_id})
        evicted: list[int] = []
        while len(self._ids) > self.pool_size:
            old = self._ids.pop(0)
            del self._weights[old]
            evicted.append(old)
            if self.directory is not None:
                (self.directory / f"snap_{old}.npz").unlink(missing_ok=True)
        if self.directory is not None:
            (self.directory / "pool.json").write_text(
                json.dumps({"ids": self._ids, "pool_size": self.pool_size})
            )
        return evicted

    def label(self, snap_id: int) -> str:
        return f"{SNAP_PREFIX}{snap_id}"

    def opponents(self, ids: list[int] | None = None) -> dict[str, Any]:
        """``label -> npz path`` (cheap to pickle) when persisted, else the weights themselves."""
        out: dict[str, Any] = {}
        for i in self._ids if ids is None else ids:
            if self.directory is not None:
                out[self.label(i)] = str(self.directory / f"snap_{i}.npz")
            else:
                out[self.label(i)] = self._weights[i]
        return out
