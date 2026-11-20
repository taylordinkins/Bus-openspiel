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

Unit 8b: the snapshot share is split between the *recent* pool (``snap:<id>``, bounded, oldest
evicted) and a *hall of fame* (``hof:<name>``: milestone snapshots every N updates, never evicted
except by thinning once over a cap, plus permanent seed entries such as ``ppo-smoke``). Within each
source a snapshot is drawn with PFSP-style weight ``max((1 - w)^2, floor)`` where ``w`` is an EMA of
the learner's training win rate against that label (hard opponents are drawn more). The fixed bots
keep their own share of games (``probs[2]``, heuristic:random split ``bot_split``).

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

from bots.net_forward import EXPORT_FORMAT, NetWeights, layout_hash
from train.atomic import atomic_write, atomic_write_text

CURRENT = "current"
RANDOM = "random"
HEURISTIC = "heuristic"
SNAP_PREFIX = "snap:"
HOF_PREFIX = "hof:"
OPP_SEATS = ("random", "all")
FIXED_BOTS = (HEURISTIC, RANDOM)
GAME_TYPES = ("self", "snap", "heuristic", "random")  # as reported by ``game_type``


def parse_league(text: str) -> tuple[float, float, float]:
    """``"0.5,0.3,0.2"`` -> normalised (self, snapshots, bots) probabilities."""
    parts = [float(x) for x in text.split(",") if x.strip()]
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError("league must be three non-negative numbers, e.g. 0.5,0.3,0.2")
    s = sum(parts)
    return (parts[0] / s, parts[1] / s, parts[2] / s)


def pfsp_weight(win_rate: float, floor: float = 0.05) -> float:
    """PFSP "hard opponents first" weight ``max((1 - w)^2, floor)`` for training win rate ``w``."""
    return max((1.0 - win_rate) ** 2, floor)


@dataclass(frozen=True)
class LeagueSpec:
    probs: tuple[float, float, float] = (0.5, 0.3, 0.2)  # self, snapshots, fixed bots
    snap_ids: tuple[int, ...] = ()  # recent pool ids, oldest first
    sampling: str = "uniform"  # or "recency" (weight proportional to age rank + 1)
    hof_names: tuple[str, ...] = ()  # hall-of-fame entries (label ``hof:<name>``)
    hof_share: float = 0.5  # share of snapshot seats drawn from the hall of fame
    snap_w: tuple[float, ...] = ()  # extra (PFSP) weights parallel to snap_ids; () = none
    hof_w: tuple[float, ...] = ()  # same for hof_names
    bot_split: float = 5 / 6  # P(heuristic) among fixed-bot games (rest random): 15% vs 3%
    opp_seats: str = "random"  # "all": non-self games give opponents n-1 seats (one learner seat)


def _draw(rng: np.random.Generator, w: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
    return rng.choice(len(w), size=k, p=w / w.sum())


def sample_assignment(spec: LeagueSpec, n: int, rng: np.random.Generator) -> list[str]:
    """Per-seat labels for one game of ``n`` seats. Deterministic given ``rng``."""
    p = np.array(spec.probs, float)
    if not spec.snap_ids and not spec.hof_names:
        p[1] = 0.0
    if p.sum() <= 0:  # e.g. league 0,1,0 with an empty pool: fall back to self-play
        return [CURRENT] * n
    kind = int(rng.choice(3, p=p / p.sum()))
    if kind == 0 or n < 2:
        return [CURRENT] * n
    if spec.opp_seats == "all":  # exactly one learner seat, randomly placed
        me = int(rng.integers(0, n))
        seats = np.array([i for i in range(n) if i != me])
    else:
        k = int(rng.integers(1, n))  # opponent seats: 1..n-1, so a current seat always remains
        seats = rng.choice(n, size=k, replace=False)
    labels = [CURRENT] * n
    if kind == 1:
        w_rec = np.ones(len(spec.snap_ids))
        if spec.sampling == "recency":
            w_rec = np.arange(1, len(spec.snap_ids) + 1, dtype=float)
        if spec.snap_w:
            w_rec = w_rec * np.array(spec.snap_w, float)
        w_hof = np.array(spec.hof_w, float) if spec.hof_w else np.ones(len(spec.hof_names))
        for s in seats:
            use_hof = bool(spec.hof_names) and (
                not spec.snap_ids or (rng.random() < spec.hof_share)
            )
            if use_hof:
                labels[int(s)] = f"{HOF_PREFIX}{spec.hof_names[int(_draw(rng, w_hof, 1)[0])]}"
            else:
                labels[int(s)] = f"{SNAP_PREFIX}{spec.snap_ids[int(_draw(rng, w_rec, 1)[0])]}"
    else:
        bot = HEURISTIC if rng.random() < spec.bot_split else RANDOM
        for s in seats:
            labels[int(s)] = bot
    return labels


def game_type(labels: list[str]) -> str:
    """``self``, ``snap``, ``heuristic`` or ``random`` (other labels count as their own name)."""
    others = [x for x in labels if x != CURRENT]
    if not others:
        return "self"
    if all(x.startswith((SNAP_PREFIX, HOF_PREFIX)) for x in others):
        return "snap"
    return others[0]


def save_weights(w: NetWeights, path: Path, extra: dict[str, Any] | None = None) -> None:
    """Write ``w`` in the ``export_npz`` layout (loadable by ``NetWeights.load``)."""
    arrays: dict[str, Any] = {}
    for i, (wi, bi) in enumerate(w.layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = wi, bi
    for i, (wi, bi) in enumerate(w.v_layers):
        arrays[f"vw{i}"], arrays[f"vb{i}"] = wi, bi
    arrays["pi_w"], arrays["pi_b"] = w.pi
    arrays["v_w"], arrays["v_b"] = w.v
    info = {
        "extra": extra or {}, "activation": w.activation, "format_version": EXPORT_FORMAT,
        "hidden": [int(x[0].shape[1]) for x in w.layers], "n_layers": len(w.layers),
        "n_v_layers": len(w.v_layers),
        "obs_size": w.obs_size, "n_actions": w.n_actions, "n_seats": w.n_seats,
        "layout_hash": layout_hash(w.features_version),
    }  # fmt: skip
    atomic_write(path, lambda f: np.savez(f, meta=np.array(json.dumps(info)), **arrays))


def thin_hof(entries: list[dict[str, Any]], cap: int) -> list[str]:
    """Names to drop so that ``len(entries) <= cap`` (best effort), thinning milestones.

    Permanent entries, the first and the newest milestone are never dropped; among the other
    milestones the one whose neighbours are closest together goes first (ties: the earliest), so
    the survivors stay spread over the whole run. ``entries`` need ``name``, ``update``,
    ``permanent``. Pure: does not modify ``entries``.
    """
    ms = sorted((e for e in entries if not e["permanent"]), key=lambda e: e["update"])
    total = len(entries)
    gone: list[str] = []
    while total > cap and len(ms) > 2:
        spans = [ms[i + 1]["update"] - ms[i - 1]["update"] for i in range(1, len(ms) - 1)]
        j = 1 + spans.index(min(spans))
        gone.append(ms.pop(j)["name"])
        total -= 1
    return gone


class League:
    """Recent snapshot pool + hall of fame + PFSP statistics, and the sampling configuration."""

    def __init__(
        self,
        probs: tuple[float, float, float] = (0.5, 0.3, 0.2),
        pool_size: int = 10,
        sampling: str = "uniform",
        directory: Path | None = None,
        *,
        hof_cap: int = 20,
        hof_share: float = 0.5,
        pfsp: bool = True,
        pfsp_floor: float = 0.05,
        ema_alpha: float = 0.2,
        opp_seats: str = "random",
    ) -> None:
        if opp_seats not in OPP_SEATS:
            raise ValueError("opp_seats must be random or all")
        if sampling not in ("uniform", "recency"):
            raise ValueError("sampling must be uniform or recency")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.probs = probs
        self.pool_size = pool_size
        self.sampling = sampling
        self.opp_seats = opp_seats
        self.hof_cap, self.hof_share = hof_cap, hof_share
        self.pfsp, self.pfsp_floor, self.ema_alpha = pfsp, pfsp_floor, ema_alpha
        self.directory = directory.resolve() if directory is not None else None
        self._ids: list[int] = []
        self._weights: dict[int, NetWeights] = {}
        self._hof: list[dict[str, Any]] = []  # name, update, permanent (oldest first)
        self._hof_weights: dict[str, NetWeights] = {}
        self.win_ema: dict[str, float] = {}  # label -> EMA of the learner's training win rate
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def ids(self) -> list[int]:
        return list(self._ids)

    @property
    def hof_names(self) -> list[str]:
        return [e["name"] for e in self._hof]

    def hof_entries(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._hof]

    def __len__(self) -> int:
        return len(self._ids)

    def _w(self, label: str, base: float) -> float:
        if not self.pfsp:
            return base
        return base * pfsp_weight(self.win_ema.get(label, 0.5), self.pfsp_floor)

    def spec(self) -> LeagueSpec:
        # recency is applied by sample_assignment itself, so pass the PFSP factor alone
        snap_w = tuple(self._w(self.label(i), 1.0) for i in self._ids) if self.pfsp else ()
        hof_w = tuple(self._w(self.hof_label(n), 1.0) for n in self.hof_names) if self.pfsp else ()
        return LeagueSpec(
            self.probs, tuple(self._ids), self.sampling, tuple(self.hof_names), self.hof_share,
            snap_w, hof_w, opp_seats=self.opp_seats,
        )  # fmt: skip

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
            self.win_ema.pop(self.label(old), None)
            evicted.append(old)
            if self.directory is not None:
                (self.directory / f"snap_{old}.npz").unlink(missing_ok=True)
        self._write_index()
        return evicted

    def add_hof(
        self, name: str, weights: NetWeights, update: int = 0, permanent: bool = False
    ) -> list[str]:
        """Add a hall-of-fame entry; returns the names thinned away to respect ``hof_cap``."""
        if name in self._hof_weights:
            raise ValueError(f"hall-of-fame entry {name!r} already exists")
        self._hof.append({"name": name, "update": update, "permanent": permanent})
        self._hof_weights[name] = weights
        if self.directory is not None:
            save_weights(
                weights, self.directory / f"hof_{name}.npz", {"hof": name, "update": update}
            )
        gone = thin_hof(self._hof, self.hof_cap)
        for g in gone:
            self._hof = [e for e in self._hof if e["name"] != g]
            del self._hof_weights[g]
            self.win_ema.pop(self.hof_label(g), None)
            if self.directory is not None:
                (self.directory / f"hof_{g}.npz").unlink(missing_ok=True)
        self._write_index()
        return gone

    def update_results(self, by_label: dict[str, list[float]]) -> None:
        """Fold one update's training results into the per-label win-rate EMAs.

        ``by_label[label] = [games, learner seats, learner wins]`` over games containing ``label``.
        """
        for label, (_, seats, wins) in by_label.items():
            if seats <= 0 or not label.startswith((SNAP_PREFIX, HOF_PREFIX)):
                continue
            wr = wins / seats
            old = self.win_ema.get(label)
            self.win_ema[label] = wr if old is None else old + self.ema_alpha * (wr - old)

    def state(self) -> dict[str, Any]:
        """Everything not recoverable from the snapshot files (saved in checkpoints)."""
        return {"ids": list(self._ids), "hof": self.hof_entries(), "win_ema": dict(self.win_ema)}

    def _write_index(self) -> None:
        if self.directory is not None:
            atomic_write_text(
                self.directory / "pool.json",
                json.dumps({"pool_size": self.pool_size, **self.state()}),
            )

    def restore_state(self, st: dict[str, Any], max_id: int | None = None) -> None:
        """``restore`` plus the hall of fame and the win-rate EMAs from ``state()``."""
        self.restore(list(st["ids"]), max_id)
        if self.directory is None:
            return
        keep = [
            e for e in st.get("hof", [])
            if (max_id is None or e["permanent"] or e["update"] <= max_id)
            and (self.directory / f"hof_{e['name']}.npz").exists()
        ]  # fmt: skip
        names = {e["name"] for e in keep}
        for f in self.directory.glob("hof_*.npz"):
            if f.stem[4:] not in names:
                f.unlink(missing_ok=True)
        self._hof = [dict(e) for e in keep]
        self._hof_weights = {
            n: NetWeights.load(self.directory / f"hof_{n}.npz") for n in sorted(names)
        }
        labels = {self.label(i) for i in self._ids} | {self.hof_label(n) for n in names}
        self.win_ema = {k: float(v) for k, v in st.get("win_ema", {}).items() if k in labels}
        self._write_index()

    def restore(self, ids: list[int], max_id: int | None = None) -> list[int]:
        """Rebuild the recent pool after a resume from the snapshot files on disk.

        Keeps the ids in ``ids`` (oldest first) that are ``<= max_id`` and whose file exists,
        trims to ``pool_size``, and deletes every other ``snap_*.npz`` (taken after the
        checkpoint, or evicted). Returns the ids kept.
        """
        if self.directory is None:
            return []
        keep = [
            i for i in ids
            if (max_id is None or i <= max_id) and (self.directory / f"snap_{i}.npz").exists()
        ]  # fmt: skip
        keep = keep[-self.pool_size :]
        for f in self.directory.glob("snap_*.npz"):
            try:
                sid = int(f.stem.split("_", 1)[1])
            except ValueError:
                continue
            if sid not in keep:
                f.unlink(missing_ok=True)
        self._ids = list(keep)
        self._weights = {i: NetWeights.load(self.directory / f"snap_{i}.npz") for i in keep}
        self._write_index()
        return list(keep)

    def label(self, snap_id: int) -> str:
        return f"{SNAP_PREFIX}{snap_id}"

    def hof_label(self, name: str) -> str:
        return f"{HOF_PREFIX}{name}"

    def opponents(self, labels: list[str] | None = None) -> dict[str, Any]:
        """``label -> npz path`` (cheap to pickle) when persisted, else the weights themselves.

        Default: every recent snapshot and hall-of-fame entry.
        """
        out: dict[str, Any] = {}
        if labels is None:
            labels = [self.label(i) for i in self._ids] + [
                self.hof_label(n) for n in self.hof_names
            ]
        for lab in labels:
            if lab.startswith(HOF_PREFIX):
                name = lab[len(HOF_PREFIX) :]
                out[lab] = (
                    str(self.directory / f"hof_{name}.npz")
                    if self.directory is not None
                    else self._hof_weights[name]
                )
            else:
                i = int(lab[len(SNAP_PREFIX) :])
                out[lab] = (
                    str(self.directory / f"snap_{i}.npz")
                    if self.directory is not None
                    else self._weights[i]
                )
        return out
