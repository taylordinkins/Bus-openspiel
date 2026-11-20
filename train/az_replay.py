# pyright: basic
"""FIFO replay of sparse AZ samples (Phase 4 unit 8h). Numpy only.

Samples arrive in chunks (one per actor task, the ``train.az_selfplay`` sparse layout). The
buffer keeps whole chunks, newest last, and drops the oldest while more than ``capacity`` samples
are held (the newest chunk is always kept, even if it alone is larger; whole-chunk eviction can
leave it a little under ``capacity``). ``batch`` densifies the
selected samples: observations (B, OBS), legal masks and policy targets (B, A), and the per-seat
fields as they are.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bus.actions import NUM_DISTINCT_ACTIONS
from train.features import OBS_SIZE  # default (v2); v3 runs pass obs_size

Array = np.ndarray[Any, Any]
ROW_KEYS = ("z", "vmask", "q", "ascore", "arank", "awin", "prog", "players")


class Chunk:
    """One task's samples; ``obs_end``/``act_end`` close the ragged offsets."""

    def __init__(self, s: dict[str, Array]) -> None:
        self.s = s
        self.n = len(s["z"])
        self.obs_end = np.append(s["obs_off"], len(s["obs_idx"]))
        self.act_end = np.append(s["act_off"], len(s["act_id"]))
        self.nbytes = sum(v.nbytes for v in s.values())


class Replay:
    def __init__(self, capacity: int, obs_size: int = OBS_SIZE) -> None:
        self.capacity = capacity
        self.obs_size = obs_size
        self.chunks: list[Chunk] = []
        self.size = 0
        self.added = 0  # samples ever added

    def add(self, samples: dict[str, Array]) -> None:
        c = Chunk(samples)
        if c.n == 0:
            return
        self.chunks.append(c)
        self.size += c.n
        self.added += c.n
        while len(self.chunks) > 1 and self.size > self.capacity:
            self.size -= self.chunks.pop(0).n

    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.chunks)

    def locate(self, idx: Array) -> list[tuple[Chunk, Array]]:
        """Group global sample indices (0 = oldest held) by chunk: [(chunk, local indices)]."""
        bounds = np.cumsum([0, *(c.n for c in self.chunks)])
        which = np.searchsorted(bounds, idx, side="right") - 1
        out = []
        for ci in np.unique(which):
            sel = which == ci
            out.append((self.chunks[int(ci)], idx[sel] - bounds[ci]))
        return out

    def batch(self, idx: Array) -> dict[str, Array]:
        """Dense minibatch of the samples at ``idx`` (order kept)."""
        idx = np.asarray(idx, np.int64)
        b = len(idx)
        obs = np.zeros((b, self.obs_size), np.float32)
        mask = np.zeros((b, NUM_DISTINCT_ACTIONS), np.bool_)
        pi = np.zeros((b, NUM_DISTINCT_ACTIONS), np.float32)
        rows: dict[str, list[Array]] = {k: [] for k in ROW_KEYS}
        order: list[Array] = []
        r0 = 0
        pos = np.argsort(idx, kind="stable")
        for c, loc in self.locate(idx[pos]):
            s = c.s
            for j, i in enumerate(loc):
                r = int(pos[r0 + j])
                a, e = c.obs_end[i], c.obs_end[i + 1]
                obs[r, s["obs_idx"][a:e]] = s["obs_tab"][s["obs_code"][a:e]]
                a, e = c.act_end[i], c.act_end[i + 1]
                acts = s["act_id"][a:e]
                mask[r, acts] = True
                pi[r, acts] = s["pi"][a:e]
            for k in ROW_KEYS:
                rows[k].append(s[k][loc])
            order.append(pos[r0 : r0 + len(loc)])
            r0 += len(loc)
        out: dict[str, Array] = {"obs": obs, "mask": mask, "pi": pi}
        inv = np.empty(b, np.int64)
        inv[np.concatenate(order)] = np.arange(b)
        for k in ROW_KEYS:
            out[k] = np.concatenate(rows[k])[inv]
        return out
