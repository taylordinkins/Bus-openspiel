"""Spill directories for rollout samples and their cleanup (Phase 4 unit 8a). No torch.

Workers write ``bus_rollout_<owner pid>_<random>`` directories (tmpfs when roomy); the owner is
the trainer process. A run removes its own directories when it ends, and at startup anyone can
remove directories whose owner pid is dead. Directories of live pids are never touched.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

PREFIX = "bus_rollout_"
_NAME = re.compile(r"^bus_rollout_(\d+)_")
LEGACY_MAX_AGE_S = 3600.0  # pid-less names from before unit 8a: stale after an hour


def spill_roots() -> list[Path]:
    roots = [Path("/dev/shm"), Path(tempfile.gettempdir())]
    return [r for i, r in enumerate(roots) if r.is_dir() and r not in roots[:i]]


def spill_dir() -> str:
    """tmpfs when it has room (samples can be hundreds of MB per update), else the system temp."""
    shm = "/dev/shm"
    if os.path.isdir(shm) and shutil.disk_usage(shm).free > 1_500_000_000:
        return shm
    return tempfile.gettempdir()


def make_spill_dir(owner_pid: int) -> str:
    return tempfile.mkdtemp(prefix=f"{PREFIX}{owner_pid}_", dir=spill_dir())


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_own(owner_pid: int) -> int:
    """Remove every spill directory owned by ``owner_pid``; returns how many."""
    n = 0
    for root in spill_roots():
        for d in root.glob(f"{PREFIX}{owner_pid}_*"):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def cleanup_stale(now: float | None = None) -> list[str]:
    """Remove spill directories of dead owners (and very old pid-less ones); returns the paths."""
    now = time.time() if now is None else now
    removed: list[str] = []
    for root in spill_roots():
        try:
            entries = list(root.glob(f"{PREFIX}*"))
        except OSError:
            continue
        for d in entries:
            if not d.is_dir():
                continue
            m = _NAME.match(d.name)
            if m is not None:
                stale = not pid_alive(int(m.group(1)))
            else:
                try:
                    stale = now - d.stat().st_mtime > LEGACY_MAX_AGE_S
                except OSError:
                    stale = False
            if stale:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))
    return removed
