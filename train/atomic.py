"""Atomic file writes (Phase 4 unit 8a): tmp file + fsync + ``os.replace``. No torch."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any


def fsync_dir(directory: Path) -> None:
    """Make a rename durable (best effort; not supported everywhere)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: str | Path, write: Callable[[IO[Any]], object], binary: bool = True) -> None:
    """Run ``write(fileobj)`` on a tmp file next to ``path`` and move it into place.

    A failure at any point leaves ``path`` untouched (old content or absent) and removes the tmp
    file. The tmp name starts with a dot so ``ckpt_*`` / ``snap_*`` globs never see it.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb" if binary else "w") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write(path, lambda f: f.write(text), binary=False)


def atomic_write_json(path: str | Path, obj: Any, indent: int | None = 1) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=list))
