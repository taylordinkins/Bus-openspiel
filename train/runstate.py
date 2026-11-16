# pyright: basic, reportMissingImports=false
"""Run directory bookkeeping for the PPO trainer (Phase 4 unit 8a): checkpoints, ``latest.json``,
retention, CSV truncation on resume, resume-argument rules, parent RSS. Needs torch."""

from __future__ import annotations

import argparse
import csv
import json
import re
import resource
import time
from pathlib import Path
from typing import Any

import torch

from train.atomic import atomic_write, atomic_write_json

CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")

# Options that may change on --resume (argparse dests). Everything else must equal args.json.
RESUME_WHITELIST = frozenset({
    "max_minutes", "workers", "eval_every", "eval_games", "eval_mcts", "eval_greedy",
    "ckpt_every", "keep_every", "lr", "lr_decay", "ent_coef", "ent_coef_final", "players",
    "league", "snapshot_every", "pool_size", "updates", "device", "stop_after", "task_timeout",
    "max_retries",
})  # fmt: skip


class ResumeError(Exception):
    """A resume request that cannot be honoured (bad override, incompatible checkpoint)."""


def rss_mb() -> float:
    """Resident set size of this process in MB (/proc, else peak from getrusage)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------- checkpoints


def save_pt(path: Path, obj: dict[str, Any]) -> None:
    atomic_write(path, lambda f: torch.save(obj, f))


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def write_latest(run: Path, update: int) -> None:
    atomic_write_json(
        run / "latest.json",
        {
            "update": update,
            "pt": f"ckpt_{update}.pt",
            "npz": f"ckpt_{update}.npz",
            "time": now_iso(),
        },
    )


def checkpoint_updates(run: Path) -> list[int]:
    return sorted(int(m.group(1)) for p in run.iterdir() if (m := CKPT_RE.match(p.name)))


def find_latest_checkpoint(run: Path) -> tuple[int, dict[str, Any]] | None:
    """(update, loaded checkpoint) from ``latest.json``; if that is missing, corrupt or points at
    an unloadable file, the newest ``ckpt_*.pt`` that loads (truncated files are skipped)."""
    tried: set[int] = set()
    try:
        u = int(json.loads((run / "latest.json").read_text())["update"])
        tried.add(u)
        return u, load_pt(run / f"ckpt_{u}.pt")
    except Exception:
        pass
    for u in reversed(checkpoint_updates(run)):
        if u in tried:
            continue
        try:
            return u, load_pt(run / f"ckpt_{u}.pt")
        except Exception:
            continue
    return None


def apply_retention(run: Path, keep_last: int = 3, keep_every: int = 250) -> list[int]:
    """Delete old ``ckpt_<u>.pt/.npz`` pairs: keep the newest ``keep_last``, every update that is
    a multiple of ``keep_every`` (0 = no milestones). ``best.*`` is not touched. Returns the
    deleted updates."""
    ups = checkpoint_updates(run)
    keep = set(ups[-keep_last:]) if keep_last > 0 else set()
    if keep_every > 0:
        keep |= {u for u in ups if u % keep_every == 0}
    gone = [u for u in ups if u not in keep]
    for u in gone:
        (run / f"ckpt_{u}.pt").unlink(missing_ok=True)
        (run / f"ckpt_{u}.npz").unlink(missing_ok=True)
    return gone


def remove_stray_tmp(run: Path) -> None:
    for p in list(run.glob(".*.tmp*")) + list((run / "pool").glob(".*.tmp*")):
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------- CSV


def truncate_csv(path: Path, header: list[str], last_update: int) -> int:
    """Make ``path`` hold ``header`` plus the rows with ``update <= last_update``; returns rows
    dropped. A missing or empty file just gets the header. A different header is an error."""
    rows: list[list[str]] = []
    dropped = 0
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            got = next(reader, [])
            if got != header:
                raise ResumeError(f"{path.name}: header differs from this version's columns")
            for r in reader:
                try:
                    keep = int(r[0]) <= last_update
                except (ValueError, IndexError):
                    keep = False  # a torn last line
                if keep:
                    rows.append(r)
                else:
                    dropped += 1

    def write(f: Any) -> None:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    atomic_write(path, write, binary=False)
    return dropped


def read_eval_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "update": int(r["update"]), "opponent": r["opponent"], "games": int(r["games"]),
                "win_rate": float(r["win_rate"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
                "mean_return": float(r["mean_return"]),
            })  # fmt: skip
    return out


# ---------------------------------------------------------------- resume arguments


_UNSET = object()  # not a str, so argparse never runs ``type`` on it


def explicit_options(parser: argparse.ArgumentParser, argv: list[str] | None) -> dict[str, Any]:
    """The options that were actually given on the command line (dest -> parsed value)."""
    import copy

    probe = copy.deepcopy(parser)
    for action in probe._actions:
        action.default = _UNSET
    ns = vars(probe.parse_args(argv))
    return {k: v for k, v in ns.items() if v is not _UNSET and k != "help"}


def _norm(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def merge_resume_args(
    saved: dict[str, Any], explicit: dict[str, Any], defaults: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Saved args with whitelisted explicit overrides applied.

    Returns (merged dict, {name: {"old", "new"}} for every value that actually changed). Any
    changed value outside ``RESUME_WHITELIST`` raises ``ResumeError`` naming the option.
    """
    base = {**defaults, **saved}
    merged = dict(base)
    changes: dict[str, dict[str, Any]] = {}
    for k, v in explicit.items():
        if k == "resume":
            continue
        if _norm(v) == _norm(base.get(k)):
            continue
        flag = "--" + k.replace("_", "-")
        if k not in RESUME_WHITELIST:
            raise ResumeError(
                f"cannot change {flag} on resume (run has {base.get(k)!r}, got {v!r}); "
                f"allowed overrides: "
                + ", ".join("--" + w.replace("_", "-") for w in sorted(RESUME_WHITELIST))
            )
        merged[k] = v
        changes[k] = {"old": _norm(base.get(k)), "new": _norm(v)}
    return merged, changes


def append_resume_log(run: Path, entry: dict[str, Any]) -> None:
    p = run / "resume_log.json"
    log: list[dict[str, Any]] = []
    if p.exists():
        try:
            log = json.loads(p.read_text())
        except json.JSONDecodeError:
            log = []
    log.append(entry)
    atomic_write_json(p, log)
