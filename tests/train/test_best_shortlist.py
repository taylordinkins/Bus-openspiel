"""Top-K best-model shortlist, formula tags and the resume horizon guard (unit 8c prep)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from tools.train_ppo import composite_score, is_better, main  # noqa: E402
from train.runstate import (  # noqa: E402
    ResumeError,
    formula_of,
    merge_resume_args,
    shortlist_insert,
    sync_best_files,
)


def rec(u: int, c: float, ref: bool = False) -> dict[str, Any]:
    r: dict[str, Any] = {"update": u, "composite": c, "formula": "h+hof" if ref else "h-only"}
    if ref:
        r["hof_ref"] = c
    return r


def test_shortlist_eviction_and_ties() -> None:
    short: list[dict[str, Any]] = []
    for u, c in [(1, 0.5), (2, 0.7), (3, 0.6)]:
        short, adm, ev = shortlist_insert(short, rec(u, c), 3)
        assert adm and not ev
    short, adm, ev = shortlist_insert(short, rec(4, 0.55), 3)
    assert adm and [r["update"] for r in ev] == [1]
    assert [r["update"] for r in short] == [2, 3, 4]
    short, adm, ev = shortlist_insert(short, rec(5, 0.55), 3)  # tie with the lowest: rejected
    assert not adm and [r["update"] for r in short] == [2, 3, 4]
    short, adm, _ = shortlist_insert(short, rec(6, 0.9), 3)
    assert short[0]["update"] == 6


def test_formula_separation() -> None:
    assert formula_of({"composite": 1.0, "hof_ref": 0.3}) == "h+hof"
    assert formula_of({"composite": 1.0}) == "h-only"
    # h-only 0.99 never beats h+hof 0.4, and h+hof 0.3 beats h-only 0.99
    assert not is_better(rec(1, 0.99), rec(2, 0.4, ref=True))
    assert is_better(rec(3, 0.3, ref=True), rec(4, 0.99))
    short = [rec(1, 0.4, ref=True), rec(2, 0.3, ref=True)]
    short, adm, _ = shortlist_insert(short, rec(3, 0.99), 2)
    assert not adm
    short, adm, ev = shortlist_insert([rec(1, 0.99), rec(2, 0.98)], rec(3, 0.2, ref=True), 2)
    assert adm and short[0]["update"] == 3 and [r["update"] for r in ev] == [2]
    rows = [
        {"opponent": "heuristic", "win_rate": 0.5},
        {"opponent": "hof_ref:x", "win_rate": 0.25},
    ]
    c, d = composite_score(rows)
    assert d["formula"] == "h+hof" and c == pytest.approx(0.45)
    assert composite_score(rows[:1])[1]["formula"] == "h-only"


def test_sync_best_files(tmp_path: Path) -> None:
    for u in (1, 2, 3):
        (tmp_path / f"best_{u}.pt").write_bytes(f"pt{u}".encode())
        (tmp_path / f"best_{u}.npz").write_bytes(f"npz{u}".encode())
    sync_best_files(tmp_path, [rec(3, 0.9), rec(2, 0.8)])
    assert not (tmp_path / "best_1.pt").exists() and not (tmp_path / "best_1.npz").exists()
    assert (tmp_path / "best.pt").read_bytes() == b"pt3"
    assert json.loads((tmp_path / "best.json").read_text())["update"] == 3
    assert [r["update"] for r in json.loads((tmp_path / "best_list.json").read_text())] == [3, 2]


def test_horizon_change_rejected_without_flag() -> None:
    defaults = {"updates": 100, "workers": 1}
    saved = {"updates": 10, "workers": 1}
    with pytest.raises(ResumeError, match="allow-horizon-change"):
        merge_resume_args(saved, {"updates": 20}, defaults)
    merged, ch = merge_resume_args(saved, {"updates": 20, "allow_horizon_change": True}, defaults)
    assert merged["updates"] == 20 and ch["updates"] == {"old": 10, "new": 20}
    assert "allow_horizon_change" not in ch
    assert merge_resume_args(saved, {"updates": 10}, defaults)[1] == {}  # unchanged is fine


def run_args(tmp: Path, *extra: str) -> list[str]:
    return [
        "--name", "r", "--out-dir", str(tmp), "--games-per-update", "4", "--workers", "0",
        "--hidden", "32,32", "--device", "cpu", "--eval-every", "1", "--eval-games", "4",
        "--ckpt-every", "100", "--seed", "3", "--envs-per-worker", "2", "--hof-seed", "",
        "--league", "1,0,0", "--best-keep", "2", *extra,
    ]  # fmt: skip


def test_run_keeps_k_best_and_resumes(tmp_path: Path) -> None:
    assert main(run_args(tmp_path, "--updates", "4", "--stop-after", "3")) == 0
    run = tmp_path / "r"
    lst = json.loads((run / "best_list.json").read_text())
    assert 1 <= len(lst) <= 2 and all(r["formula"] == "h-only" for r in lst)
    assert lst[0] == json.loads((run / "best.json").read_text())
    for r in lst:
        assert (run / f"best_{r['update']}.pt").exists()
        assert (run / f"best_{r['update']}.npz").exists()
    assert len(list(run.glob("best_*.pt"))) == len(lst)
    assert main(["--resume", str(run)]) == 0
    lst2 = json.loads((run / "best_list.json").read_text())
    assert len(lst2) == 2 and lst2 == sorted(lst2, key=lambda r: -r["composite"])
    assert len(list(run.glob("best_*.pt"))) == 2
    assert (run / "best.pt").exists() and (run / "best.npz").exists()
