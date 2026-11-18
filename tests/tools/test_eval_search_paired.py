"""Paired differences in tools.eval_search (Phase 4 interaction, experiment (b))."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eval_search import main, paired

NET = "models/ppo-long.npz"


def _row(cand: str, game: int, seat: int, winners: list[int]) -> dict:
    return {"candidate": cand, "baseline": "ppo_sample", "players": 4, "game": game,
            "cand_seat": seat, "winners": winners}  # fmt: skip


def test_paired_hand_case() -> None:
    rows = [
        _row("ref", 0, 0, [0]), _row("ref", 1, 1, [2]), _row("ref", 2, 2, [2, 3]),
        _row("new", 0, 0, [1]), _row("new", 1, 1, [1]), _row("new", 2, 2, [2]),
        _row("new", 3, 3, [3]),  # no ref game 3: not paired
    ]  # fmt: skip
    (p,) = paired(rows, "ref")
    assert p["candidate"] == "new" and p["games"] == 3
    assert p["diff"] == pytest.approx(((0 - 1) + (1 - 0) + (1 - 0.5)) / 3)
    assert p["half_width"] > 0
    assert paired(rows, "missing") == []


def test_cli_paired_ref(tmp_path: Path) -> None:
    args = ["--candidates", "net;gumbel:2,t=2", "--baselines", "heuristic", "--players", "3",
            "--games", "3", "--workers", "1", "--net", NET, "--out", str(tmp_path), "--quiet",
            "--paired-ref", "net"]  # fmt: skip
    lines: list[str] = []
    assert main(args, out=lines.append) == 0
    data = json.loads((tmp_path / "summary.json").read_text())
    assert [p["candidate"] for p in data["paired"]] == ["gumbel:2,t=2"]
    assert data["paired"][0]["games"] == 3
    assert any(line.startswith("paired gumbel:2,t=2 - net") for line in lines)
