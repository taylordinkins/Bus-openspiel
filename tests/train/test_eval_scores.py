# pyright: basic, reportMissingImports=false
"""Phase 4 unit 8c: raw game-score columns in eval.csv and old-header resume compatibility."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tools.train_ppo import EVAL_COLUMNS, format_eval_table  # noqa: E402
from train.runstate import read_eval_rows, truncate_csv  # noqa: E402

OLD = ["update", "opponent", "players", "games", "win_rate", "lo", "hi", "mean_return"]
NEW = ["mean_score", "mean_opp_score", "mean_win_score"]


def test_new_columns_are_appended() -> None:
    assert EVAL_COLUMNS[: len(OLD)] == OLD and EVAL_COLUMNS[len(OLD) :] == NEW


def test_old_eval_header_is_upgraded_on_resume(tmp_path: Path) -> None:
    p = tmp_path / "eval.csv"
    p.write_text(
        ",".join(OLD) + "\n10,random,3,100,0.66,0.5,0.7,0.44\n20,random,3,100,0.7,0.5,0.8,0.5\n"
    )
    assert truncate_csv(p, EVAL_COLUMNS, 10) == 1
    with open(p, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == EVAL_COLUMNS
    assert rows[1][: len(OLD)] == ["10", "random", "3", "100", "0.66", "0.5", "0.7", "0.44"]
    assert rows[1][len(OLD) :] == ["", "", ""]
    got = read_eval_rows(p)
    assert got[0]["mean_return"] == pytest.approx(0.44)
    assert got[0]["mean_score"] != got[0]["mean_score"]  # NaN for the old row
    assert "10" in format_eval_table(got)  # the table copes with blank scores


def test_scores_roundtrip_and_table(tmp_path: Path) -> None:
    p = tmp_path / "eval.csv"
    row = {
        "update": 5, "opponent": "heuristic", "players": 4, "games": 8, "win_rate": 0.5,
        "lo": 0.2, "hi": 0.8, "mean_return": 0.1,
        "mean_score": 12.5, "mean_opp_score": 10.0, "mean_win_score": 15.0,
    }  # fmt: skip
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, EVAL_COLUMNS)
        w.writeheader()
        w.writerow(row)
    got = read_eval_rows(p)
    assert got[0]["mean_score"] == 12.5 and got[0]["mean_win_score"] == 15.0
    assert "12.5/10.0/15.0" in format_eval_table(got)
