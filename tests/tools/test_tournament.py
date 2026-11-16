"""tools.tournament: Wilson interval, rotation, reproducibility, CLI validation, JSON schema."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tools.tournament import main, run_tournament, seat_types_for, wilson_interval


def test_wilson_known_values() -> None:
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.404) < 0.001 and abs(hi - 0.596) < 0.001


def test_wilson_edges() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert 0.0 < wilson_interval(0, 10)[1] < 1.0


def test_rotation_covers_every_rotation_equally() -> None:
    seats = ["a", "b", "c", "d"]
    rots = Counter(tuple(seat_types_for(seats, g, True)) for g in range(12))
    assert len(rots) == 4 and set(rots.values()) == {3}
    for g in range(12):
        assert sorted(seat_types_for(seats, g, True)) == seats
        assert seat_types_for(seats, g, False) == seats


def test_reproducible_across_worker_counts() -> None:
    seats = ["random"] * 4

    def strip(d: dict[str, Any]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if "seconds" not in k} for r in d["games"]]

    a = run_tournament(seats, 12, seed=3, workers=1)
    b = run_tournament(seats, 12, seed=3, workers=2)
    assert strip(a) == strip(b)
    assert len(a["games"]) == 12


def test_json_schema_and_cli(tmp_path: Path) -> None:
    out_file = tmp_path / "r.json"
    lines: list[str] = []
    rc = main(
        ["--seats", "heuristic,random,random,random", "--games", "6", "--workers", "1",
         "--seed", "1", "--out", str(out_file)],
        out=lines.append,
    )  # fmt: skip
    assert rc == 0
    assert "rounded up from 6 to 8" in lines[0]
    data = json.loads(out_file.read_text())
    assert {"config", "summary", "games", "mean_rounds", "end_types"} <= set(data)
    assert {"players", "seats", "games", "seed", "rotate"} <= set(data["config"])
    assert set(data["summary"]) == {"by_seat_type", "by_position"}
    stats = data["summary"]["by_seat_type"]["heuristic"]
    assert {
        "games", "outright_wins", "shared_credit", "win_share", "wilson_low",
        "wilson_high", "mean_score", "mean_margin", "mean_decision_ms",
    } <= set(stats)  # fmt: skip
    assert stats["games"] == 8
    assert data["summary"]["by_seat_type"]["random"]["games"] == 24
    row = data["games"][0]
    assert {"game", "seat_types", "winners", "final_scores", "end_types", "rounds"} <= set(row)


def test_cli_rejects_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--players", "4", "--seats", "random,random,random"]) == 2
    assert main(["--players", "3", "--seats", "random,random,bogus"]) == 2
    assert main(["--players", "3", "--seats", "random,random,human"]) == 2
    assert main(["--players", "3", "--seats", "random,random,mcts:x"]) == 2
    assert "error" in capsys.readouterr().err
