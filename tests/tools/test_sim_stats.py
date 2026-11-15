import json
from pathlib import Path

from tools import sim_stats


def test_report_keys_and_consistency() -> None:
    data = sim_stats.run(3, (3, 4), seed=5)
    assert set(data["by_players"]) == {"3", "4"}
    for key, r in data["by_players"].items():
        n = int(key)
        assert r["games"] == 3
        assert sum(r["end_types"].values()) == 3
        assert sum(r["winner_count_dist"].values()) == 3
        assert len(r["seat_win_rate"]) == n
        assert abs(sum(r["seat_win_rate"]) - 1.0) < 1e-9
        assert r["moves"]["max"] <= r["max_game_length"]
        assert r["rounds"]["min"] <= r["rounds"]["median"] <= r["rounds"]["max"]
        assert 0 <= r["sites_full_rate"] <= 1
        assert all(0 <= u <= 25 for u in r["line_markers_used_mean"])
    text = sim_stats.format_report(data)
    assert "== 3 players" in text and "end types:" in text and "branching factor" in text


def test_deterministic_by_seed() -> None:
    def strip(d: dict) -> dict:  # type: ignore[type-arg]
        for r in d["by_players"].values():
            r.pop("seconds_per_game")
        return d

    assert strip(sim_stats.run(2, (3,), seed=9)) == strip(sim_stats.run(2, (3,), seed=9))


def test_main_json(tmp_path: Path) -> None:
    lines: list[str] = []
    f = tmp_path / "o.json"
    code = sim_stats.main(
        ["--games", "1", "--players", "3", "--seed", "1", "--json", str(f)], out=lines.append
    )
    assert code == 0 and json.loads(f.read_text())["games"] == 1 and lines
