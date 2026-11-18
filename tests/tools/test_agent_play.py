import json
from pathlib import Path

import pytest

from bus.scenario import load_scenario
from tools import agent_play, replay


def run(argv: list[str]) -> tuple[int, str]:
    lines: list[str] = []
    code = agent_play.main(argv, out=lines.append)
    return code, "\n".join(lines)


def legal_ids(text: str) -> list[int]:
    ids: list[int] = []
    on = False
    for line in text.splitlines():
        if line.startswith("-- P") and "legal actions" in line:
            on = True
        elif on and line.startswith("  "):
            ids.append(int(line.split(":")[0]))
    return ids


def play_to_end(game: Path, text: str) -> str:
    for _ in range(5000):
        if "GAME OVER" in text:
            return text
        code, text = run(["act", str(legal_ids(text)[0]), "--game", str(game), "--note", "first"])
        assert code == 0
    raise AssertionError("game did not end")


def test_full_game_and_replay(tmp_path: Path) -> None:
    g = tmp_path / "g.json"
    code, text = run(
        [
            "new",
            "--players",
            "3",
            "--agent",
            "1",
            "--bots",
            "heuristic",
            "--seed",
            "2",
            "--out",
            str(g),
        ]
    )
    assert code == 0 and "P1 to move" in text and "Player 1" not in text
    text = play_to_end(g, text)
    assert "winners:" in text
    rp = tmp_path / "g.replay.json"
    doc = json.loads(rp.read_text())
    assert load_scenario(doc).is_terminal()
    assert replay.main([str(rp)], out=lambda _s: None) == 0
    code, notes = run(["notes", "--game", str(g)])
    assert code == 0 and "note: first" in notes


def test_illegal_action_leaves_file_unchanged(tmp_path: Path) -> None:
    g = tmp_path / "g.json"
    run(["new", "--players", "3", "--agent", "0", "--seed", "1", "--out", str(g)])
    before = g.read_bytes()
    code, _ = run(["act", "999999", "--game", str(g)])
    assert code == 2 and g.read_bytes() == before


def test_reload_reproduces_state(tmp_path: Path) -> None:
    g = tmp_path / "g.json"
    _, text = run(["new", "--players", "3", "--agent", "0", "--seed", "4", "--out", str(g)])
    _, text = run(["act", str(legal_ids(text)[0]), "--game", str(g)])
    _, shown = run(["show", "--game", str(g)])
    assert shown.split("-- state --")[1] == text.split("-- state --")[1]
    data = json.loads(g.read_text())
    s = agent_play.build_state(data["config"], data["history"])
    assert str(s) in shown


def test_net_bot_seat(tmp_path: Path) -> None:
    g = tmp_path / "g.json"
    code, text = run(
        [
            "new",
            "--players",
            "4",
            "--agent",
            "0",
            "--bots",
            "ppo-smoke",
            "--seed",
            "3",
            "--out",
            str(g),
        ]
    )
    assert code == 0
    assert json.loads(g.read_text())["config"]["bots"]["1"] == "net:ppo-smoke"
    code, text = run(["act", str(legal_ids(text)[0]), "--game", str(g)])
    assert code == 0


def test_bad_bots_count(tmp_path: Path) -> None:
    code, _ = run(
        ["new", "--players", "4", "--bots", "heuristic,random", "--out", str(tmp_path / "x.json")]
    )
    assert code == 2


@pytest.mark.parametrize("spec", ["nosuchmodel"])
def test_unknown_model(tmp_path: Path, spec: str) -> None:
    code, _ = run(["new", "--players", "3", "--bots", spec, "--out", str(tmp_path / "x.json")])
    assert code == 2
