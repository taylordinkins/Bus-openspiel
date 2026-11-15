import json
from pathlib import Path

from bus.scenario import dump_scenario, load_scenario
from tools import play_text, replay


def run(argv: list[str]) -> tuple[int, list[str]]:
    lines: list[str] = []
    return replay.main(argv, out=lines.append), lines


def saved_game(tmp_path: Path, n: int = 4) -> tuple[Path, dict]:  # type: ignore[type-arg]
    f = tmp_path / "g.json"
    play_text.main(["--players", str(n), "--seed", "2", "--save", str(f)], out=lambda _: None)
    return f, json.loads(f.read_text())


def test_save_then_replay_reproduces_dump(tmp_path: Path) -> None:
    f, doc = saved_game(tmp_path)
    code, lines = run([str(f)])
    assert code == 0 and any(line.startswith("final scores:") for line in lines)
    state = load_scenario(doc)
    assert lines[-5] == str(state)
    assert dump_scenario(state)["phase"] == "END"


def test_steps_line_count(tmp_path: Path) -> None:
    f, doc = saved_game(tmp_path, 3)
    n = len(doc["history"])
    code, lines = run([str(f), "--steps"])
    assert code == 0
    assert sum(1 for line in lines if line.startswith("move ")) == n
    code, compact = run([str(f), "--steps", "--compact"])
    assert code == 0 and len(compact) == 2 * n + 5
    assert compact[0].startswith("move 1: P")


def test_illegal_id_exits_nonzero(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    f, doc = saved_game(tmp_path)
    doc["history"][3] = 4999
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    code, _ = run([str(bad), "--steps"])
    assert code == 1
    assert "S-HISTORY" in capsys.readouterr().err


def test_missing_file(tmp_path: Path) -> None:
    code, _ = run([str(tmp_path / "nope.json")])
    assert code == 2
