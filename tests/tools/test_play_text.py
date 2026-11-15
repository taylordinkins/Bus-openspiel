import json
from pathlib import Path

import pytest

from bus.scenario import dump_scenario, load_scenario
from tools import play_text


def run(argv: list[str], inputs: list[str] | None = None) -> tuple[int, list[str], list[str]]:
    feed = iter(inputs or [])
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    lines: list[str] = []
    code = play_text.main(argv, input_fn=input_fn, out=lines.append)
    return code, lines, prompts


@pytest.mark.parametrize("n", [3, 4, 5])
def test_all_bot_game_terminates(n: int) -> None:
    code, lines, _ = run(["--players", str(n), "--seed", "3"])
    assert code == 0
    text = "\n".join(lines)
    assert "final scores:" in text and "winners:" in text and "returns:" in text
    assert "end types:" in text


def test_determinism_and_save(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run(["--seed", "5", "--save", str(a)])
    run(["--seed", "5", "--save", str(b)])
    run(["--seed", "6", "--save", str(tmp_path / "c.json")])
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    assert da == db and da["history"]
    assert da["history"] != json.loads((tmp_path / "c.json").read_text())["history"]
    assert dump_scenario(load_scenario(da))["phase"] == "END"


def test_margin_returns_saved(tmp_path: Path) -> None:
    f = tmp_path / "m.json"
    run(["--returns", "margin", "--save", str(f)])
    assert json.loads(f.read_text())["parameters"]["returns"] == "margin"


def test_human_choices_undo_quit() -> None:
    # number, invalid, #id of a legal action, undo, quit
    code, lines, _ = run(["--human", "0"], ["0", "zzz", "99999", "u", "q"])
    text = "\n".join(lines)
    assert code == 0 and "quit" in text
    assert text.count("invalid entry") == 2
    assert "P0 to move:" in text and "Player 1" not in text
    first = next(line for line in lines if line.startswith("  0) "))
    assert "P0" in first


def test_hash_id_choice_and_undo_rewinds(tmp_path: Path) -> None:
    f = tmp_path / "h.json"
    # find first legal id by quitting immediately
    _, lines, _ = run(["--human", "0"], ["q"])
    first = next(line for line in lines if line.startswith("  0) "))
    aid = int(first.rsplit("#", 1)[1].rstrip("]"))
    # play #aid, then undo it, then quit: history saved is empty
    code, lines, _ = run(["--human", "0", "--save", str(f)], [f"#{aid}", "u", "q"])
    assert code == 0
    assert json.loads(f.read_text())["history"] == []
    # without undo the move is kept
    run(["--human", "0", "--save", str(f)], [f"#{aid}", "q"])
    assert json.loads(f.read_text())["history"][:1] == [aid]


def test_undo_with_nothing() -> None:
    _, lines, _ = run(["--human", "0"], ["u", "q"])
    assert "nothing to undo" in "\n".join(lines)


def test_human_list_forms_and_range() -> None:
    assert play_text.parse_humans(["0,2", "1"], 4) == {0, 1, 2}
    code, _, _ = run(["--human", "7"])
    assert code == 2


def test_no_autoforce_prompts_more() -> None:
    # EOF on input ends as quit; with autoforce a forced seat is never prompted
    code, lines, _ = run(["--human", "0", "--no-autoforce"], [])
    assert code == 0 and "quit" in "\n".join(lines)
