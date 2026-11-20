import re
from pathlib import Path

from bus.board import load_board
from bus.lines import line_ends
from bus.rules import board
from tools import agent_play, board_legend


def run(argv: list[str]) -> tuple[int, str]:
    lines: list[str] = []
    code = agent_play.main(argv, out=lines.append)
    return code, "\n".join(lines)


def test_every_node_street_site_listed() -> None:
    b = load_board()
    text = board_legend.legend(b)
    for n in b.nodes:
        assert re.search(rf"^n{n.id} \(", text, re.M)
    for s in b.streets:
        assert f"s{s.id}: n{s.a}-n{s.b}" in text
    for t in b.sites:
        assert f"site {t.id} ({t.tier}, n{t.node})" in text
    for n in b.stations + b.roundabouts:
        assert f"n{n}" in text


def test_neighbours_symmetric_with_board() -> None:
    b = load_board()
    text = board_legend.legend(b)
    nb: dict[int, set[tuple[int, int]]] = {}
    for line in text.splitlines():
        m = re.match(r"^n(\d+) \(.*?:(.*) \| sites:", line)
        if m:
            nb[int(m[1])] = {(int(a), int(c)) for a, c in re.findall(r"n(\d+) \(s(\d+)\)", m[2])}
    assert len(nb) == len(b.nodes)
    for s in b.streets:
        assert (s.b, s.id) in nb[s.a]
        assert (s.a, s.id) in nb[s.b]
    assert sum(len(v) for v in nb.values()) == 2 * len(b.streets)


def test_legend_subcommand_and_module() -> None:
    code, text = run(["legend"])
    assert code == 0
    assert text == board_legend.legend()


def start(tmp_path: Path) -> Path:
    g = tmp_path / "g.json"
    code, _ = run(["new", "--players", "3", "--agent", "0", "--seed", "3", "--out", str(g)])
    assert code == 0
    return g


def play_some(g: Path, n: int) -> str:
    text = ""
    for _ in range(n):
        _, text = run(["show", "--game", str(g)])
        if "GAME OVER" in text:
            break
        first = next(
            int(x.split(":")[0]) for x in text.split("legal actions", 1)[1].splitlines()[1:] if x
        )
        run(["act", str(first), "--game", str(g)])
    return text


def test_summary_mentions_line_ends_and_default_unchanged(tmp_path: Path) -> None:
    g = start(tmp_path)
    play_some(g, 60)
    _, plain = run(["show", "--game", str(g)])
    code, withs = run(["show", "--game", str(g), "--summary"])
    assert code == 0
    assert "-- summary" not in plain
    assert withs.startswith(plain)
    data = agent_play.load(g)
    state = agent_play.build_state(data["config"], data["history"])
    b = board()
    seen = 0
    for seat, st in enumerate(state.core.seats):
        if st.line:
            e0, e1 = line_ends(b, st.line)
            assert f"ends n{e0}, n{e1}" in withs
            assert f"P{seat}:" in withs
            seen += 1
    assert seen >= 1


def _mid_game(tmp_path: Path) -> Path:
    g = tmp_path / "g.json"
    code, text = run(["new", "--players", "3", "--agent", "0", "--seed", "2", "--out", str(g)])
    assert code == 0
    for _ in range(12):
        ids = [int(ln.split(":")[0]) for ln in text.splitlines() if re.match(r"^  \d+: ", ln)]
        code, text = run(["act", str(ids[0]), "--game", str(g)])
        assert code == 0 and "GAME OVER" not in text
    return g


def test_legend_accepts_game_flag(tmp_path: Path) -> None:
    g = _mid_game(tmp_path)
    code, text = run(["legend", "--game", str(g)])
    assert code == 0 and text == run(["legend"])[1]


def test_summary_markers_and_post_clock(tmp_path: Path) -> None:
    g = _mid_game(tmp_path)
    code, text = run(["show", "--game", str(g), "--summary"])
    assert code == 0
    for seat in range(3):
        assert re.search(rf"P{seat}: am=\d+ \(placed this round: \d+\)", text)
    assert "R-066" in text and "if the clock advances:" in text
    assert text.count("Vrrooomm! targets if the clock advances") == 3
    assert "-- action board (compact)" not in text


def test_board_compact_lists_every_space(tmp_path: Path) -> None:
    from bus.actions import ACTION_NAMES

    g = _mid_game(tmp_path)
    code, text = run(["show", "--game", str(g), "--board-compact"])
    assert code == 0
    section = text.split("-- action board (compact)")[1]
    for name in ACTION_NAMES:
        assert f"  {name} [" in section
    assert "-- summary" in text
