import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from bus.board import load_board

ROOT = Path(__file__).resolve().parents[2]


def test_render_board_writes_wellformed_svg(tmp_path: Path) -> None:
    out = tmp_path / "board.svg"
    subprocess.run(
        [sys.executable, "-m", "tools.render_board", "--out", str(out)], cwd=ROOT, check=True
    )
    root = ET.parse(out).getroot()
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    board = load_board()
    assert {f"street-{s.id}" for s in board.streets} <= ids
    assert {f"site-{s.id}" for s in board.sites} <= ids
    assert {f"node-{n.id}" for n in board.nodes} <= ids
    text = out.read_text()
    assert "Player 1" not in text and "P0" in text
