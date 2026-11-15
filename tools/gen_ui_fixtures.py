"""Write real API output as web fixtures: ``uv run python -m tools.gen_ui_fixtures``.

Produces ``web/src/games/bus/fixtures/{geometry,mid,end}.json`` from the in-process server
(deterministic seeds), so the web render tests never need the network.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from server.app import create_app

OUT = Path(__file__).resolve().parent.parent / "web" / "src" / "games" / "bus" / "fixtures"


def main() -> None:
    c: Any = TestClient(create_app(inline_bots=True))
    OUT.mkdir(parents=True, exist_ok=True)
    geo = c.get("/api/boards/bus").json()
    (OUT / "geometry.json").write_text(json.dumps(geo, indent=1) + "\n")

    rng = random.Random(5)
    g: dict[str, Any] = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": 4}, "seats": ["human"] * 4, "seed": 3},
    ).json()
    while True:
        st = g["view"]["view"]["state"]
        if (
            st["round"] >= 3
            and st["phase"] == "SEL"
            and sum(1 for v in st["action_board"].values() for s in v if s is not None) >= 5
        ):
            break
        act = rng.choice(g["legal_actions"])["id"]
        g = c.post(f"/api/games/{g['id']}/actions", json={"action": act}).json()
    (OUT / "mid.json").write_text(json.dumps(g["view"], indent=1) + "\n")

    e = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": 4}, "seats": ["random"] * 4, "seed": 11},
    ).json()
    assert e["view"]["is_terminal"], "random-only game should run to END"
    (OUT / "end.json").write_text(json.dumps(e["view"], indent=1) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
