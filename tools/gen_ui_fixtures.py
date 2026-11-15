"""Write real API output as web fixtures: ``uv run python -m tools.gen_ui_fixtures``.

Produces ``web/src/games/bus/fixtures/{geometry,mid,end}.json`` from the in-process server
(deterministic seeds), plus ``phases/<CODE>.json`` (a full GameView per phase code, loaded from
``tests/scenarios``), ``log.json`` (a finished random game with per-action events) and
``five.json`` (a 5-player mid-selection view), so the web render tests never need the network.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from server.app import create_app

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "games" / "bus" / "fixtures"
SCENARIOS = ROOT / "tests" / "scenarios"

# phase code -> scenario file whose position rests in that phase (real adapter view for each)
PHASE_SCENARIOS = {
    "SB": "setup_sb_midway.json",
    "L1": "setup_l1_mid.json",
    "L2": "setup_l2_mid.json",
    "SEL": "sel_pass2_dropped.json",
    "LN": "resolve_ln.json",
    "PX": "resolve_px.json",
    "BD": "resolve_bd.json",
    "CK": "resolve_ck.json",
    "VP": "resolve_vp.json",
    "VD": "resolve_vp.json",  # no VD scenario has a legal action: pick a passenger first
    "END": "end_both.json",
}


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
    (OUT / "log.json").write_text(json.dumps(e, indent=1) + "\n")

    phases = OUT / "phases"
    phases.mkdir(exist_ok=True)
    for code, name in PHASE_SCENARIOS.items():
        scenario = json.loads((SCENARIOS / name).read_text())
        n = scenario.get("parameters", {}).get("players", 4)
        r = c.post(
            "/api/games",
            json={
                "game": "bus",
                "params": {"num_players": n},
                "seats": ["human"] * n,
                "scenario": scenario,
            },
        )
        assert r.status_code == 200, (name, r.text)
        v = r.json()
        if code == "VD":
            act = v["legal_actions"][0]["id"]
            v = c.post(f"/api/games/{v['id']}/actions", json={"action": act}).json()
        assert v["view"]["view"]["prompt"]["phase"] == code, (name, code)
        (phases / f"{code}.json").write_text(json.dumps(v, indent=1) + "\n")

    rng = random.Random(9)
    f: dict[str, Any] = c.post(
        "/api/games",
        json={"game": "bus", "params": {"num_players": 5}, "seats": ["human"] * 5, "seed": 2},
    ).json()
    while f["view"]["view"]["state"]["round"] < 2 or f["view"]["view"]["state"]["phase"] != "SEL":
        act = rng.choice(f["legal_actions"])["id"]
        f = c.post(f"/api/games/{f['id']}/actions", json={"action": act}).json()
    (OUT / "five.json").write_text(json.dumps(f, indent=1) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
