"""Write real API output as web fixtures: ``uv run python -m tools.gen_ui_fixtures``.

Produces ``web/src/games/bus/fixtures/{geometry,mid,end}.json`` from the in-process server
(deterministic seeds), plus ``phases/<CODE>.json`` (a full GameView per phase code, loaded from
``tests/scenarios``), ``log.json`` (a finished random game with per-action events) and
``five.json`` (a 5-player mid-selection view), and ``interaction/<name>.json`` (UI_DESIGN 6.7:
last stone or not, several picks / deliver sites, board-full SEL, LN with street_ends and a real
loop). ``interaction/LN.loop.json`` is hand-edited (a synthetic double-end case) and is not
written here. The web render tests never need the network.
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
    "VD": "resolve_vp.json",  # replaced by the VP.multi / VD.multi variants below
    "END": "end_both.json",
}


def _spread_markers(board: dict[str, list[Any]], seats: list[int]) -> dict[str, list[Any]]:
    """Fill every space of an action board, giving seats[i] its quota (see the SEL scenario)."""
    it = iter(seats)
    return {k: [next(it) for _ in v] for k, v in board.items()}


def _full_sel(scenario: dict[str, Any]) -> dict[str, Any]:
    """SEL pass 2 with every action space taken: Drop out is the only legal action (R-023)."""
    s = json.loads(json.dumps(scenario))
    board = s["action_board"]
    # seat 0 acts with 2 markers placed; seat 1 has 2; seats 2 and 3 dropped out holding the rest
    quota = [0, 0, 1, 1] + [2] * 12 + [3] * 11
    s["action_board"] = _spread_markers(board, quota)
    for i, left in enumerate([18, 18, 8, 9]):
        s["seats"][i]["markers_left"] = left
    s["pending"] = {"pass": 2, "dropped": [2, 3]}
    return s


def _last_stone(scenario: dict[str, Any]) -> dict[str, Any]:
    """CK with one stone left on the clock (R-053): the other four are held by the seats."""
    s = json.loads(json.dumps(scenario))
    s["stones_on_clock"] = 1
    s["seats"][0]["stones"] = 4
    return s


def _variants() -> dict[str, tuple[dict[str, Any], list[int]]]:
    """Extra interaction views (UI_DESIGN 6.7): name -> (scenario, ids picked after loading)."""

    def load(name: str) -> dict[str, Any]:
        return json.loads((SCENARIOS / name).read_text())

    ck = load("resolve_ck.json")
    vp = load("resolve_vp.json")
    vp["waiting"].update({"19": 1, "20": 1})  # several passengers on the line to pick from
    vd = load("resolve_vp.json")
    # several sites to deliver to: empty houses at n19 and n20, none beside the waiting passenger
    # at n16 (a VD position must be one the Clock step leaves behind, S-CLOCKSTEP / D22)
    vd["buildings"] += [{"site": t, "type": "house"} for t in (8, 39)]
    return {
        "LN.ends": (load("resolve_ln.json"), []),
        "LN.fullloop": (load("lines/full_loop.json"), []),  # a real R-041 loop at one node
        "CK.notlast": (ck, []),
        "CK.last": (_last_stone(ck), []),
        "VP.multi": (vp, []),
        "VD.multi": (vd, [309]),
        "SEL.dropout_only": (_full_sel(load("sel_pass2_dropped.json")), []),
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

    variants = _variants()
    phases = OUT / "phases"
    phases.mkdir(exist_ok=True)
    for code, name in PHASE_SCENARIOS.items():
        scenario = json.loads((SCENARIOS / name).read_text())
        picks: list[int] = []
        if code in ("VP", "VD"):  # several passengers / sites to choose from
            scenario, picks = variants["VD.multi" if code == "VD" else "VP.multi"]
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
        for act in picks:
            v = c.post(f"/api/games/{v['id']}/actions", json={"action": act}).json()
        assert v["view"]["view"]["prompt"]["phase"] == code, (name, code)
        (phases / f"{code}.json").write_text(json.dumps(v, indent=1) + "\n")

    inter = OUT / "interaction"
    inter.mkdir(exist_ok=True)
    for name, (scenario, picks) in variants.items():
        n = scenario.get("parameters", {}).get("players", 4)
        r = c.post(
            "/api/games",
            json={"game": "bus", "params": {"num_players": n}, "seats": ["human"] * n,
                  "scenario": scenario},
        )  # fmt: skip
        assert r.status_code == 200, (name, r.text)
        v = r.json()
        for act in picks:
            v = c.post(f"/api/games/{v['id']}/actions", json={"action": act}).json()
        (inter / f"{name}.json").write_text(json.dumps(v, indent=1) + "\n")

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
