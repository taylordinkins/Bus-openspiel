"""Phase 2 unit 2: Bus server adapter (docs/UI_DESIGN.md section 6), written ahead of the code.

Covers `load`, the `view` contract (`state` round trip 6.1, `derived` 6.2, `prompt` 6.3,
`targets` 6.4) and labels (ACTIONS 4, 5). The adapter is `server.adapters.bus.BusAdapter`.
Engine expectations are read through the public scenario API (`dump_scenario`), never by
reimplementing the engine.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import pytest

import bus.game as bus_game
from server.adapters.bus import BusAdapter
from tests import harness as H
from tests.support import play as P

Json = dict[str, Any]

ACTION_KEYS = ["lines", "buses", "passengers", "buildings", "clock", "vrrooomm", "starting_player"]
SPACES = {k: (1 if k in ("buses", "clock", "starting_player") else 6) for k in ACTION_KEYS}
RESOLVE_ACTION = {
    "LN": "lines",
    "PX": "passengers",
    "BD": "buildings",
    "CK": "clock",
    "VP": "vrrooomm",
    "VD": "vrrooomm",
}
NEXT_CLOCK = {"house": "office", "office": "pub", "pub": "house"}

# Phase-coverage fixtures (tests/scenarios), keyed by the phase each rests at.
FIXTURES: dict[str, list[str]] = {
    "SB": ["setup_sb_midway", "fresh_4p"],
    "L1": ["setup_l1_mid"],
    "L2": ["setup_l2_mid"],
    "SEL": ["sel_round1", "sel_pass1_mid", "sel_pass2_dropped"],
    "LN": ["resolve_ln", "lines/res_bite_power3", "lines/five_player_shared", "lines/full_loop",
           "lines/shared_end", "lines/three_player_per_end", "lines/res_two_markers_widen"],
    "PX": ["resolve_px"],
    "BD": ["resolve_bd"],
    "CK": ["resolve_ck", "clock/ck_last_stone_3p"],
    "VP": ["resolve_vp"],
    "VD": ["resolve_vd"],
    "END": ["end_both", "end_last_stone", "end_sites_full", "end/returns_margin_3p"],
}  # fmt: skip
ALL_FIXTURES = [(ph, f) for ph, fs in FIXTURES.items() for f in fs]


def dump(s: Any) -> Json:
    return P.dump(s)


def phase_of(s: Any) -> str:
    return dump(s)["phase"]


def jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))


def reachable_vp() -> Any:
    """A reachable Vrrooomm! pick state: vrrooomm/base with P1's marker on letter a, one move."""
    d = H.load_doc("vrrooomm/base")
    d["seats"][1]["markers_left"] = 19
    d["phase"] = "VP"
    d["action_board"] = {
        "lines": [None] * 6, "buses": [None], "passengers": [None] * 6,
        "buildings": [None] * 6, "clock": [None], "vrrooomm": [1] + [None] * 5,
        "starting_player": [None],
    }  # fmt: skip
    d["pending"] = {"cursor": {"letter": 0}, "moves_left": 1}
    d["current_player"] = 1
    return H.load_doc_state(d)


def reachable_vd() -> Any:
    """A VD state with legal deliver_to ids: reachable_vp after a legal PICK_PASSENGER."""
    s = reachable_vp()
    s.apply_action(s.legal_actions()[0])
    assert phase_of(s) == "VD"
    return s


@pytest.fixture(scope="module")
def adapter() -> BusAdapter:
    return BusAdapter()


@pytest.fixture(scope="module")
def playout_states() -> list[Any]:
    """Sampled states of seeded random games, 3/4/5 players, incl. the initial and END states."""
    out: list[Any] = []
    for n, seed in [(3, 1), (4, 2), (5, 3), (4, 11)]:
        out += P.random_game(n, seed, every=9)
    return out


@pytest.fixture(scope="module")
def fixture_states() -> list[tuple[str, str, Any]]:
    out = [(ph, f, H.load_fixture(f)) for ph, f in ALL_FIXTURES]
    out += [("VP", "reachable_vp", reachable_vp()), ("VD", "reachable_vd", reachable_vd())]
    return out


def all_states(playout_states: list[Any], fixture_states: list[tuple[str, str, Any]]) -> list[Any]:
    return playout_states + [s for _, _, s in fixture_states]


# ---------------------------------------------------------------- 1. load


def test_adapter_name(adapter: BusAdapter) -> None:
    assert adapter.name == "bus"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_load_players(adapter: BusAdapter, n: int) -> None:
    g = adapter.load({"num_players": n})
    assert g.num_players() == n


def test_load_default_is_four_players(adapter: BusAdapter) -> None:
    assert adapter.load({}).num_players() == 4


@pytest.mark.parametrize("n", [2, 6, 0, 1, 9])
def test_load_rejects_bad_player_counts(adapter: BusAdapter, n: int) -> None:
    with pytest.raises(ValueError):
        adapter.load({"num_players": n})


@pytest.mark.parametrize("ret", ["win_loss", "margin"])
def test_load_returns_passes_through(adapter: BusAdapter, ret: str) -> None:
    g = adapter.load({"num_players": 3, "returns": ret})
    assert dump(g.new_initial_state())["parameters"] == {"players": 3, "returns": ret}


# ---------------------------------------------------------------- 2. state (6.1)


def test_view_has_four_keys(adapter: BusAdapter) -> None:
    v = adapter.view(P.game(4).new_initial_state(), 0)
    assert set(v) == {"state", "derived", "prompt", "targets"}


def test_view_is_json_serialisable(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    for s in all_states(playout_states, fixture_states):
        json.dumps(adapter.view(s, 0))


def test_view_same_for_every_seat(adapter: BusAdapter, playout_states: list[Any]) -> None:
    """Bus is perfect information: the view does not depend on the viewing seat."""
    for s in playout_states[::5]:
        views = [adapter.view(s, seat) for seat in range(s.get_game().num_players())]
        assert all(v == views[0] for v in views)


def test_state_round_trip_playouts(adapter: BusAdapter, playout_states: list[Any]) -> None:
    """UI_DESIGN 6.1: loading view.state gives a state with the same observer string."""
    assert any(s.is_terminal() for s in playout_states)
    for s in playout_states:
        doc = copy.deepcopy(adapter.view(s, 0)["state"])
        back = H.load_doc_state(doc, s.get_game())
        assert str(back) == str(s)
        assert back.legal_actions() == s.legal_actions()


@pytest.mark.parametrize(("phase", "name"), ALL_FIXTURES)
def test_state_round_trip_fixtures(adapter: BusAdapter, phase: str, name: str) -> None:
    s = H.load_fixture(name)
    assert phase_of(s) == phase
    doc = copy.deepcopy(adapter.view(s, 0)["state"])
    back = H.load_doc_state(doc, s.get_game())
    assert str(back) == str(s)
    assert back.legal_actions() == s.legal_actions()


def test_state_has_no_history_or_expect(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states[::4]:
        st = adapter.view(s, 0)["state"]
        assert "history" not in st
        assert "expect" not in st
        assert st["format"] == 1


def test_state_top_level_keys(adapter: BusAdapter, playout_states: list[Any]) -> None:
    required = {"format", "parameters", "round", "phase", "current_player", "start_player",
                "clock", "stones_on_clock", "seats", "buildings", "waiting", "supply",
                "action_board", "pending", "move_offset"}  # fmt: skip
    for s in playout_states[::3]:
        st = adapter.view(s, 0)["state"]
        # 6.5.11: current_player is absent at END (the loader rejects it there)
        need = required - {"current_player"} if s.is_terminal() else required
        assert need <= set(st)
        assert set(st) <= required | {"end"}
        assert ("end" in st) == s.is_terminal()
        assert st["phase"] == phase_of(s)


def test_state_canonical_forms(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states:
        st = adapter.view(s, 0)["state"]
        n = s.get_game().num_players()
        assert isinstance(st["seats"], list)
        assert len(st["seats"]) == n
        sites = [b["site"] for b in st["buildings"]]
        assert sites == sorted(sites)
        assert len(set(sites)) == len(sites)
        for k, c in st["waiting"].items():
            assert isinstance(k, str)
            assert int(k) >= 0
            assert isinstance(c, int)
            assert c >= 1
        ab = st["action_board"]
        assert list(ab) == ACTION_KEYS or set(ab) == set(ACTION_KEYS)
        for k in ACTION_KEYS:
            assert len(ab[k]) == SPACES[k]
            assert all(x is None or (isinstance(x, int) and 0 <= x < n) for x in ab[k])


def test_state_parameters_follow_game(adapter: BusAdapter) -> None:
    for n in (3, 4, 5):
        s = P.game(n, returns="margin").new_initial_state()
        assert adapter.view(s, 0)["state"]["parameters"] == {"players": n, "returns": "margin"}


def test_state_move_offset_is_history_length_plus_offset(adapter: BusAdapter) -> None:
    """6.1: move_offset = scenario move number of the viewed position."""
    s = H.load_fixture("sel_pass1_mid")
    base = adapter.view(s, 0)["state"]["move_offset"]
    s2 = s.clone()
    s2.apply_action(s2.legal_actions()[0])
    assert adapter.view(s2, 0)["state"]["move_offset"] == base + 1


def test_state_pending_matches_engine(adapter: BusAdapter, fixture_states: Any) -> None:
    for _, _, s in fixture_states:
        assert adapter.view(s, 0)["state"]["pending"] == dump(s)["pending"]


# ---------------------------------------------------------------- 3. derived (6.2)


def test_derived_keys(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states[::5]:
        d = adapter.view(s, 0)["derived"]
        assert set(d) == {"seats", "street_lines", "open_tier", "resolving", "result"}
        assert len(d["seats"]) == s.get_game().num_players()
        for ds in d["seats"]:
            assert set(ds) == {"line_markers_left", "line_nodes", "ends", "dropped"}


def test_derived_line_facts(adapter: BusAdapter, playout_states: list[Any]) -> None:
    saw_line = saw_empty = False
    for s in playout_states:
        v = adapter.view(s, 0)
        for seat, ds in zip(v["derived"]["seats"], v["state"]["seats"], strict=True):
            line = ds["line"]
            assert seat["line_markers_left"] == 25 - len(line)
        for seat, st_seat in zip(v["derived"]["seats"], v["state"]["seats"], strict=True):
            line = st_seat["line"]
            if not line:
                saw_empty = True
                assert seat["line_nodes"] == []
                assert seat["ends"] is None
            else:
                saw_line = True
                nodes = seat["line_nodes"]
                assert len(nodes) == len(line) + 1
                assert seat["ends"] == [nodes[0], nodes[-1]]
                assert all(isinstance(x, int) for x in nodes)
    assert saw_line
    assert saw_empty  # the initial states have no lines


def test_derived_line_nodes_start_at_street_a_for_one_street_line(adapter: BusAdapter) -> None:
    """ACTIONS 1.3 orientation: end 0 of a one-street line is the street's `a` node, so the two
    ends of a single street are distinct and consecutive line_nodes."""
    s = H.load_fixture("resolve_ln")
    d = adapter.view(s, 0)["derived"]["seats"]
    for st_seat, ds in zip(s_seats(adapter, s), d, strict=True):
        if len(st_seat["line"]) == 1:
            assert len(ds["line_nodes"]) == 2
            assert ds["ends"][0] != ds["ends"][1]


def s_seats(adapter: BusAdapter, s: Any) -> list[Json]:
    return adapter.view(s, 0)["state"]["seats"]


def test_derived_street_lines(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    shared_seen = False
    for s in all_states(playout_states, fixture_states):
        v = adapter.view(s, 0)
        expect: dict[str, list[int]] = {}
        for i, seat in enumerate(v["state"]["seats"]):
            for st in seat["line"]:
                expect.setdefault(str(st), []).append(i)
        assert v["derived"]["street_lines"] == expect
        for seats in expect.values():
            assert seats == sorted(seats)
            assert len(set(seats)) == len(seats)
            shared_seen |= len(seats) > 1
    assert shared_seen


def test_derived_dropped_in_sel(adapter: BusAdapter) -> None:
    s = H.load_fixture("sel_pass2_dropped")
    dr = [x["dropped"] for x in adapter.view(s, 0)["derived"]["seats"]]
    assert dr == [i in dump(s)["pending"]["dropped"] for i in range(4)]
    assert all(isinstance(x, bool) for x in dr)


def test_derived_dropped_all_bool(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states[::4]:
        for x in adapter.view(s, 0)["derived"]["seats"]:
            assert isinstance(x["dropped"], bool)


def test_derived_open_tier_initial_is_a(adapter: BusAdapter) -> None:
    assert adapter.view(P.game(4).new_initial_state(), 0)["derived"]["open_tier"] == "A"


def test_derived_open_tier_matches_build_actions(
    adapter: BusAdapter, playout_states: list[Any]
) -> None:
    """R-010: every legal build is on the lowest open tier, which is `open_tier`."""
    checked = 0
    for s in playout_states:
        if s.is_terminal():
            continue
        metas = [bus_game.action_meta(s, a) for a in s.legal_actions()]
        tiers = {m["tier"] for m in metas if m["kind"] == "build"}
        if tiers:
            checked += 1
            assert tiers == {adapter.view(s, 0)["derived"]["open_tier"]}
    assert checked >= 3


@pytest.mark.parametrize(("built", "tier"), [(12, "B"), (23, "C"), (32, "D"), (47, None)])
def test_derived_open_tier_by_fill(adapter: BusAdapter, built: int, tier: str | None) -> None:
    """R-010: sites 0..46 are tier-ordered (A 12, B 11, C 9, D 15 per data/board.yaml)."""
    doc = H.load_doc("sel_round1")
    doc["buildings"] = [{"site": i, "type": "house"} for i in range(built)]
    s = H.load_doc_state(doc)
    assert adapter.view(s, 0)["derived"]["open_tier"] == tier


def test_derived_open_tier_letters(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states:
        assert adapter.view(s, 0)["derived"]["open_tier"] in ("A", "B", "C", "D", None)


def expected_power(s: Any) -> int | None:
    """Hand-computed from R-031 (M - k), R-032 (M = max buses now), R-033 (+1 for Lines at 5p)."""
    d = dump(s)
    if d["phase"] not in ("LN", "PX", "BD"):
        return None
    m = max(x["buses"] for x in d["seats"])
    p = m - d["pending"]["cursor"]["letter"]
    if d["phase"] == "LN" and d["parameters"]["players"] == 5:
        p += 1
    return p


@pytest.mark.parametrize(
    ("name", "power"),
    [
        ("lines/res_bite_power3", 3),  # M=3, letter a
        (
            "lines/five_player_shared",
            2,
        ),  # M=1, letter a, 5p Lines +1 (units_left is 1 mid-resolution)
        ("resolve_px", 1),  # M=1, letter a: PX has no +1
    ],
)
def test_derived_resolving_power_by_hand(adapter: BusAdapter, name: str, power: int) -> None:
    """R-031/R-033: `power` is the marker's power, not the units still to do."""
    s = H.load_fixture(name)
    r = adapter.view(s, 0)["derived"]["resolving"]
    assert r["power"] == power
    assert expected_power(s) == power


@pytest.mark.parametrize(
    ("name", "action", "letter", "owner", "power"),
    [
        ("resolve_px", "passengers", 0, 2, 1),
        ("resolve_bd", "buildings", 0, 1, 1),
        ("resolve_ck", "clock", 0, 3, None),
        ("resolve_vp", "vrrooomm", 1, 1, None),
        ("resolve_vd", "vrrooomm", 1, 1, None),
    ],
)
def test_derived_resolving_fields(
    adapter: BusAdapter, name: str, action: str, letter: int, owner: int, power: int | None
) -> None:
    s = H.load_fixture(name)
    r = adapter.view(s, 0)["derived"]["resolving"]
    assert set(r) == {"action", "letter", "owner", "power"}
    assert r["action"] == action
    assert r["letter"] == letter
    assert r["power"] == power
    d = dump(s)
    assert r["owner"] == d["action_board"][action][letter] == d["current_player"]


def test_derived_resolving_null_outside_resolution(
    adapter: BusAdapter, fixture_states: Any
) -> None:
    for ph, _, s in fixture_states:
        r = adapter.view(s, 0)["derived"]["resolving"]
        if ph in RESOLVE_ACTION:
            assert r is not None
        else:
            assert r is None


def test_derived_resolving_in_playouts(adapter: BusAdapter, playout_states: list[Any]) -> None:
    seen = set()
    for s in playout_states:
        d = dump(s)
        r = adapter.view(s, 0)["derived"]["resolving"]
        ph = d["phase"]
        if ph not in RESOLVE_ACTION:
            assert r is None
            continue
        seen.add(ph)
        cur = d["pending"]["cursor"]["letter"]
        assert r["action"] == RESOLVE_ACTION[ph]
        assert r["letter"] == cur
        assert r["owner"] == d["current_player"] == d["action_board"][r["action"]][cur]
        assert r["power"] == expected_power(s)
        if ph in ("LN", "PX", "BD"):
            assert r["power"] >= 1  # power <= 0 markers produce no decision node
            assert d["pending"]["units_left"] <= r["power"]
    assert seen & {"LN", "PX", "BD"}


def test_derived_result_null_before_end(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states:
        r = adapter.view(s, 0)["derived"]["result"]
        if s.is_terminal():
            assert r is not None
        else:
            assert r is None


def test_derived_result_equals_game_result_at_end(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    ends = [s for s in all_states(playout_states, fixture_states) if s.is_terminal()]
    assert len(ends) >= 5
    for s in ends:
        r = adapter.view(s, 0)["derived"]["result"]
        assert r == jsonable(bus_game.game_result(s))
        assert set(r) == {"end_types", "winners", "final_scores", "returns"}


# ---------------------------------------------------------------- 4. prompt (6.3)


def test_prompt_keys(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states[::4]:
        assert set(adapter.view(s, 0)["prompt"]) == {
            "phase", "actor", "units_left", "power", "context"
        }  # fmt: skip


def test_prompt_phase_actor(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states:
        p = adapter.view(s, 0)["prompt"]
        assert p["phase"] == phase_of(s)
        assert p["actor"] == (None if s.is_terminal() else s.current_player())
        if s.is_terminal():
            assert p["phase"] == "END"
            assert p["context"] == {}
            assert p["units_left"] is None
            assert p["power"] is None


def test_prompt_units_left_and_power(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states:
        d = dump(s)
        p = adapter.view(s, 0)["prompt"]
        pend = d["pending"]
        want = pend.get("units_left", pend.get("moves_left"))
        assert p["units_left"] == want
        r = adapter.view(s, 0)["derived"]["resolving"]
        assert p["power"] == (None if r is None else r["power"])


@pytest.mark.parametrize(
    ("name", "units", "power"),
    [
        ("lines/res_bite_power3", 3, 3),
        ("lines/five_player_shared", 1, 2),
        ("resolve_px", 3, 1),
        ("resolve_bd", 2, 1),
        ("resolve_ck", None, None),
        ("resolve_vp", 1, None),
        ("resolve_vd", 1, None),
        ("sel_round1", None, None),
        ("setup_sb_midway", None, None),
    ],
)
def test_prompt_units_power_fixtures(
    adapter: BusAdapter, name: str, units: int | None, power: int | None
) -> None:
    p = adapter.view(H.load_fixture(name), 0)["prompt"]
    assert p["units_left"] == units
    assert p["power"] == power


@pytest.mark.parametrize(
    ("name", "phase", "done"),
    [("setup_sb_midway", "SB", 3), ("setup_l1_mid", "L1", 2), ("setup_l2_mid", "L2", 1)],
)
def test_prompt_context_setup(adapter: BusAdapter, name: str, phase: str, done: int) -> None:
    p = adapter.view(H.load_fixture(name), 0)["prompt"]
    assert p["phase"] == phase
    assert p["context"] == {"done": done}


def test_prompt_context_setup_actor(adapter: BusAdapter) -> None:
    """R-016: each seat places two buildings in a row, so done=3 means seat 1 (start seat 0)."""
    p = adapter.view(H.load_fixture("setup_sb_midway"), 0)["prompt"]
    assert p["actor"] == 1


@pytest.mark.parametrize(
    ("name", "pass_", "mandatory"),
    [("sel_pass1_mid", 1, True), ("sel_pass2_dropped", 2, False)],
)
def test_prompt_context_sel(adapter: BusAdapter, name: str, pass_: int, mandatory: bool) -> None:
    s = H.load_fixture(name)
    p = adapter.view(s, 0)["prompt"]
    assert set(p["context"]) == {"pass", "mandatory", "markers_left"}
    assert p["context"]["pass"] == pass_
    assert p["context"]["mandatory"] is mandatory
    assert isinstance(p["context"]["markers_left"], int)


def test_prompt_context_sel_round_start(adapter: BusAdapter) -> None:
    p = adapter.view(H.load_fixture("sel_round1"), 0)["prompt"]
    assert p["context"]["pass"] == 0
    assert p["context"]["mandatory"] is True
    assert p["context"]["markers_left"] == 20  # actor's hand: nothing placed yet this game


def test_prompt_context_px(adapter: BusAdapter) -> None:
    s = H.load_fixture("resolve_px")
    p = adapter.view(s, 0)["prompt"]
    assert p["context"] == {"supply": dump(s)["supply"]}


def test_prompt_context_bd(adapter: BusAdapter) -> None:
    s = H.load_fixture("resolve_bd")
    v = adapter.view(s, 0)
    assert v["prompt"]["context"] == {"open_tier": v["derived"]["open_tier"]}


def test_prompt_context_ck(adapter: BusAdapter) -> None:
    s = H.load_fixture("resolve_ck")
    d = dump(s)
    c = adapter.view(s, 0)["prompt"]["context"]
    assert c == {
        "clock": d["clock"],
        "next_clock": NEXT_CLOCK[d["clock"]],
        "stones_on_clock": d["stones_on_clock"],
        "last_stone": d["stones_on_clock"] == 1,
    }


def test_prompt_context_vp_vd(adapter: BusAdapter) -> None:
    sp = H.load_fixture("resolve_vp")
    dp = dump(sp)
    assert adapter.view(sp, 0)["prompt"]["context"] == {
        "clock": dp["clock"],
        "moves_left": dp["pending"]["moves_left"],
        "picked_node": None,
    }
    sd = H.load_fixture("resolve_vd")
    dd = dump(sd)
    assert adapter.view(sd, 0)["prompt"]["context"] == {
        "clock": dd["clock"],
        "moves_left": dd["pending"]["moves_left"],
        "picked_node": dd["pending"]["picked_node"],
    }
    assert dd["pending"]["picked_node"] == 16


def test_prompt_context_end(adapter: BusAdapter) -> None:
    p = adapter.view(H.load_fixture("end_both"), 0)["prompt"]
    assert p["phase"] == "END"
    assert p["actor"] is None
    assert p["context"] == {}


def test_prompt_context_keys_over_playouts(adapter: BusAdapter, playout_states: list[Any]) -> None:
    keys = {
        "SB": {"done"}, "L1": {"done"}, "L2": {"done"},
        "SEL": {"pass", "mandatory", "markers_left"},
        "PX": {"supply"}, "BD": {"open_tier"},
        "CK": {"clock", "next_clock", "stones_on_clock", "last_stone"},
        "VP": {"clock", "moves_left", "picked_node"},
        "VD": {"clock", "moves_left", "picked_node"},
        "END": set(),
    }  # fmt: skip
    seen = set()
    for s in playout_states:
        p = adapter.view(s, 0)["prompt"]
        seen.add(p["phase"])
        if p["phase"] in keys:
            assert set(p["context"]) == keys[p["phase"]]
        else:
            assert p["phase"] == "LN"  # LN context is not specified in 6.3 (see report)
            assert isinstance(p["context"], dict)
        if p["phase"] == "SEL":
            assert p["context"]["mandatory"] == (p["context"]["pass"] < 2)
        if p["phase"] == "VP":
            assert p["context"]["picked_node"] is None
        if p["phase"] == "VD":
            assert isinstance(p["context"]["picked_node"], int)
    assert {"SB", "L1", "L2", "SEL", "END"} <= seen


# ---------------------------------------------------------------- 5. targets (6.4)

GROUPS = {"streets", "street_ends", "sites", "stations", "nodes", "actions", "clock", "buttons"}


def ids_of(v: Any) -> list[int]:
    """Flatten an id or a list of ids."""
    if v is None:
        return []
    return [v] if isinstance(v, int) else list(v)


def flatten(t: Json) -> list[int]:
    out: list[int] = []
    for g in ("streets", "sites", "stations", "nodes", "actions"):
        for v in t[g].values():
            out += ids_of(v)
    out += ids_of(t["clock"]["advance"]) + ids_of(t["clock"]["pause"])
    out += [b["id"] for b in t["buttons"]]
    return out


def test_targets_group_keys(adapter: BusAdapter, playout_states: list[Any]) -> None:
    for s in playout_states[::4]:
        t = adapter.view(s, 0)["targets"]
        assert set(t) == GROUPS
        assert set(t["clock"]) == {"advance", "pause"}
        assert isinstance(t["buttons"], list)


def test_targets_union_equals_legal_actions(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    for s in all_states(playout_states, fixture_states):
        ids = flatten(adapter.view(s, 0)["targets"])
        assert len(ids) == len(set(ids)), "an id appears in two groups or twice"
        assert sorted(ids) == sorted(s.legal_actions())


def test_targets_group_placement(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    kinds_seen: set[str] = set()
    for s in all_states(playout_states, fixture_states):
        if s.is_terminal():
            continue
        t = adapter.view(s, 0)["targets"]
        for a in s.legal_actions():
            m = bus_game.action_meta(s, a)
            k = m["kind"]
            kinds_seen.add(k)
            if k == "extend":
                assert a in ids_of(t["streets"].get(str(m["street"])))
            elif k == "build":
                assert a in ids_of(t["sites"].get(str(m["site"])))
            elif k == "deliver_to":
                assert ids_of(t["sites"].get(str(m["site"]))) == [a]
            elif k == "passenger_to":
                assert ids_of(t["stations"].get(str(m["node"]))) == [a]
            elif k == "pick_passenger":
                assert ids_of(t["nodes"].get(str(m["node"]))) == [a]
            elif k == "select":
                assert ids_of(t["actions"].get(ACTION_KEYS[m["action"]])) == [a]
            elif k == "clock_advance":
                assert t["clock"]["advance"] == a
            elif k == "clock_pause":
                assert t["clock"]["pause"] == a
            elif k == "drop_out":
                assert {"id": a, "kind": "drop_out"} in t["buttons"]
            else:
                pytest.fail(f"unexpected kind {k}")
    assert {"build", "extend", "select", "passenger_to", "pick_passenger", "deliver_to",
            "clock_advance", "clock_pause", "drop_out"} <= kinds_seen  # fmt: skip


def test_targets_group_keys_are_strings(adapter: BusAdapter, fixture_states: Any) -> None:
    for _, _, s in fixture_states:
        t = adapter.view(s, 0)["targets"]
        for g in ("streets", "sites", "stations", "nodes"):
            assert all(isinstance(k, str) and k.isdigit() for k in t[g])
        assert set(t["actions"]) <= set(ACTION_KEYS)


def test_targets_streets_loop_case_has_two_ends(adapter: BusAdapter) -> None:
    """R-041: a street legal from both ends of a loop has two ids, distinguished by `end`."""
    s = H.load_fixture("lines/full_loop")
    t = adapter.view(s, 0)["targets"]
    two = {k: v for k, v in t["streets"].items() if len(ids_of(v)) == 2}
    assert two, "full_loop fixture should offer a street from both ends"
    for v in two.values():
        ends = {bus_game.action_meta(s, a)["end"] for a in ids_of(v)}
        assert ends == {0, 1}
    for v in t["streets"].values():
        assert 1 <= len(ids_of(v)) <= 2


def test_targets_build_has_one_id_per_type(adapter: BusAdapter) -> None:
    s = H.load_fixture("resolve_bd")
    t = adapter.view(s, 0)["targets"]
    assert t["sites"]
    for ids in t["sites"].values():
        types = {bus_game.action_meta(s, a)["building_type"] for a in ids_of(ids)}
        assert types == {"house", "office", "pub"}
        assert len(ids_of(ids)) == 3


def test_targets_clock_both_ids(adapter: BusAdapter) -> None:
    t = adapter.view(H.load_fixture("resolve_ck"), 0)["targets"]
    assert isinstance(t["clock"]["advance"], int)
    assert isinstance(t["clock"]["pause"], int)
    assert t["buttons"] == []


def test_targets_clock_null_outside_ck(adapter: BusAdapter) -> None:
    t = adapter.view(H.load_fixture("sel_round1"), 0)["targets"]
    assert t["clock"] == {"advance": None, "pause": None}
    assert len(t["actions"]) >= 1


def test_targets_drop_out_button(adapter: BusAdapter) -> None:
    s = H.load_fixture("sel_pass2_dropped")
    t = adapter.view(s, 0)["targets"]
    drops = [a for a in s.legal_actions() if bus_game.action_meta(s, a)["kind"] == "drop_out"]
    assert len(drops) == 1
    assert t["buttons"] == [{"id": drops[0], "kind": "drop_out"}]


def test_targets_no_drop_out_in_mandatory_pass(adapter: BusAdapter) -> None:
    """R-023: no drop out in the two mandatory passes."""
    t = adapter.view(H.load_fixture("sel_round1"), 0)["targets"]
    assert t["buttons"] == []
    assert t["streets"] == {}
    assert t["sites"] == {}


def test_targets_pick_and_deliver(adapter: BusAdapter) -> None:
    tp = adapter.view(H.load_fixture("resolve_vp"), 0)["targets"]
    assert tp["nodes"]
    assert tp["sites"] == {} and tp["streets"] == {}
    tp = adapter.view(reachable_vp(), 0)["targets"]
    assert tp["nodes"]
    s = reachable_vd()
    td = adapter.view(s, 0)["targets"]
    assert td["sites"]
    assert sorted(flatten(td)) == sorted(s.legal_actions())
    assert td["nodes"] == {}


def test_targets_passenger_stations(adapter: BusAdapter) -> None:
    s = H.load_fixture("resolve_px")
    t = adapter.view(s, 0)["targets"]
    assert len(t["stations"]) == 2  # the board has two stations (R-045)
    assert sorted(flatten(t)) == sorted(s.legal_actions())


def test_targets_end_all_empty(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    ends = [s for s in all_states(playout_states, fixture_states) if s.is_terminal()]
    assert ends
    for s in ends:
        t = adapter.view(s, 0)["targets"]
        assert t["streets"] == {}
        assert t["sites"] == {}
        assert t["stations"] == {}
        assert t["nodes"] == {}
        assert t["actions"] == {}
        assert t["clock"] == {"advance": None, "pause": None}
        assert t["buttons"] == []


# ---------------------------------------------------------------- 6. labels / action_meta


def test_labels_match_engine(
    adapter: BusAdapter, playout_states: list[Any], fixture_states: Any
) -> None:
    n = 0
    for s in all_states(playout_states, fixture_states):
        if s.is_terminal():
            continue
        for a in s.legal_actions():
            m = adapter.action_meta(s, a)
            eng = bus_game.action_meta(s, a)
            assert isinstance(m["label"], str)
            assert m["label"]
            assert m["label"] == eng["label"]
            assert m["kind"] == eng["kind"]
            n += 1
    assert n > 100


def test_labels_use_zero_based_seat_names(adapter: BusAdapter, playout_states: list[Any]) -> None:
    seen_p0 = False
    for s in playout_states:
        if s.is_terminal():
            continue
        cur = s.current_player()
        for a in s.legal_actions():
            lab = adapter.action_meta(s, a)["label"]
            assert "Player 1" not in lab
            assert not re.search(r"Player \d", lab)
            assert lab.startswith(f"P{cur}: ")
            seen_p0 |= cur == 0
    assert seen_p0


def test_action_meta_params_content(adapter: BusAdapter) -> None:
    """ACTIONS 5 keys are available through `action_meta` (flattened or under `params`)."""

    def get(m: Json, key: str) -> Any:
        return m["params"][key] if key in m.get("params", {}) else m[key]

    s = H.load_fixture("resolve_bd")
    a = s.legal_actions()[0]
    m = adapter.action_meta(s, a)
    assert m["kind"] == "build"
    for key in ("site", "node", "tier", "building_type"):
        assert get(m, key) == bus_game.action_meta(s, a)[key]
    s = H.load_fixture("sel_round1")
    m = adapter.action_meta(s, s.legal_actions()[0])
    assert m["kind"] == "select"
    assert get(m, "action_name") == bus_game.action_meta(s, s.legal_actions()[0])["action_name"]


def test_labels_cover_sample_formats(adapter: BusAdapter) -> None:
    """ACTIONS 4 spot checks on fixtures."""
    s = H.load_fixture("resolve_ck")
    labs = {adapter.action_meta(s, a)["label"] for a in s.legal_actions()}
    cur = s.current_player()
    assert labs == {f"P{cur}: clock advance", f"P{cur}: clock pause"}
    s = H.load_fixture("sel_round1")
    cur = s.current_player()
    labs = {adapter.action_meta(s, a)["label"] for a in s.legal_actions()}
    assert f"P{cur}: select Lines" in labs
    assert f"P{cur}: select Vrrooomm!" in labs
    assert f"P{cur}: select Starting Player" in labs
    s = H.load_fixture("resolve_px")
    labs = {adapter.action_meta(s, a)["label"] for a in s.legal_actions()}
    cur = s.current_player()
    assert all(lab.startswith(f"P{cur}: passenger to station n") for lab in labs)
