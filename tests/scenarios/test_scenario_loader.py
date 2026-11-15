"""Scenario loader (docs/SCENARIOS.md; ACTIONS §1.3, §1.7, §3).

Expected API of the scenario module
  load_scenario(doc: dict | str | Path, game=None) -> state   applies ``history``
  dump_scenario(state) -> dict                                fully resolved, JSON-serialisable
  check_expect(state, expect: dict) -> None                   raises AssertionError on mismatch
  ScenarioError(code, message)                                ``.code`` is an ``S-*`` code
A ``str`` argument is read as a file path (JSON or YAML, by content/extension).

Fixtures are JSON in this directory. ``invalid_cases.json`` holds generated documents:
``cases`` (must fail with the given code), ``ambiguous`` (SCENARIOS.md names no single code, so any
listed code is accepted and the ambiguity is reported), ``controls`` (boundary documents that must
load). Everything here is pending until the loader lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyspiel
import pytest
import yaml

from bus import constants as C
from bus.actions import encode_build
from bus.board import load_board
from tests import harness as H

B = load_board()
CASES = json.loads((H.SCENARIO_DIR / "invalid_cases.json").read_text())
FIXTURES = H.fixture_names()
ROUND_TRIP = [n for n in FIXTURES if n != "unreachable_tier_skip"]
TIER_A = [t for t, s in enumerate(B.sites) if s.tier == "A"]

pending = pytest.mark.pending


def _load(doc: Any, game: Any = None) -> Any:
    return H.scenario_module().load_scenario(doc, game)


def _dump(state: Any) -> dict[str, Any]:
    return H.scenario_module().dump_scenario(state)


def _plain(doc: dict[str, Any]) -> dict[str, Any]:
    """Copy of a fixture document without its expect block."""
    d = json.loads(json.dumps(doc))
    d.pop("expect", None)
    return d


def _initial(n: int = 4, **kw: Any) -> Any:
    import bus.game  # noqa: F401

    return pyspiel.load_game("python_bus", {"players": n, **kw}).new_initial_state()


# ---------------------------------------------------------------- fresh / defaults (§7, §2)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_SCN7_fresh_scenario_equals_initial_state(n: int) -> None:
    """SCENARIOS §7: {"format":1} (with players) is the state right after R-014: round 0, SB,
    done 0, seat 0 to act, R-015 passengers; identical to the game's initial state."""
    s = _load({"format": 1, "parameters": {"players": n}})
    ref = _initial(n)
    assert s.round == 0 and s.phase == "SB"
    assert s.current_player() == 0
    assert str(s) == str(ref)
    assert s.legal_actions() == ref.legal_actions()
    assert s.stones_on_clock == ref.stones_on_clock == C.time_stones(n)
    assert s.passenger_supply == 11
    for node in B.roundabouts:
        assert s.waiting_at(node) == 1


def test_SCN7_bare_format_defaults_to_four_players() -> None:
    """SCENARIOS §2: players defaults to 4 (R-001)."""
    s = _load({"format": 1})
    assert s.get_game().num_players() == 4
    assert str(s) == str(_initial(4))


def test_SCN7_fresh_fixture_runs_expect() -> None:
    """SCENARIOS §8 fresh_4p: its expect block (phase SB, seat 0, not terminal) holds."""
    s = H.run_fixture("fresh_4p")
    assert s.phase == "SB"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_SCN2_parameters_players_reach_game(n: int) -> None:
    """SCENARIOS §2 parameters: ``players`` selects the game's player count."""
    s = _load({"format": 1, "parameters": {"players": n}})
    assert s.get_game().num_players() == n


def test_SCN2_parameters_returns_margin_reaches_game() -> None:
    """SCENARIOS §2 parameters: ``returns`` selects the returns mode; default win_loss."""
    s = _load({"format": 1, "parameters": {"players": 4, "returns": "margin"}})
    assert s.get_game().get_parameters()["returns"] == "margin"
    assert _load({"format": 1}).get_game().get_parameters()["returns"] == "win_loss"


def test_SCN2_round1_shorthand_defaults() -> None:
    """SCENARIOS §2/§7: omitted phase -> SEL, current_player -> seat 0 (start player), clock
    house, stones 5, supply derived (15 - 4 on board), seats default to 1 bus / 20 markers / 0
    points / 0 stones; dump fills move_offset 0."""
    s = _load(_plain(H.load_doc("sel_round1")))
    assert s.round == 1 and s.phase == "SEL"
    assert s.current_player() == 0
    assert s.clock == "house"
    assert s.stones_on_clock == 5
    assert s.passenger_supply == 11
    for p in range(4):
        v = s.seat(p)
        assert (v.points, v.stones, v.buses) == (0, 0, 1)
        assert v.action_markers == 20
        assert len(v.line) == 2
        assert v.line_markers == C.LINE_MARKERS_PER_PLAYER - 2
    assert [list(x) for x in s.action_board] == [[] for _ in range(7)]
    assert _dump(s)["move_offset"] == 0


def test_SCN2_three_player_default_stones() -> None:
    """SCENARIOS §2 stones_on_clock: default 4 with three players."""
    assert _load({"format": 1, "parameters": {"players": 3}}).stones_on_clock == 4


def test_SCN2_seats_as_map_equal_to_list() -> None:
    """SCENARIOS §2 seats: a map "0"... is accepted and equals the list form."""
    doc = _plain(H.load_doc("sel_round1"))
    as_map = json.loads(json.dumps(doc))
    as_map["seats"] = {str(i): seat for i, seat in enumerate(doc["seats"])}
    assert str(_load(as_map)) == str(_load(doc))


def test_SCN2_empty_seat_objects_use_defaults() -> None:
    """SCENARIOS §3.1: every seat field has a default (``{}`` is a fresh seat)."""
    s = _load({"format": 1, "seats": [{}, {}, {}, {}]})
    assert str(s) == str(_initial(4))


def test_SCN3_markers_left_and_bus_line_markers_derived() -> None:
    """SCENARIOS §3.1: markers_left is Action markers in hand;."""
    s = _load(_plain(H.load_doc("sel_pass1_mid")))
    assert [s.seat(p).action_markers for p in range(4)] == [18, 18, 19, 19]
    assert all(s.seat(p).line_markers == 25 - 2 for p in range(4))


# ---------------------------------------------------------------- move_offset (§2, §6 S-POINTS)


def _score_doc(**top: Any) -> dict[str, Any]:
    d = _plain(H.load_doc("sel_round1"))
    d["seats"][0].update(points=2, last_score_move=4)
    d["seats"][1].update(points=1, last_score_move=6)
    d.update(top)
    return d


def test_SCN2_move_offset_defaults_to_max_last_score_move() -> None:
    """SCENARIOS §2 move_offset: default = max of the seats' last_score_move."""
    assert _dump(_load(_score_doc()))["move_offset"] == 6


def test_SCN2_move_offset_zero_when_nobody_scored() -> None:
    """SCENARIOS §2 move_offset: 0 if all last_score_move are null."""
    assert _dump(_load(_plain(H.load_doc("sel_round1"))))["move_offset"] == 0


def test_SCN2_move_offset_override_is_kept() -> None:
    """SCENARIOS §2 move_offset: an explicit value (>= every last_score_move) is used."""
    assert _dump(_load(_score_doc(move_offset=12)))["move_offset"] == 12


def test_SCN2_move_offset_counts_post_load_history() -> None:
    """SCENARIOS §2 move_offset / ACTIONS §1.7: the first action after load has move number
    move_offset + 1, so after k history actions the position's own move number is offset + k.
    Assumption on dump: it emits the move number of the dumped position as ``move_offset``."""
    ids = [encode_build(TIER_A[i], 0) for i in range(3)]
    assert _dump(_load({"format": 1, "move_offset": 5, "history": ids}))["move_offset"] == 8
    assert _dump(_load({"format": 1, "history": ids}))["move_offset"] == 3


def test_SCN2_move_offset_zero_is_allowed() -> None:
    """SCENARIOS §2 move_offset: int >= 0 (zero valid)."""
    assert _dump(_load({"format": 1, "move_offset": 0}))["move_offset"] == 0


# ---------------------------------------------------------------- history / expect (§5)


@pytest.mark.parametrize("n", [3, 4])
def test_SCN5_history_is_applied_after_load(n: int) -> None:
    """SCENARIOS §5: history ids are applied after the scenario; the result equals playing them."""
    ids = [encode_build(TIER_A[i], i % 3) for i in range(3)]
    s = _load({"format": 1, "parameters": {"players": n}, "history": ids})
    ref = _initial(n)
    for a in ids:
        ref.apply_action(a)
    assert str(s) == str(ref)
    assert s.current_player() == ref.current_player() == 1
    assert s.legal_actions() == ref.legal_actions()


def test_SCN5_history_on_a_non_fresh_scenario() -> None:
    """SCENARIOS §5: history extends a mid-setup position (SB with 3 buildings + 1 build =
    SB with 4 buildings and done 4)."""
    doc = _plain(H.load_doc("setup_sb_midway"))
    s = _load({**doc, "history": [encode_build(TIER_A[3], 1)]})
    doc4 = {
        **doc,
        "pending": {"done": 4},
        "buildings": doc["buildings"] + [{"site": TIER_A[3], "type": "office"}],
    }
    s2 = _load(doc4)
    assert s.current_player() == s2.current_player() == 2
    assert s.legal_actions() == s2.legal_actions()
    assert all(s.site_building(t) == s2.site_building(t) for t in range(len(B.sites)))


def test_SCN5_check_expect_accepts_correct_values() -> None:
    """SCENARIOS §5: every listed expect key holds on the fresh 4p state."""
    s = _initial(4)
    legal = s.legal_actions()
    tier_b = next(t for t, x in enumerate(B.sites) if x.tier == "B")
    exp = {
        "phase": "SB",
        "current_player": 0,
        "terminal": False,
        "points": [0, 0, 0, 0],
        "buses": [1, 1, 1, 1],
        "stones": [0, 0, 0, 0],
        "clock": "house",
        "supply": 11,
        "start_player": 0,
        "legal_actions": legal,
        "legal_includes": [legal[0]],
        "waiting": {str(n): 1 for n in B.roundabouts},
        "legal_excludes": [encode_build(tier_b, 0)],
    }
    H.scenario_module().check_expect(s, exp)


@pytest.mark.parametrize(
    "bad",
    [
        {"phase": "SEL"},
        {"current_player": 1},
        {"terminal": True},
        {"points": [1, 0, 0, 0]},
        {"buses": [2, 1, 1, 1]},
        {"stones": [1, 0, 0, 0]},
        {"clock": "pub"},
        {"supply": 12},
        {"start_player": 2},
        {"legal_includes": [0]},
        {"legal_actions": [0, 1]},
        {"waiting": {"0": 1}},
    ],
    ids=lambda b: next(iter(b)),
)
def test_SCN5_check_expect_rejects_wrong_values(bad: dict[str, Any]) -> None:
    """SCENARIOS §5: a failing assertion raises AssertionError (never passes silently)."""
    with pytest.raises(AssertionError):
        H.scenario_module().check_expect(_initial(4), bad)


def test_SCN5_legal_excludes_rejects_a_legal_action() -> None:
    """SCENARIOS §5 legal_excludes: listing a legal id fails."""
    s = _initial(4)
    with pytest.raises(AssertionError):
        H.scenario_module().check_expect(s, {"legal_excludes": [s.legal_actions()[0]]})


@pytest.mark.parametrize(
    "name",
    FIXTURES,
)
def test_SCN5_fixture_expect_blocks_hold(name: str) -> None:
    """SCENARIOS §5/§8: every fixture loads and its own expect block holds (harness.run_fixture)."""
    H.run_fixture(name)


def test_SCN5_harness_load_fixture_by_name() -> None:
    """Harness: load_fixture(name) returns the state at the fixture position."""
    s = H.load_fixture("sel_round1")
    assert s.phase == "SEL" and s.round == 1


# ---------------------------------------------------------------- phase shapes (§4)

EXPECT_SHAPE = {
    "setup_sb_midway": ("SB", 0, 1),
    "setup_l1_mid": ("L1", 0, 2),
    "setup_l2_mid": ("L2", 0, 2),
    "sel_round1": ("SEL", 1, 0),
    "sel_pass1_mid": ("SEL", 1, 2),
    "sel_pass2_dropped": ("SEL", 1, 0),
    "resolve_ln": ("LN", 1, 1),
    "resolve_px": ("PX", 1, 2),
    "resolve_bd": ("BD", 1, 1),
    "resolve_ck": ("CK", 1, 3),
    "resolve_vp": ("VP", 1, 1),
    "resolve_vd": ("VD", 1, 1),
}


@pytest.mark.parametrize("name", sorted(EXPECT_SHAPE))
def test_SCN4_phase_round_and_default_current_player(name: str) -> None:
    """SCENARIOS §4: round 0 for SB/L1/L2, >= 1 otherwise; current_player derived from done /
    pass / marker owner (L1 forward from start player, L2 reverse, SB two per seat)."""
    phase, rnd, cp = EXPECT_SHAPE[name]
    s = H.load_fixture(name)
    assert (s.phase, s.round, s.current_player()) == (phase, rnd, cp)
    assert not s.is_terminal()


@pytest.mark.parametrize("name", ["end_sites_full", "end_both", "end_last_stone"])
def test_SCN47_end_fixtures_are_terminal(name: str) -> None:
    """SCENARIOS §4.7: an END scenario is terminal."""
    s = H.load_fixture(name)
    assert s.phase == "END" and s.is_terminal()


def test_SCN4_end_last_stone_has_zero_stones_on_clock() -> None:
    """SCENARIOS §6 S-STONES: stones_on_clock = 0 is allowed in END."""
    s = H.load_fixture("end_last_stone")
    assert s.stones_on_clock == 0
    assert sum(s.seat(p).stones for p in range(4)) == 5


def test_SCN4_setup_phases_derive_seat_from_start_player() -> None:
    """SCENARIOS §4.3: setup current_player derives from turn order starting at start_player."""
    doc = _plain(H.load_doc("setup_l1_mid"))
    first = [seat.get("line", [])[:1] for seat in doc["seats"]]
    doc["start_player"] = 1
    # order 1, 2, 3, 0: seats 1 and 2 are served
    doc["seats"] = [{}, {"line": first[0]}, {"line": first[1]}, {}]
    doc["pending"] = {"done": 2}
    assert _load(doc).current_player() == 3


def test_SCN4_sel_board_is_kept_as_given() -> None:
    """SCENARIOS §3.4/§4.1: pass-2 selection with seat 3 dropped keeps the board as given."""
    s = H.load_fixture("sel_pass2_dropped")
    assert s.current_player() == 0
    assert list(s.action_board[0]) == [0, 1, 2, 3]
    assert list(s.action_board[2]) == [0, 1, 2, 3]


def test_SCN4_resolution_board_is_kept() -> None:
    """SCENARIOS §4.2: during resolution the board still shows the round's markers."""
    s = H.load_fixture("resolve_ln")
    assert list(s.action_board[0]) == [0, 1]


def test_SCN4_buildings_appear_at_their_sites() -> None:
    """SCENARIOS §3.2: listed buildings appear at their sites with their type, others stay empty."""
    doc = _plain(H.load_doc("sel_round1"))
    s = _load(doc)
    listed = {b["site"]: b["type"] for b in doc["buildings"]}
    for t in range(len(B.sites)):
        assert s.site_building(t) == listed.get(t)


# ---------------------------------------------------------------- dump / round trip


@pytest.mark.parametrize("name", ROUND_TRIP)
def test_SCN1_load_dump_load_is_identity(name: str) -> None:
    """SCENARIOS §1/§7: dump_scenario emits a resolved scenario; load(dump(s)) is the same state
    and dumps to the same document (JSON round trip included)."""
    s1 = _load(H.load_doc(name))
    d1 = _dump(s1)
    s2 = _load(json.loads(json.dumps(d1)))
    assert str(s2) == str(s1)
    assert (s2.phase, s2.round) == (s1.phase, s1.round)
    if not s1.is_terminal():
        assert s2.current_player() == s1.current_player()
    if s1.phase == "SB":
        assert s2.legal_actions() == s1.legal_actions()
    assert _dump(s2) == d1


REQUIRED_DUMP_KEYS = {
    "format",
    "parameters",
    "round",
    "phase",
    "current_player",
    "start_player",
    "clock",
    "stones_on_clock",
    "seats",
    "buildings",
    "waiting",
    "supply",
    "action_board",
    "pending",
    "move_offset",
}


@pytest.mark.parametrize("name", ["fresh_4p", "sel_pass1_mid", "resolve_vd", "end_last_stone"])
def test_SCN1_dump_is_fully_resolved(name: str) -> None:
    """SCENARIOS §1: dump emits all fields explicitly (every default filled in).
    ``current_player`` may be absent only in END."""
    d = _dump(H.load_fixture(name))
    need = REQUIRED_DUMP_KEYS - ({"current_player"} if d["phase"] == "END" else set())
    assert need <= set(d)
    assert d["format"] == 1
    assert d["parameters"]["players"] == len(d["seats"]) == 4
    for seat in d["seats"]:
        assert {"points", "stones", "buses", "markers_left", "line", "last_score_move"} <= set(seat)
    assert set(d["action_board"]) == {
        "lines",
        "buses",
        "passengers",
        "buildings",
        "clock",
        "vrrooomm",
        "starting_player",
    }
    assert d.get("history", []) == []
    assert "expect" not in d
    json.dumps(d)


def test_SCN1_dump_of_fresh_state_has_derived_values() -> None:
    """SCENARIOS §7: the fresh dump lists the four roundabout passengers, supply 11, done 0."""
    d = _dump(_initial(4))
    assert d["round"] == 0 and d["phase"] == "SB"
    assert d["waiting"] == {str(n): 1 for n in B.roundabouts}
    assert d["supply"] == 11
    assert d["pending"] == {"done": 0}
    assert d["buildings"] == []
    assert d["stones_on_clock"] == 5 and d["clock"] == "house"


def test_SCN1_dump_after_engine_play_loads_back() -> None:
    """SCENARIOS §1: any state reached by play dumps and reloads to the same state (SB playout)."""
    s = _initial(4)
    for i in range(5):
        s.apply_action(encode_build(TIER_A[i], i % 3))
    r = _load(json.loads(json.dumps(_dump(s))))
    assert str(r) == str(s)
    assert r.current_player() == s.current_player()
    assert r.legal_actions() == s.legal_actions()


def test_SCN1_dump_of_history_state_equals_dump_of_position() -> None:
    """SCENARIOS §5: loading with history and loading the resulting position (no history) agree."""
    ids = [encode_build(TIER_A[i], 0) for i in range(2)]
    s = _load({"format": 1, "history": ids})
    assert str(_load(_dump(s))) == str(s)


# ---------------------------------------------------------------- files, YAML


def test_SCN0_load_from_path_object() -> None:
    """SCENARIOS §0: load_scenario accepts a Path to a JSON file."""
    assert str(_load(H.fixture_path("sel_round1"))) == str(H.load_fixture("sel_round1"))


def test_SCN0_load_from_str_path() -> None:
    """load_scenario accepts a str path."""
    got = _load(str(H.fixture_path("sel_pass1_mid")))
    assert str(got) == str(H.load_fixture("sel_pass1_mid"))


def test_SCN0_yaml_accepted(tmp_path: Path) -> None:
    """SCENARIOS intro: the loader also accepts YAML with the same schema."""
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(H.load_doc("sel_pass1_mid")))
    want = str(H.load_fixture("sel_pass1_mid"))
    assert str(_load(p)) == want
    assert str(_load(str(p))) == want


def test_SCN0_yaml_errors_use_scenario_error(tmp_path: Path) -> None:
    """A YAML file with an unknown key fails like JSON (S-SCHEMA)."""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"format": 1, "bogus": 1}))
    with pytest.raises(H.scenario_module().ScenarioError) as e:
        _load(p)
    assert e.value.code == "S-SCHEMA"


def test_SCN0_load_does_not_mutate_the_input_document() -> None:
    """The loader must not edit the caller's dict (fixtures are reused across tests)."""
    doc = H.load_doc("sel_pass1_mid")
    before = json.dumps(doc, sort_keys=True)
    _load(doc)
    assert json.dumps(doc, sort_keys=True) == before


def test_SCN0_loaded_states_are_independent() -> None:
    """Two loads of the same document give independent states (no shared mutable defaults)."""
    a = H.load_fixture("setup_sb_midway")
    b = H.load_fixture("setup_sb_midway")
    before = str(b)
    a.apply_action(a.legal_actions()[0])
    assert str(b) == before
    assert str(a) != before


def test_SCN0_scenario_error_carries_code_and_message() -> None:
    """API: ScenarioError(code, message) exposes ``.code``; it is an Exception."""
    E = H.scenario_module().ScenarioError
    e = E("S-ID", "bad id")
    assert isinstance(e, Exception)
    assert e.code == "S-ID"
    assert "bad id" in str(e)


# ---------------------------------------------------------------- validation (§6)


def _case_id(c: dict[str, Any]) -> str:
    return c["name"]


@pytest.mark.parametrize("case", CASES["cases"], ids=_case_id)
def test_SCN6_invalid_scenario_fails_with_code(case: dict[str, Any]) -> None:
    """SCENARIOS §6: each invalid document fails with exactly the listed code."""
    with pytest.raises(H.scenario_module().ScenarioError) as e:
        _load(case["doc"])
    assert e.value.code == case["code"]


@pytest.mark.parametrize("case", CASES["ambiguous"], ids=_case_id)
def test_SCN6_ambiguous_invalid_scenario_fails_with_a_listed_code(case: dict[str, Any]) -> None:
    """SCENARIOS §6: the document is invalid, but the spec names no single code (see ``note``);
    any code in the list is accepted. Reported as a spec ambiguity."""
    with pytest.raises(H.scenario_module().ScenarioError) as e:
        _load(case["doc"])
    assert e.value.code in case["codes"], case["note"]


@pytest.mark.parametrize("case", CASES["controls"], ids=_case_id)
def test_SCN6_boundary_documents_load(case: dict[str, Any]) -> None:
    """SCENARIOS §6: documents at the edge of each check (but valid) load without error and dump
    back to a loadable document."""
    s = _load(case["doc"])
    assert str(_load(json.loads(json.dumps(_dump(s))))) == str(s)


def test_SCN6_every_s_code_has_a_failing_case() -> None:
    """SCENARIOS §6: the case table covers every code in the section 6 table."""
    codes = {c["code"] for c in CASES["cases"]}
    table = {
        "S-SCHEMA",
        "S-ID",
        "S-PLAYERS",
        "S-LINE",
        "S-BUS",
        "S-MARKERS",
        "S-STONES",
        "S-SITES",
        "S-PASSENGERS",
        "S-BOARD",
        "S-PHASE",
        "S-SEL",
        "S-POINTS",
        "S-SETUP",
        "S-END",
        "S-HISTORY",
    }
    assert table <= codes


def test_SCN6_malformed_input_raises_only_scenario_error() -> None:
    """Malformed input must raise ScenarioError, never a bare KeyError/TypeError."""
    E = H.scenario_module().ScenarioError
    for bad in (
        {"format": 1, "seats": 3},
        {"format": 1, "buildings": {"a": 1}},
        {"format": 1, "waiting": [1, 2]},
        {"format": 1, "parameters": 7},
        {"format": 1, "seats": [None, None, None, None]},
        [],
    ):
        with pytest.raises(E):
            _load(bad)


def test_SCN6_strict_flag_rejects_tier_skip_but_default_accepts() -> None:
    """SCENARIOS §6 (last paragraph): tier-fill order is not checked by default (the unreachable
    fixture loads); ``"strict": true`` makes it an error for editor-built games."""
    doc = H.load_doc("unreachable_tier_skip")
    _load(doc)
    with pytest.raises(H.scenario_module().ScenarioError):
        _load({**doc, "strict": True})
