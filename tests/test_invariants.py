"""Invariant checker (PLAN §6 item 3; .claude/rules/engine.md).

Expected API: ``tests/support/invariants.py``
  check_invariants(state, prev=None) -> list[str]
``prev`` is the state one action earlier (or None). An empty list means no violation; each
violation is one non-empty string. The list covers every PLAN §6.3 invariant checkable on one
state or a (prev, state) pair:
  passengers on board + in buildings + supply = 15; at most one passenger per building;
  every line a valid trail (contiguous, no repeated street); line markers + supply = 25 per seat;
  buses 1..5; Action markers only decrease; points never decrease; sites filled tier by tier;
  stones on clock + taken = 5 (4 at 3p); non-terminal states have a legal action; win_loss
  returns sum to 0 when terminal.

Public-API limits (reported to the lead): the loader enforces passenger conservation, bus range,
stone totals, line validity and marker counts, so those can only be broken through the engine
itself. Here they are covered by (a) fresh and played states passing, and (b) duck-typed proxies
that override one accessor of a real state. The tier-order, monotonic-points, and marker-recovery
invariants are broken through ``load_scenario`` pairs.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pyspiel
import pytest

from bus.actions import encode_build
from bus.board import load_board
from tests import harness as H

B = load_board()


def _check(state: Any, prev: Any = None) -> list[str]:
    from tests.support.invariants import check_invariants

    return check_invariants(state, prev)


def _initial(n: int = 4, **kw: Any) -> Any:
    import bus.game  # noqa: F401

    return pyspiel.load_game("python_bus", {"players": n, **kw}).new_initial_state()


def _doc(name: str = "sel_round1", **mods: Any) -> dict[str, Any]:
    d = H.load_doc(name)
    d.pop("expect", None)
    d.update(mods)
    return d


class Proxy:
    """Read-through wrapper over a real state with selected attributes overridden."""

    def __init__(self, inner: Any, **override: Any) -> None:
        self._inner = inner
        self._override = override

    def __getattr__(self, name: str) -> Any:
        if name in self._override:
            v = self._override[name]
            return v if callable(v) else (lambda v=v: v)
        return getattr(self._inner, name)


# ------------------------------------------------------------------ passing states


@pytest.mark.parametrize("n", [3, 4, 5])
def test_INV_fresh_states_have_no_violations(n: int) -> None:
    """PLAN §6.3: a fresh game satisfies every invariant."""
    assert _check(_initial(n)) == []


@pytest.mark.parametrize("n,seed", [(3, 1), (4, 2), (5, 3), (4, 4)])
def test_INV_random_setup_buildings_playout_stays_clean(n: int, seed: int) -> None:
    """PLAN §6.3: along a random SB playout every state, and every (prev, state) pair, is clean."""
    rng = random.Random(seed)
    s = _initial(n)
    assert _check(s) == []
    steps = 0
    while s.phase == "SB" and not s.is_terminal():
        prev = s.clone()
        s.apply_action(rng.choice(s.legal_actions()))
        steps += 1
        assert _check(s, prev) == [], f"after step {steps}"
    assert steps == 2 * n
    assert _check(s, None) == []


@pytest.mark.parametrize("name", H.fixture_names())
def test_INV_fixtures_other_than_unreachable_are_clean(name: str) -> None:
    """Fixtures that are reachable positions satisfy the single-state invariants, except the
    deliberately unreachable one (tier skip). Non-terminal fixtures whose phase has no legal
    actions yet are skipped (their legal-action invariant is covered by later units)."""
    s = H.load_fixture(name)
    out = _check(s)
    if name == "unreachable_tier_skip":
        assert out
        return
    if not s.is_terminal() and s.phase not in ("SB",):
        try:
            has_actions = bool(s.legal_actions())
        except Exception:  # phase not implemented yet
            has_actions = False
        if not has_actions:
            pytest.skip("legal actions for this phase arrive in a later unit")
    assert out == []


# ------------------------------------------------------------------ broken single states


def test_INV_site_tiers_must_fill_in_order() -> None:
    """PLAN §6.3 'sites filled tier by tier' (R-010): a building on tier B while tier A has an
    empty site is a violation; the loader accepts it (SCENARIOS §6 'Not checked')."""
    s = H.load_fixture("unreachable_tier_skip")
    out = _check(s)
    assert out
    assert all(isinstance(x, str) and x for x in out)


def test_INV_non_terminal_state_without_legal_action_is_flagged() -> None:
    """PLAN §6.3: a non-terminal state must have a legal action (proxy: legal_actions -> [])."""
    p = Proxy(_initial(4), legal_actions=[])
    assert _check(p)


def test_INV_win_loss_returns_must_sum_to_zero() -> None:
    """PLAN §6.3: win_loss returns sum to 0 at a terminal state."""
    p = Proxy(H.load_fixture("end_sites_full"), returns=[1.0, 1.0, 0.0, 0.0])
    assert _check(p)


def test_INV_balanced_win_loss_returns_are_accepted() -> None:
    """PLAN §6.3: terminal states with returns summing to 0 produce no violation from that rule
    (proxy values; other invariants hold on the END fixture)."""
    p = Proxy(
        H.load_fixture("end_sites_full"),
        returns=[1.0, -1.0 / 3, -1.0 / 3, -1.0 / 3],
        legal_actions=[],
    )
    assert _check(p) == []


def test_INV_returns_rule_is_for_win_loss_only() -> None:
    """PLAN §6.3: only ``win_loss`` returns must sum to 0; margin returns are not constrained here.
    Assumption (flagged): the checker reads the mode from the game parameters."""
    doc = _doc("end_sites_full", parameters={"players": 4, "returns": "margin"})
    p = Proxy(H.load_doc_state(doc), returns=[3.0, 1.0, -1.0, -1.0 + 5.0])
    assert _check(p) == []


# ------------------------------------------------------------------ (prev, state) pairs


def _scored(points: int, last: int | None) -> Any:
    d = _doc()
    d["seats"][0].update(points=points, last_score_move=last)
    return H.load_doc_state(d)


def test_INV_points_must_not_decrease() -> None:
    """PLAN §6.3 'points never decrease': prev seat 0 has 3, state has 2 -> violation."""
    assert _check(_scored(2, 5), _scored(3, 5))


def test_INV_points_may_stay_or_increase() -> None:
    """PLAN §6.3: equal or higher points are fine."""
    assert _check(_scored(3, 5), _scored(3, 5)) == []
    assert _check(_scored(4, 6), _scored(3, 5)) == []


def _with_markers(left0: int) -> Any:
    d = _doc()
    d["seats"][0]["markers_left"] = left0
    return H.load_doc_state(d)


def test_INV_action_markers_are_never_recovered() -> None:
    """PLAN §6.3 'markers only decrease': 18 -> 19 is a violation."""
    assert _check(_with_markers(19), _with_markers(18))


def test_INV_action_markers_may_decrease() -> None:
    """PLAN §6.3: 19 -> 18 and unchanged are fine."""
    assert _check(_with_markers(18), _with_markers(19)) == []
    assert _check(_with_markers(19), _with_markers(19)) == []


def test_INV_line_must_not_shrink() -> None:
    """PLAN §6.3 'markers only decrease' read as Bus Line markers: a line shorter than in prev
    (line markers left increased) is a violation. Interpretation flagged in the report."""
    longer = _doc()
    shorter = _doc()
    shorter["seats"][0]["line"] = shorter["seats"][0]["line"][:1]
    assert _check(H.load_doc_state(shorter), H.load_doc_state(longer))


def test_INV_pair_check_accepts_a_real_transition() -> None:
    """PLAN §6.3: a real engine step produces no pair violation (also covered by the playout)."""
    prev = _initial(4)
    s = prev.clone()
    s.apply_action(encode_build(next(t for t, x in enumerate(B.sites) if x.tier == "A"), 0))
    assert _check(s, prev) == []


def test_INV_result_is_a_list_of_strings() -> None:
    """API: check_invariants returns list[str], also when called with positional prev=None."""
    out = _check(_initial(4), None)
    assert isinstance(out, list)
    json.dumps(out)
