"""Batch action endpoint: all-or-nothing, history gets the plain frozen ids."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.support.toy_adapter import toy_app

Json = dict[str, Any]
TARGET_0 = 3  # toy game: actions 3..5 choose the target slot after a colour pick (0..2)


@pytest.fixture
def two_step() -> tuple[TestClient, Json, list[int]]:
    """Two human seats; seat 0's turn is two decisions (colour pick, then target)."""
    c = TestClient(toy_app())
    g: Json = c.post(
        "/api/games",
        json={"game": "toy", "params": {"num_players": 2}, "seats": ["human", "human"], "seed": 1},
    ).json()
    assert g["current_player"] == 0
    return c, g, [g["legal_actions"][0]["id"], TARGET_0]


def test_batch_equals_single_actions(two_step: tuple[TestClient, Json, list[int]]) -> None:
    c, g, ids = two_step
    n0 = g["history_length"]
    r = c.post(f"/api/games/{g['id']}/actions/batch?seat=0", json={"action_ids": ids})
    assert r.status_code == 200, r.text
    batched = r.json()
    assert [h["id"] for h in batched["history"][n0 : n0 + len(ids)]] == ids

    exp = c.get(f"/api/games/{g['id']}/export").json()
    assert exp["history"][n0 : n0 + len(ids)] == ids

    # Same result as applying the actions one by one.
    c2 = TestClient(toy_app())
    g2 = c2.post(
        "/api/games",
        json={"game": "toy", "params": {"num_players": 2}, "seats": ["human", "human"], "seed": 1},
    ).json()
    for a in ids:
        g2 = c2.post(f"/api/games/{g2['id']}/actions?seat=0", json={"action": a}).json()
    assert g2["view"] == batched["view"]


def test_single_back_undoes_batch(two_step: tuple[TestClient, Json, list[int]]) -> None:
    c, g, ids = two_step
    c.post(f"/api/games/{g['id']}/actions/batch?seat=0", json={"action_ids": ids})
    r = c.post(f"/api/games/{g['id']}/back?seat=0")
    assert r.status_code == 200, r.text
    assert r.json()["history_length"] == g["history_length"]
    assert r.json()["view"]["state"] == g["view"]["state"]


def test_illegal_middle_step_is_atomic(two_step: tuple[TestClient, Json, list[int]]) -> None:
    c, g, ids = two_step
    bad = [ids[0], ids[1], 999, ids[1]]
    r = c.post(f"/api/games/{g['id']}/actions/batch", json={"action_ids": bad})
    assert r.status_code == 409
    after = c.get(f"/api/games/{g['id']}?seat=0").json()
    assert after["history_length"] == g["history_length"]
    assert after["view"]["state"] == g["view"]["state"]


def test_wrong_seat_is_409(two_step: tuple[TestClient, Json, list[int]]) -> None:
    c, g, ids = two_step
    r = c.post(f"/api/games/{g['id']}/actions/batch?seat=1", json={"action_ids": ids})
    assert r.status_code == 409
    assert c.get(f"/api/games/{g['id']}").json()["history_length"] == g["history_length"]


def test_batch_after_back_works(two_step: tuple[TestClient, Json, list[int]]) -> None:
    c, g, ids = two_step
    c.post(f"/api/games/{g['id']}/actions/batch?seat=0", json={"action_ids": ids})
    c.post(f"/api/games/{g['id']}/back?seat=0")
    r = c.post(f"/api/games/{g['id']}/actions/batch?seat=0", json={"action_ids": ids})
    assert r.status_code == 200, r.text


def test_batch_limits() -> None:
    c = TestClient(toy_app())
    g = c.post("/api/games", json={"game": "toy", "seats": ["human", "random"], "seed": 1}).json()
    assert c.post(f"/api/games/{g['id']}/actions/batch", json={"action_ids": []}).status_code == 422
    r = c.post(f"/api/games/{g['id']}/actions/batch", json={"action_ids": [0] * 33})
    assert r.status_code == 422
