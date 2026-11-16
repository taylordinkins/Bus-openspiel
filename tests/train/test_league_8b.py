# pyright: basic, reportMissingImports=false
"""Phase 4 unit 8b: hall of fame, PFSP, value trunk format 2, ratings, schedules, per-count eval."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bots.net_forward import EXPORT_FORMAT, NetWeights
from train.features import OBS_SIZE
from train.league import (
    CURRENT,
    League,
    LeagueSpec,
    game_type,
    pfsp_weight,
    sample_assignment,
    thin_hof,
)
from train.ratings import bradley_terry, elo_ratings, pair_wins
from train.rollout import RolloutConfig, play_games

ROOT = Path(__file__).resolve().parents[2]


def read(p: Path) -> list[dict[str, str]]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _weights(seed: int = 0, hidden: tuple[int, ...] = (16,), separate: bool = True) -> NetWeights:
    torch = pytest.importorskip("torch")
    from train.model import BusNet
    from train.ppo import net_to_weights

    torch.manual_seed(seed)
    return net_to_weights(BusNet(hidden=hidden, separate_value=separate))


# ---------------------------------------------------------------- PFSP and league sampling


def test_pfsp_weight_math() -> None:
    assert pfsp_weight(0.0) == 1.0
    assert pfsp_weight(0.5) == pytest.approx(0.25)
    assert pfsp_weight(0.9) == pytest.approx(0.05)  # (0.1)^2 = 0.01 -> floor
    assert pfsp_weight(1.0) == 0.05 and pfsp_weight(1.0, floor=0.2) == 0.2


def test_snapshot_share_split_recent_vs_hof() -> None:
    spec = LeagueSpec((0.0, 1.0, 0.0), (1, 2), "uniform", ("a", "b"), 0.5)
    rng = np.random.default_rng(0)
    c: Counter[str] = Counter()
    for _ in range(3000):
        c.update(x for x in sample_assignment(spec, 4, rng) if x != CURRENT)
    recent = c["snap:1"] + c["snap:2"]
    hof = c["hof:a"] + c["hof:b"]
    assert recent / (recent + hof) == pytest.approx(0.5, abs=0.04)
    # hof only, recent only
    only_hof = LeagueSpec((0.0, 1.0, 0.0), (), "uniform", ("a",), 0.5)
    assert {x for x in sample_assignment(only_hof, 4, rng) if x != CURRENT} <= {"hof:a"}
    assert game_type(["current", "hof:a", "snap:1"]) == "snap"


def test_pfsp_prefers_hard_opponents_in_sampling() -> None:
    # snap:1 beats the learner (low learner win rate -> weight 1), snap:2 is easy (floor)
    lg = League((0.0, 1.0, 0.0), 5)
    w = _weights()
    lg.add(1, w)
    lg.add(2, w)
    lg.update_results({"snap:1": [10, 40, 4], "snap:2": [10, 40, 38]})
    spec = lg.spec()
    assert spec.snap_w == pytest.approx((pfsp_weight(0.1), 0.05))
    rng = np.random.default_rng(1)
    c: Counter[str] = Counter()
    for _ in range(2000):
        c.update(x for x in lg.sample_assignment(4, rng) if x != CURRENT)
    assert c["snap:1"] > 8 * c["snap:2"] > 0
    off = League((0.0, 1.0, 0.0), 5, pfsp=False)
    off.add(1, w)
    assert off.spec().snap_w == ()


def test_bot_split_heuristic_15_random_3() -> None:
    spec = LeagueSpec((0.79, 0.0, 0.21), (), "uniform", bot_split=5 / 6)
    rng = np.random.default_rng(3)
    c: Counter[str] = Counter()
    n = 8000
    for i in range(n):
        c[game_type(sample_assignment(spec, 3 + i % 3, rng))] += 1
    assert c["heuristic"] / n == pytest.approx(0.175, abs=0.02)
    assert c["random"] / n == pytest.approx(0.035, abs=0.012)


def test_win_ema_update() -> None:
    lg = League(ema_alpha=0.5)
    lg.update_results({"snap:3": [1, 4, 2], "heuristic": [1, 4, 4]})
    assert lg.win_ema == {"snap:3": 0.5}  # first observation taken as is; bots not tracked
    lg.update_results({"snap:3": [1, 2, 2]})
    assert lg.win_ema["snap:3"] == pytest.approx(0.75)


# ---------------------------------------------------------------- hall of fame


def _entries(updates: list[int], permanent: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    out = [{"name": p, "update": 0, "permanent": True} for p in permanent]
    out += [{"name": f"m{u}", "update": u, "permanent": False} for u in updates]
    return out


def test_thin_hof_keeps_permanent_first_last_and_cap() -> None:
    ents = _entries(list(range(100, 3100, 100)), ("ppo-smoke",))  # 30 milestones + 1 permanent
    gone = thin_hof(ents, 20)
    kept = [e for e in ents if e["name"] not in gone]
    assert len(kept) == 20 and len(ents) == 31  # input untouched
    names = {e["name"] for e in kept}
    assert {"ppo-smoke", "m100", "m3000"} <= names
    # survivors stay spread: no gap bigger than ~3x the uniform spacing
    ups = sorted(e["update"] for e in kept if not e["permanent"])
    assert max(b - a for a, b in zip(ups, ups[1:], strict=False)) <= 400
    assert thin_hof(_entries([100, 200]), 1) == []  # nothing removable: first/last/permanent only
    assert thin_hof(_entries([100, 200, 300]), 5) == []  # under the cap


def test_hof_never_evicted_below_cap_and_files(tmp_path: Path) -> None:
    lg = League(directory=tmp_path / "pool", hof_cap=4)
    w = _weights()
    lg.add_hof("ppo-smoke", w, 0, permanent=True)
    assert lg.add_hof("m100", w, 100) == [] and lg.add_hof("m200", w, 200) == []
    assert lg.add_hof("m300", w, 300) == []  # exactly at the cap
    assert lg.add_hof("m400", w, 400) == ["m200"]  # over: tightest interior milestone goes
    assert lg.hof_names == ["ppo-smoke", "m100", "m300", "m400"]
    files = sorted(p.name for p in (tmp_path / "pool").glob("hof_*.npz"))
    assert files == sorted(f"hof_{n}.npz" for n in lg.hof_names)
    assert set(lg.opponents()) == {f"hof:{n}" for n in lg.hof_names}
    with pytest.raises(ValueError):
        lg.add_hof("m100", w, 100)


def test_league_state_roundtrip(tmp_path: Path) -> None:
    w = _weights()
    a = League(directory=tmp_path / "a", pool_size=3, hof_cap=5)
    a.add_hof("ppo-smoke", w, 0, permanent=True)
    for u in (10, 20, 30, 40):
        a.add(u, w)
    a.add_hof("m20", w, 20)
    a.add_hof("m40", w, 40)
    a.update_results({"snap:30": [1, 3, 1], "hof:m20": [1, 3, 3]})
    st = json.loads(json.dumps(a.state()))  # survives JSON like pool.json does
    b = League(directory=tmp_path / "a", pool_size=3, hof_cap=5)
    b.restore_state(st, max_id=30)  # snap 40 and hof m40 were taken after the checkpoint
    assert b.ids == [20, 30] and b.hof_names == ["ppo-smoke", "m20"]
    assert b.win_ema == a.win_ema
    assert not (tmp_path / "a" / "hof_m40.npz").exists()
    assert (tmp_path / "a" / "hof_ppo-smoke.npz").exists()  # permanent: any max_id
    assert b.spec().hof_w == a.spec().hof_w[:2]
    pool = json.loads((tmp_path / "a" / "pool.json").read_text())
    assert pool["hof"][0]["permanent"] is True and "win_ema" in pool


def test_rollout_against_hof_label_and_by_label() -> None:
    w, hof = _weights(0), _weights(1)
    spec = LeagueSpec((0.0, 1.0, 0.0), (), "uniform", ("x",), 1.0)
    cfg = RolloutConfig(players=(4,), envs_per_worker=3, league=spec, lam=1.0)
    res = play_games(w, 8, 5, cfg, opponents={"hof:x": hof})
    st = res["stats"]
    assert set(st["by_label"]) == {"hof:x"} and st["by_label"]["hof:x"][0] == st["games"] == 8
    assert st["illegal"] == 0
    s = res["samples"]
    assert s is not None and ((s["prog"] >= 0) & (s["prog"] < 1)).all()


# ---------------------------------------------------------------- net format 2


@pytest.mark.parametrize("separate", [True, False])
@pytest.mark.parametrize("batch", [1, 7])
def test_torch_numpy_parity_value_trunk(tmp_path: Path, separate: bool, batch: int) -> None:
    torch = pytest.importorskip("torch")
    from train.export import export_npz
    from train.features import MAX_SEATS
    from train.model import BusNet

    torch.manual_seed(1)
    net = BusNet(hidden=(32, 16), separate_value=separate)
    torch.nn.init.normal_(net.pi.weight, std=0.3)
    torch.nn.init.normal_(net.v.weight, std=0.5)
    export_npz(net, tmp_path / "n.npz", {"k": 1})
    w = NetWeights.load(tmp_path / "n.npz")
    assert w.meta["format_version"] == EXPORT_FORMAT == 2
    assert len(w.v_layers) == (2 if separate else 0) == w.meta["n_v_layers"]
    rng = np.random.default_rng(batch)
    obs = rng.random((batch, OBS_SIZE), dtype=np.float32)
    mask = rng.random((batch, w.n_actions)) < 0.2
    mask[:, 0] = True
    with torch.no_grad():
        tl, tv = net(torch.from_numpy(obs), torch.from_numpy(mask))
    nl, nv = w.forward(obs, mask)
    assert nv.shape == (batch, MAX_SEATS)
    assert np.allclose(tl.numpy()[mask], nl[mask], atol=1e-5)
    assert np.allclose(tv.numpy(), nv, atol=1e-5)
    assert np.array_equal(w.logits(obs, mask), nl)
    if separate:  # the value head really does not read the policy trunk
        with torch.no_grad():
            for prm in net.trunk.parameters():
                prm.add_(1.0)
        _, tv2 = net.forward(torch.from_numpy(obs), torch.from_numpy(mask))
        assert torch.allclose(tv, tv2)


def test_old_format_ppo_smoke_still_loads_and_plays() -> None:
    w = NetWeights.load(ROOT / "models" / "ppo-smoke.npz")
    assert w.meta.get("format_version", 1) == 1 and w.v_layers == ()
    rng = np.random.default_rng(0)
    obs = rng.random((2, OBS_SIZE), dtype=np.float32)
    mask = np.ones((2, w.n_actions), bool)
    logits, vals = w.forward(obs, mask)
    assert np.isfinite(logits).all() and vals.shape[0] == 2
    # as a fixed opponent in a mixed-seat rollout, next to a new-format learner
    spec = LeagueSpec((0.0, 1.0, 0.0), (), "uniform", ("ppo-smoke",), 1.0)
    cfg = RolloutConfig(players=(4,), envs_per_worker=2, league=spec)
    res = play_games(_weights(), 2, 1, cfg, opponents={"hof:ppo-smoke": w})
    assert res["stats"]["illegal"] == 0 and res["stats"]["games"] == 2


def test_net_forward_still_torch_free_with_league_imports() -> None:
    import subprocess
    import sys

    code = (
        "import sys, bots.net_forward, bots.net; "
        "sys.exit(1 if any(k == 'torch' or k.startswith('torch.') for k in sys.modules) else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- ratings


def test_bradley_terry_recovers_strengths() -> None:
    rng = np.random.default_rng(0)
    true = {"a": 400.0, "b": 0.0, "c": -200.0}  # Elo
    names = list(true)
    games = []
    for _ in range(3000):
        seats = list(rng.choice(names, size=3, replace=False))
        # score = Gumbel noise + strength: Plackett-Luce-like ordering
        sc = [true[x] * math.log(10) / 400 + rng.gumbel() for x in seats]
        games.append((seats, sc))
    res = elo_ratings(games, names, anchor="b", anchor_rating=1000.0)
    assert res["b"]["rating"] == pytest.approx(1000.0)
    assert res["a"]["rating"] > res["b"]["rating"] > res["c"]["rating"]
    assert res["a"]["rating"] - res["b"]["rating"] == pytest.approx(400, abs=60)
    assert res["b"]["rating"] - res["c"]["rating"] == pytest.approx(200, abs=60)


def test_pair_wins_ties_and_same_label() -> None:
    w, app = pair_wins([(["x", "y", "x"], [3.0, 3.0, 1.0])], ["x", "y"])
    assert app.tolist() == [2.0, 1.0]
    # x(0) vs y: tie -> .5/.5 ; y vs x(2): y wins ; x vs x skipped
    assert w[0, 1] == 0.5 and w[1, 0] == 1.5
    p = bradley_terry(np.array([[0.0, 9.0], [1.0, 0.0]]))
    assert p[0] > p[1] and math.isclose(math.prod(p), 1.0, rel_tol=1e-6)


def test_bradley_terry_finite_without_losses() -> None:
    p = bradley_terry(np.array([[0.0, 5.0, 5.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.isfinite(p).all() and p[0] > p[1] > p[2]


# ---------------------------------------------------------------- schedules and diagnostics


def test_cosine_lr_and_entropy_floor() -> None:
    from train.ppo import cosine, ent_at, lr_at

    assert cosine(1.0, 0.1, 0, 100) == pytest.approx(1.0)
    assert cosine(1.0, 0.1, 50, 100) == pytest.approx(0.55)
    assert cosine(1.0, 0.1, 100, 100) == pytest.approx(0.1)
    assert cosine(1.0, 0.1, 500, 100) == pytest.approx(0.1)
    assert lr_at(5, 3e-4, False, "cosine", 100) == 3e-4
    assert lr_at(100, 3e-4, True, "cosine", 100, 0.1) == pytest.approx(3e-5)
    assert lr_at(50, 1.0, True, "linear", 100, 0.0) == pytest.approx(0.5)
    assert ent_at(0, 0.01, 0.001, 100, 0.5) == pytest.approx(0.01)
    assert ent_at(25, 0.01, 0.001, 100, 0.5) == pytest.approx(0.0055)
    assert ent_at(50, 0.01, 0.001, 100, 0.5) == pytest.approx(0.001)
    assert ent_at(99, 0.01, 0.001, 100, 0.5) == pytest.approx(0.001)  # constant floor
    assert ent_at(99, 0.01, None, 100, 0.5) == 0.01


def test_ev_by_quartile() -> None:
    from train.ppo import ev_by_quartile

    rng = np.random.default_rng(0)
    n = 4000
    prog = rng.random(n).astype(np.float32)
    target = rng.normal(size=n)
    pred = np.where(prog < 0.5, target, 0.0)  # perfect early, useless late
    ev = ev_by_quartile(pred, target, prog)
    assert ev[0] == pytest.approx(1.0) and ev[1] == pytest.approx(1.0)
    assert ev[2] < 0.05 and ev[3] < 0.05
    assert math.isnan(ev_by_quartile(pred[:0], target[:0], prog[:0])[0])


def test_player_weights_resolution_and_sampling() -> None:
    from tools.train_ppo import resolve_player_weights

    assert resolve_player_weights((3, 4, 5), None) == (0.3, 0.4, 0.3)
    assert resolve_player_weights((4,), None) is None
    assert resolve_player_weights((3, 4), "1,3") == (1.0, 3.0)
    with pytest.raises(ValueError):
        resolve_player_weights((3, 4), "1")
    cfg = RolloutConfig(players=(3, 4, 5), envs_per_worker=8, player_weights=(0.3, 0.4, 0.3))
    res = play_games(_weights(), 120, 9, cfg)
    sizes = Counter(n for _, n in res["stats"]["ego_returns"])
    games = {n: c // n for n, c in sizes.items()}
    assert sum(games.values()) == 120
    assert games[4] > games[3] and games[4] > games[5]
    assert 0.25 < games[3] / 120 < 0.45


# ---------------------------------------------------------------- training runs


def run_args(tmp: Path, name: str, *extra: str) -> list[str]:
    return [
        "--name", name, "--out-dir", str(tmp), "--games-per-update", "4", "--workers", "0",
        "--hidden", "16,16", "--device", "cpu", "--eval-every", "2", "--eval-games", "2",
        "--ckpt-every", "100", "--seed", "5", "--envs-per-worker", "2", "--snapshot-every", "1",
        "--pool-size", "2", "--hof-every", "2", "--hof-cap", "3", "--ratings-every", "3",
        "--ratings-games", "6", "--league", "0.1,0.6,0.3", *extra,
    ]  # fmt: skip


def test_run_with_hof_ratings_and_per_count_eval(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tools.train_ppo import COLUMNS, EVAL_COLUMNS, RATINGS_COLUMNS, main

    assert main(run_args(tmp_path, "r", "--updates", "6", "--players", "3,4")) == 0
    run = tmp_path / "r"
    rows = read(run / "log.csv")
    assert list(rows[0]) == COLUMNS and int(rows[-1]["hof_size"]) == 3
    ev = read(run / "eval.csv")
    assert list(ev[0]) == EVAL_COLUMNS
    at2 = [(e["opponent"].split(":")[0], e["players"]) for e in ev if e["update"] == "2"]
    assert ("random", "3") in at2 and ("random", "4") in at2 and ("heuristic", "3") in at2
    assert any(o == "hof_ref" for o, _ in at2)  # ppo-smoke reference
    assert {e["players"] for e in ev} == {"3", "4"}
    assert rows[1]["eval_composite"] != ""
    best = json.loads((run / "best.json").read_text())
    assert "composite" in best and 0.0 <= best["composite"] <= 1.0
    rt = read(run / "ratings.csv")
    assert list(rt[0]) == RATINGS_COLUMNS
    assert {r["update"] for r in rt} == {"3", "6"}
    by_kind = {r["entrant"]: r for r in rt if r["update"] == "3"}
    assert float(by_kind["heuristic"]["rating"]) == pytest.approx(1000.0)
    assert {"latest", "hof:ppo-smoke", "random"} <= set(by_kind)
    assert rows[2]["rating_latest"] != "" and rows[1]["rating_latest"] == ""
    pool = json.loads((run / "pool" / "pool.json").read_text())
    names = [e["name"] for e in pool["hof"]]
    assert names[0] == "ppo-smoke" and len(names) <= 3  # cap 3 including the permanent entry
    assert "m6" in names and "m2" in names
    assert all((run / "pool" / f"hof_{n}.npz").exists() for n in names)
    w = NetWeights.load(run / "ckpt_6.npz")
    assert w.meta["format_version"] == 2 and len(w.v_layers) == 2


def test_resume_roundtrip_with_hof_ema_and_ratings(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    from tools.train_ppo import main

    extra = ["--updates", "6", "--lr-decay", "--ent-coef-final", "0.001"]
    assert main(run_args(tmp_path, "s", *extra)) == 0
    assert main(run_args(tmp_path, "p", *extra, "--stop-after", "4")) == 0
    # stage B style change: more games per update from the resume point on is allowed
    assert (
        main(["--resume", str(tmp_path / "p"), "--stop-after", "0", "--games-per-update", "4"]) == 0
    )
    a, b = tmp_path / "s", tmp_path / "p"
    ca = torch.load(a / "ckpt_6.pt", weights_only=False)
    cb = torch.load(b / "ckpt_6.pt", weights_only=False)
    for k in ca["model"]:
        assert torch.equal(ca["model"][k], cb["model"][k]), k
    la, lb = ca["state"]["league"], cb["state"]["league"]
    assert la == lb and la["win_ema"] and len(la["hof"]) >= 2
    assert read(a / "ratings.csv") == read(b / "ratings.csv")
    assert read(a / "eval.csv") == read(b / "eval.csv")
    assert sorted(p.name for p in (a / "pool").glob("hof_*")) == sorted(
        p.name for p in (b / "pool").glob("hof_*")
    )


def test_resume_allows_games_per_update_override(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tools.train_ppo import main

    assert main(run_args(tmp_path, "g", "--updates", "2")) == 0
    assert (
        main(
            [
                "--resume",
                str(tmp_path / "g"),
                "--updates",
                "3",
                "--games-per-update",
                "6",
                "--allow-horizon-change",
            ]
        )
        == 0
    )
    cur = json.loads((tmp_path / "g" / "args.json").read_text())
    assert cur["games_per_update"] == 6
    log = json.loads((tmp_path / "g" / "resume_log.json").read_text())
    assert log[0]["overrides"]["games_per_update"] == {"old": 4, "new": 6}
    rows = read(tmp_path / "g" / "log.csv")
    assert [int(r["games"]) for r in rows] == [4, 8, 14]
