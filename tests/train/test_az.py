# pyright: basic, reportMissingImports=false
"""Phase 4 unit 8h: Gumbel AZ actors, replay, learner targets, trainer resume and export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bots.gumbel import GumbelConfig
from bots.net import load_weights
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS
from train.az_replay import Replay
from train.az_selfplay import (
    CURRENT,
    HEURISTIC,
    ActorConfig,
    play_selfplay,
    table_labels,
    win_credit,
)
from train.features import OBS_SIZE

NET = "ppo-long"


def _cfg(**kw: Any) -> ActorConfig:
    base: dict[str, Any] = {
        "players": (3,), "sims": 4, "concurrent": 2,
        "search": GumbelConfig(c_scale=0.3, gumbel_scale=1.0),
    }  # fmt: skip
    base.update(kw)
    return ActorConfig(**base)


@pytest.fixture(scope="module")
def selfplay() -> dict[str, Any]:
    return play_selfplay(load_weights(NET), 2, 11, _cfg())


# ---------------------------------------------------------------- 1/k win credit


@pytest.mark.parametrize("n", [3, 4, 5])
def test_win_credit_is_one_over_k(n: int) -> None:
    for k in range(1, n + 1):
        wl_win = (n - k) / (k * (n - 1))
        assert win_credit(wl_win, n) == pytest.approx(1.0 / k)
    assert win_credit(-1.0 / (n - 1), n) == 0.0


def test_win_credit_matches_engine_winners() -> None:
    import random

    for seed in range(6):
        rng = random.Random(seed)
        core = rules.initial_core(4)
        while core.phase != rules.PHASE_END:
            core = rules.apply(core, rng.choice(rules.legal_actions(core)))
        w = rules.winners(core)
        wl = rules.returns_for(core, "win_loss")
        for p in range(4):
            assert win_credit(wl[p], 4) == pytest.approx(1 / len(w) if p in w else 0.0)


# ---------------------------------------------------------------- actor samples


def test_samples_policy_targets_and_values(selfplay: dict[str, Any]) -> None:
    s = selfplay["samples"]
    st = selfplay["stats"]
    n = len(s["z"])
    assert n == st["decisions"] > 0 and st["games"] == 2
    ends = np.append(s["act_off"], len(s["act_id"]))
    for i in range(n):
        acts = s["act_id"][ends[i] : ends[i + 1]]
        pi = s["pi"][ends[i] : ends[i + 1]]
        assert len(acts) >= 2 and len(set(acts.tolist())) == len(acts)
        assert pi.sum() == pytest.approx(1.0, abs=1e-5) and (pi >= 0).all()
    assert (s["players"] == 3).all()
    np.testing.assert_array_equal(s["vmask"][:, :3], 1.0)
    np.testing.assert_array_equal(s["vmask"][:, 3:], 0.0)
    np.testing.assert_array_equal(s["z"][:, 3:], 0.0)
    # R is relative: win_loss and the margin both sum to ~0 over seats (margin unless clipped)
    assert np.all(np.abs(s["z"].sum(axis=1)) < 0.5 + 1e-6)
    assert ((s["prog"] > 0) & (s["prog"] < 1)).all()
    assert 0.0 <= st["differs"] <= st["decisions"]


def test_samples_ego_rotation_consistent(selfplay: dict[str, Any]) -> None:
    # within one game, the ego-ordered z of different seats are rotations of one vector
    s = selfplay["samples"]
    zs = {tuple(np.round(z[:3], 6)) for z in s["z"]}
    for z in zs:
        rots = {tuple(np.roll(np.array(z), -k)) for k in range(3)}
        assert rots & zs


def test_selfplay_deterministic(selfplay: dict[str, Any]) -> None:
    again = play_selfplay(load_weights(NET), 2, 11, _cfg())["samples"]
    for k, v in selfplay["samples"].items():
        np.testing.assert_array_equal(v, again[k], err_msg=k)


def test_frozen_seats_make_no_samples() -> None:
    w = load_weights(NET)
    r = play_selfplay(w, 2, 5, _cfg(frozen_share=1.0), {"hof:x": w})
    st = r["stats"]
    seats = st["tables"]["frozen"][0]
    assert "self" not in st["tables"] and 1 <= seats <= 2 * 2  # 1..n-1 learner seats per game
    assert len(r["samples"]["z"]) == st["decisions"]


@pytest.mark.parametrize(("fs", "hs"), [(0.2, 0.15), (0.0, 0.3), (0.5, 0.5), (0.3, 0.0)])
def test_table_shares_respected(fs: float, hs: float) -> None:
    rng = np.random.default_rng(3)
    cfg = _cfg(frozen_share=fs, heuristic_share=hs)
    counts = {"self": 0, "frozen": 0, "heuristic": 0}
    n_tab = 20_000
    for _ in range(n_tab):
        labels = table_labels(rng, 4, cfg, ["hof:a", "snap:b"])
        others = {x for x in labels if x != CURRENT}
        assert len(others) <= 1 and 1 <= labels.count(CURRENT) <= 4  # never mixed, learner seated
        kind = "self" if not others else "heuristic" if HEURISTIC in others else "frozen"
        if kind != "self":
            assert labels.count(CURRENT) <= 3
        counts[kind] += 1
    for kind, share in (("frozen", fs), ("heuristic", hs), ("self", 1 - fs - hs)):
        assert counts[kind] / n_tab == pytest.approx(share, abs=0.015), kind
    # no frozen labels: no frozen tables, the heuristic share is unchanged
    got = sum(HEURISTIC in table_labels(rng, 4, cfg, []) for _ in range(n_tab))
    assert got / n_tab == pytest.approx(hs, abs=0.015)


def test_heuristic_seats_make_no_samples_and_deterministic() -> None:
    w = load_weights(NET)
    cfg = _cfg(frozen_share=0.0, heuristic_share=1.0, concurrent=3)
    r = play_selfplay(w, 3, 8, cfg)
    st = r["stats"]
    assert set(st["tables"]) == {"heuristic"} and st["tables"]["heuristic"][3] == 3
    seats = st["tables"]["heuristic"][0]
    assert 3 <= seats <= 3 * 2  # 1..n-1 learner seats per game
    assert len(r["samples"]["z"]) == st["decisions"] > 0
    again = play_selfplay(w, 3, 8, cfg)
    for k, v in r["samples"].items():
        np.testing.assert_array_equal(v, again["samples"][k], err_msg=k)
    assert again["stats"]["tables"] == st["tables"]


def test_mixed_tables_batched_search() -> None:
    # self, frozen and heuristic tables in flight together in one batched search
    w = load_weights(NET)
    r = play_selfplay(w, 6, 2, _cfg(frozen_share=0.35, heuristic_share=0.35, concurrent=6),
                      {"hof:x": w})  # fmt: skip
    st = r["stats"]
    assert st["games"] == 6 and sum(t[3] for t in st["tables"].values()) == 6
    assert set(st["tables"]) == {"self", "frozen", "heuristic"}  # all three kinds (seed 2)
    assert len(r["samples"]["z"]) == st["decisions"]
    assert st["tables"]["self"][0] == 3 * st["tables"]["self"][3]
    for kind in ("frozen", "heuristic"):
        assert st["tables"][kind][3] <= st["tables"][kind][0] <= 2 * st["tables"][kind][3]


# ---------------------------------------------------------------- replay


def _dense_reference(s: dict[str, Any], i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oe = np.append(s["obs_off"], len(s["obs_idx"]))
    ae = np.append(s["act_off"], len(s["act_id"]))
    obs = np.zeros(OBS_SIZE, np.float32)
    obs[s["obs_idx"][oe[i] : oe[i + 1]]] = s["obs_tab"][s["obs_code"][oe[i] : oe[i + 1]]]
    pi = np.zeros(NUM_DISTINCT_ACTIONS, np.float32)
    pi[s["act_id"][ae[i] : ae[i + 1]]] = s["pi"][ae[i] : ae[i + 1]]
    return obs, pi > 0, pi


def test_replay_batch_densifies_in_order(selfplay: dict[str, Any]) -> None:
    s = selfplay["samples"]
    half = len(s["z"]) // 2
    a = {k: v for k, v in s.items()}
    rep = Replay(10**6)
    # split into two chunks via concat round trip
    rep.add(a)
    rep.add(a)
    assert rep.size == 2 * len(s["z"])
    idx = np.array([len(s["z"]) + 3, 0, half, 3, len(s["z"]) + 3])
    b = rep.batch(idx)
    for r, gi in enumerate(idx):
        i = int(gi) % len(s["z"])
        obs, _, pi = _dense_reference(s, i)
        np.testing.assert_array_equal(b["obs"][r], obs)
        np.testing.assert_array_equal(b["pi"][r], pi)
        assert b["mask"][r].sum() == (np.append(s["act_off"], len(s["act_id"]))[i + 1]
                                      - s["act_off"][i])  # fmt: skip
        np.testing.assert_array_equal(b["z"][r], s["z"][i])
        assert b["prog"][r] == s["prog"][i]


def test_replay_fifo_eviction(selfplay: dict[str, Any]) -> None:
    s = selfplay["samples"]
    n = len(s["z"])
    rep = Replay(capacity=n + 1)
    rep.add(s)
    rep.add(s)
    assert rep.size == n and len(rep.chunks) == 1 and rep.added == 2 * n
    rep = Replay(capacity=1)  # the newest chunk is always kept
    rep.add(s)
    assert rep.size == n


def test_obs_values_coded_losslessly(selfplay: dict[str, Any]) -> None:
    from train.features import encode

    s = selfplay["samples"]
    assert s["obs_code"].dtype == np.uint8 and len(s["obs_tab"]) <= 256
    # replay one game's first learner decision: the initial core of a 3p game
    core = rules.initial_core(3)
    from bots.gumbel import skip_forced

    core, _ = skip_forced(core)
    obs, _, _ = _dense_reference(s, 0)
    np.testing.assert_array_equal(obs, encode(core).astype(np.float32))


# ---------------------------------------------------------------- learner


def test_losses_and_value_only_warmup(selfplay: dict[str, Any]) -> None:
    torch = pytest.importorskip("torch")
    from train.az_learn import LearnConfig, losses, to_tensors, train_steps
    from train.model import BusNet

    torch.manual_seed(0)
    model = BusNet(hidden=(16, 16), aux=True)
    rep = Replay(10**6)
    rep.add(selfplay["samples"])
    b = rep.batch(np.arange(8))
    d = to_tensors(b, torch.device("cpu"))
    cfg = LearnConfig(batch=8, aux_coef=0.0)
    _, info = losses(model, d, cfg, value_only=False)
    logits, v = model(d["obs"], d["mask"])
    logp = torch.log_softmax(logits, -1)
    ce = -(d["pi"] * torch.where(d["mask"], logp, torch.zeros_like(logp))).sum(-1).mean()
    mse = (((v - d["z"]) ** 2) * d["vmask"]).sum() / d["vmask"].sum()
    assert info["loss_pi"] == pytest.approx(ce.item(), rel=1e-5)
    assert info["loss_v"] == pytest.approx(mse.item(), rel=1e-5)
    # zq target averages in the root search value
    _, info_q = losses(model, d, LearnConfig(batch=8, aux_coef=0.0, value_target="zq"), False)
    mse_q = (((v - 0.5 * (d["z"] + d["q"])) ** 2) * d["vmask"]).sum() / d["vmask"].sum()
    assert info_q["loss_v"] == pytest.approx(mse_q.item(), rel=1e-5)

    before = {k: t.clone() for k, t in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    train_steps(
        model, opt, rep, 3, np.random.default_rng(0), LearnConfig(batch=16, aux_coef=0.25),
        torch.device("cpu"), value_only=True,
    )  # fmt: skip
    after = model.state_dict()
    for k in before:
        same = torch.equal(before[k], after[k])
        if k.startswith(("trunk.", "pi.")):
            assert same, k  # policy side frozen during the value-only warm-up
        elif k.startswith(("v_trunk.", "v.", "aux_")):
            assert not same, k


# ---------------------------------------------------------------- trainer


def _read(p: Path) -> list[dict[str, str]]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


TIMING = {
    "sims_per_s", "games_per_h", "decisions_per_s", "rss_mb", "worker_rss_mb", "gen_s",
    "learn_s", "eval_s", "elapsed_s",
}  # fmt: skip


def _args(tmp: Path, name: str, *extra: str) -> list[str]:
    return [
        "--name", name, "--out-dir", str(tmp), "--hidden", "32,32", "--device", "cpu",
        "--workers", "0", "--players", "3", "--games-per-gen", "2", "--concurrent", "2",
        "--sims", "4", "--generations", "5", "--eval-every", "2", "--eval-games", "3",
        "--eval-nets", f"ppo-long={Path('models/ppo-long.npz').resolve()}", "--no-eval-heuristic",
        "--batch", "32", "--snapshot-every", "2", "--frozen-share", "0.5", "--warmup-gens", "1",
        "--replay-size", "1", "--ckpt-every", "100", "--seed", "4", *extra,
    ]  # fmt: skip


def test_resume_equals_straight_run_and_export_parity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from bots.net_forward import NetWeights
    from tools.train_az import main
    from train.model import BusNet

    assert main(_args(tmp_path, "straight")) == 0
    assert main(_args(tmp_path, "split", "--stop-after", "3")) == 0
    assert [r["gen"] for r in _read(tmp_path / "split" / "log.csv")] == ["1", "2", "3"]
    assert main(["--resume", str(tmp_path / "split"), "--stop-after", "0"]) == 0

    a = _read(tmp_path / "straight" / "log.csv")
    b = _read(tmp_path / "split" / "log.csv")
    assert [r["gen"] for r in b] == ["1", "2", "3", "4", "5"]
    det = [{k: v for k, v in r.items() if k not in TIMING} for r in a]
    assert det == [{k: v for k, v in r.items() if k not in TIMING} for r in b]
    assert _read(tmp_path / "straight" / "eval.csv") == _read(tmp_path / "split" / "eval.csv")
    assert {r["gen"] for r in _read(tmp_path / "split" / "eval.csv")} == {"0", "2", "4"}

    ca = torch.load(tmp_path / "straight" / "ckpt_5.pt", weights_only=False)
    cb = torch.load(tmp_path / "split" / "ckpt_5.pt", weights_only=False)
    for k in ca["model"]:
        assert torch.equal(ca["model"][k], cb["model"][k]), k
    sa, sb = ca["opt"]["state"], cb["opt"]["state"]
    assert sa.keys() == sb.keys() and len(sa) > 0
    for i in sa:
        for k in sa[i]:
            assert torch.equal(torch.as_tensor(sa[i][k]), torch.as_tensor(sb[i][k])), (i, k)
    assert int(a[0]["value_only"]) == 1 and int(a[1]["value_only"]) == 0

    # the npz export matches the torch model (aux heads are not exported)
    model = BusNet(hidden=(32, 32), aux=True)
    model.load_state_dict(ca["model"])
    model.eval()
    w = NetWeights.load(tmp_path / "straight" / "ckpt_5.npz")
    rng = np.random.default_rng(0)
    obs = (rng.random((6, OBS_SIZE)) < 0.05).astype(np.float32)
    mask = np.zeros((6, NUM_DISTINCT_ACTIONS), bool)
    mask[:, :40] = True
    with torch.no_grad():
        tl, tv = model(torch.from_numpy(obs), torch.from_numpy(mask))
    nl, nv = w.forward(obs, mask)
    np.testing.assert_allclose(nl[mask], tl.numpy()[mask], atol=1e-4)
    np.testing.assert_allclose(nv, tv.numpy(), atol=1e-5)


def test_heuristic_share_run_defining_on_resume(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tools.train_az import main

    base = _args(tmp_path, "h", "--heuristic-share", "0.5", "--frozen-share", "0.25",
                 "--stop-after", "1", "--eval-every", "0")  # fmt: skip
    assert main(base) == 0
    rows = _read(tmp_path / "h" / "log.csv")
    assert len(rows) == 1
    fr = [float(rows[0][f"tables_{k}"]) for k in ("self", "frozen", "heuristic")]
    assert sum(fr) == pytest.approx(1.0) and all(0 <= x <= 1 for x in fr)
    run = str(tmp_path / "h")
    assert main(["--resume", run, "--stop-after", "1", "--heuristic-share", "0.2"]) == 2
    assert main(["--resume", run, "--stop-after", "1"]) == 0
    import json

    assert json.loads((tmp_path / "h" / "args.json").read_text())["heuristic_share"] == 0.5
    assert len(_read(tmp_path / "h" / "log.csv")) == 2
    assert main(_args(tmp_path, "bad", "--heuristic-share", "0.9", "--frozen-share", "0.2")) == 2
