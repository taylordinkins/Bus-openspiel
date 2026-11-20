# pyright: basic, reportMissingImports=false
"""Observation versions side by side (Phase 4 interaction, direction 2).

Old v2 models (``ppo-long``, ``ppo-smoke``, ``runs/az*/best.npz``) load as v2 and play as
``net:``, ``gumbel:`` and ``mcts_net`` seats exactly as before; a v2 net widened to v3 (zero
columns for the new inputs) computes the same outputs on v3 observations, numpy and torch side.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyspiel
import pytest

import bus.game  # noqa: F401  (registers python_bus)
from bots import gumbel as gb
from bots import net as netmod
from bots.net import ModelInfo, NetBot, load_weights
from bots.net_forward import NetWeights, StaleModelError, layout_hash
from bus import rules
from bus.actions import NUM_DISTINCT_ACTIONS
from train import features as F
from train.league import save_weights

ROOT = Path(__file__).resolve().parents[2]
OLD_MODELS = [
    ROOT / "models/ppo-long.npz", ROOT / "models/ppo-smoke.npz", ROOT / "runs/az1/best.npz",
    ROOT / "runs/az2/best.npz",
]  # fmt: skip


def _cores(n: int, count: int, seed: int) -> list[rules.Core]:
    rng = random.Random(seed)
    out = []
    core = rules.initial_core(n)
    while len(out) < count:
        if core.phase == rules.PHASE_END:
            core = rules.initial_core(n)
        legal = rules.legal_actions(core)
        if len(legal) > 1 and rng.random() < 0.3:
            out.append(core)
        core = rules.apply(core, rng.choice(legal))
    return out


def _mask(cores: list[rules.Core]) -> np.ndarray:
    m = np.zeros((len(cores), NUM_DISTINCT_ACTIONS), bool)
    for i, c in enumerate(cores):
        m[i, rules.legal_actions(c)] = True
    return m


def _state(core: rules.Core) -> pyspiel.State:
    s = pyspiel.load_game("python_bus", {"players": core.players}).new_initial_state()
    s.core = core  # type: ignore[attr-defined]
    return s


@pytest.mark.parametrize("path", OLD_MODELS, ids=lambda p: p.parent.name + "/" + p.name)
def test_old_models_load_as_v2_and_encode_v2(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} not present (git-ignored run output)")
    w = NetWeights.load(path)
    assert w.features_version == F.FEATURES_V2 and w.obs_size == F.OBS_SIZE
    assert w.meta["layout_hash"] == layout_hash(F.FEATURES_V2)
    assert w.encoder is F.encode
    cores = _cores(4, 12, 3)
    obs = np.stack([F.encode(c) for c in cores])
    np.testing.assert_array_equal(np.stack([w.encode(c) for c in cores]), obs)


def test_manifest_models_play_as_before_in_every_seat_type() -> None:
    from bots.registry import make_bot

    w = load_weights("ppo-long")
    assert load_weights("ppo-smoke").features_version == F.FEATURES_V2
    cores = _cores(4, 10, 8)
    logits, _ = w.forward(np.stack([F.encode(c) for c in cores]), _mask(cores))
    net = NetBot("ppo-long")
    for c, lg in zip(cores, logits, strict=True):
        assert net.choose(_state(c), random.Random(0)) == int(np.argmax(lg))
    rng = random.Random(1)
    for kind in ("gumbel:8-ppo-long", "mcts_net:8-ppo-long"):
        bot = make_bot(kind)
        for c in cores[:4]:
            assert bot.choose(_state(c), rng) in rules.legal_actions(c)
    # gumbel's net evaluations go through the v2 encoder (same results as an explicit encode)
    g = gb.GumbelBot(w, 16)
    for c in cores[:3]:
        a = g.search(c, random.Random(0))
        b = gb.run_searches(gb.sparse_forward(w), [gb.GumbelSearch(c, 16)], F.encode)[0]
        assert a.action == b.action
        np.testing.assert_array_equal(a.visits, b.visits)


def test_manifest_features_version_must_match_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    real = netmod.resolve_model("ppo-long")
    for v in (F.FEATURES_V3, 9):
        bad = ModelInfo("bad", real.file, "x", "", real.players, v)
        monkeypatch.setattr(netmod, "manifest", lambda bad=bad: {"bad": bad})
        load_weights.cache_clear()
        try:
            with pytest.raises(StaleModelError, match="features"):
                load_weights("bad")
        finally:
            load_weights.cache_clear()


# ------------------------------------------------------------------ warm start (numpy)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_widened_numpy_net_equals_the_v2_net(n: int) -> None:
    w2 = load_weights("ppo-long")
    w3 = w2.widened(F.FEATURES_V3)
    assert w3.features_version == F.FEATURES_V3 and w3.obs_size == F.OBS_SIZE_V3
    assert w3.encoder is F.encode_v3
    cores = _cores(n, 40, n)
    mask = _mask(cores)
    l2, v2 = w2.forward(np.stack([F.encode(c) for c in cores]), mask)
    l3, v3 = w3.forward(np.stack([F.encode_v3(c) for c in cores]), mask)
    np.testing.assert_allclose(l3, l2, atol=2e-5)
    np.testing.assert_allclose(v3, v2, atol=2e-5)
    assert (l3.argmax(axis=1) == l2.argmax(axis=1)).all()


def test_widened_npz_round_trips_and_plays_as_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w2 = load_weights("ppo-long")
    save_weights(w2.widened(F.FEATURES_V3), tmp_path / "v3.npz", {"features_version": 2})
    w3 = NetWeights.load(tmp_path / "v3.npz")
    assert w3.features_version == F.FEATURES_V3 and w3.meta["layout_hash"] == layout_hash(2)
    cores = _cores(4, 6, 21)
    # the widened net searches exactly like the original (same decisions, same visits)
    for c in cores:
        a = gb.GumbelBot(w2, 24).search(c, random.Random(0))
        b = gb.GumbelBot(w3, 24).search(c, random.Random(0))
        assert a.action == b.action
        np.testing.assert_array_equal(a.visits, b.visits)
    # and as a manifest model in every seat type
    info = ModelInfo("v3net", "v3.npz", "x", "", (3, 4, 5), F.FEATURES_V3)
    monkeypatch.setattr(netmod, "manifest", lambda: {"v3net": info})
    monkeypatch.setattr(netmod, "MODELS_DIR", tmp_path)
    load_weights.cache_clear()
    try:
        from bots.registry import make_bot

        assert load_weights("v3net").features_version == F.FEATURES_V3
        rng = random.Random(2)
        bots = [NetBot("v3net"), make_bot("gumbel:6-v3net"), make_bot("mcts_net:6-v3net")]
        for bot in bots:
            for c in cores[:3]:
                assert bot.choose(_state(c), rng) in rules.legal_actions(c)
    finally:
        load_weights.cache_clear()


def test_mixed_version_tables_in_az_selfplay() -> None:
    """A v3 learner with v2 frozen seats: each net reads its own observation."""
    from train.az_selfplay import ActorConfig, play_selfplay

    w2 = load_weights("ppo-long")
    w3 = w2.widened(F.FEATURES_V3)
    cfg = ActorConfig(
        players=(3,), sims=4, concurrent=2, frozen_share=1.0,
        search=gb.GumbelConfig(c_scale=0.3, gumbel_scale=1.0),
    )  # fmt: skip
    res = play_selfplay(w3, 2, 5, cfg, {"hof:ppo-long": w2})
    s = res["samples"]
    assert s is not None and int(s["obs_idx"].max()) < F.OBS_SIZE_V3
    assert int(s["obs_idx"].max()) >= F.OBS_SIZE  # v3 blocks were recorded
    assert res["stats"]["tables"]["frozen"][3] == 2


# ------------------------------------------------------------------ warm start (torch)


def test_widen_model_is_function_preserving() -> None:
    torch = pytest.importorskip("torch")
    from train.model import BusNet, widen_model

    torch.manual_seed(0)
    src = BusNet(obs_size=F.OBS_SIZE, hidden=(64, 48), aux=True)
    dst = BusNet(obs_size=F.OBS_SIZE_V3, hidden=(64, 48), aux=True)
    widen_model(src, dst, F.FEATURES_V2, F.FEATURES_V3)
    cores = _cores(4, 32, 4)
    mask = torch.from_numpy(_mask(cores))
    o2 = torch.from_numpy(np.stack([F.encode(c) for c in cores]))
    o3 = torch.from_numpy(np.stack([F.encode_v3(c) for c in cores]))
    with torch.no_grad():
        l2, v2 = src(o2, mask)
        l3, v3 = dst(o3, mask)
    torch.testing.assert_close(l3, l2, atol=1e-5, rtol=0)
    torch.testing.assert_close(v3, v2, atol=1e-5, rtol=0)
    sd = dst.state_dict()
    assert torch.count_nonzero(sd["trunk.0.weight"][:, F.OBS_SIZE :]) == 0
    assert torch.count_nonzero(sd["v_trunk.0.weight"][:, F.OBS_SIZE :]) == 0
    with pytest.raises(ValueError):
        widen_model(dst, src, F.FEATURES_V3, F.FEATURES_V2)


def _az_args(tmp: Path, name: str, *extra: str) -> list[str]:
    return [
        "--name", name, "--out-dir", str(tmp), "--hidden", "32,32", "--device", "cpu",
        "--workers", "0", "--players", "3", "--games-per-gen", "1", "--concurrent", "1",
        "--sims", "2", "--generations", "1", "--eval-every", "0", "--batch", "16",
        "--frozen-share", "0", "--snapshot-every", "0", "--warmup-gens", "1", "--aux-coef", "0",
        "--reuse", "0", *extra,
    ]  # fmt: skip


def test_az_trainer_warm_starts_v3_from_a_v2_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from tools.train_az import main
    from train.model import BusNet
    from train.ppo import net_to_weights
    from train.runstate import save_pt

    torch.manual_seed(1)
    src = BusNet(obs_size=F.OBS_SIZE, hidden=(32, 32))
    meta = {"hidden": [32, 32], "features_version": 1, "layout_hash": layout_hash(1),
            "obs_size": F.OBS_SIZE, "value_trunk": "separate"}  # fmt: skip
    save_pt(tmp_path / "v2.pt", {"model": src.state_dict(), "meta": meta})
    extra = ("--features", "v3", "--init-from", str(tmp_path / "v2.pt"),
             "--value-quantity", "leader", "--opp-temp", "2")  # fmt: skip
    assert main(_az_args(tmp_path, "w3", *extra)) == 0
    ckpts = sorted((tmp_path / "w3").glob("ckpt_*.npz"))
    assert ckpts
    w = NetWeights.load(ckpts[-1])
    assert w.features_version == F.FEATURES_V3
    cores = _cores(3, 10, 6)
    mask = _mask(cores)
    l2, v2 = net_to_weights(src).forward(np.stack([F.encode(c) for c in cores]), mask)
    l3, v3 = w.forward(np.stack([F.encode_v3(c) for c in cores]), mask)
    np.testing.assert_allclose(l3, l2, atol=2e-5)  # --reuse 0: no training steps were taken
    np.testing.assert_allclose(v3, v2, atol=2e-5)
    # a v3 checkpoint cannot start a v2 run
    bad = ("--features", "v2", "--init-from", str(ckpts[-1].with_suffix(".pt")))
    assert main(_az_args(tmp_path, "w2", *bad)) == 2
