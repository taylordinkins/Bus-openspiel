# pyright: basic, reportMissingImports=false
"""Phase 4 unit 8e: dense relative-score reward, return-to-go targets, aux heads, stone counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bus.actions import CLOCK_PAUSE  # noqa: E402
from bus.rules import final_scores, returns_for  # noqa: E402
from tools.train_ppo import build_parser, load_init_from, main  # noqa: E402
from train.env import BusEnv, potentials  # noqa: E402
from train.export import export_npz  # noqa: E402
from train.features import FEATURES_VERSION  # noqa: E402
from train.model import BusNet  # noqa: E402
from train.ppo import PPOConfig, net_to_weights, ppo_update  # noqa: E402
from train.rollout import (  # noqa: E402
    RolloutConfig,
    _dense_targets,
    aux_targets,
    gae,
    play_games,
)
from train.runstate import ResumeError, save_pt  # noqa: E402


def _weights() -> Any:
    torch.manual_seed(1)
    return net_to_weights(BusNet(hidden=(16, 16)))


def _trajectories(samples: dict[str, Any]) -> list[slice]:
    """Row ranges of single-seat trajectories (``prog`` restarts at each one)."""
    prog = samples["prog"]
    starts = [0] + [i for i in range(1, len(prog)) if prog[i] <= prog[i - 1]]
    return [slice(a, b) for a, b in zip(starts, [*starts[1:], len(prog)], strict=True)]


def test_potentials_zero_sum_and_matches_margin() -> None:
    env = BusEnv(4)
    env.reset(seed=3)
    assert potentials(env.state.core) == pytest.approx(np.zeros(4))
    rng = np.random.default_rng(0)
    while not env.done:
        env.step(int(rng.choice(env.legal_actions())))
    phi = potentials(env.state.core)
    assert phi.sum() == pytest.approx(0, abs=1e-5)
    raw = [p for p in phi]
    clipped = np.clip(raw, -1, 1)
    assert clipped == pytest.approx(returns_for(env.state.core, "margin"), abs=1e-6)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("mix", [0.0, 0.5, 0.8])
def test_dense_rewards_telescope(n: int, mix: float) -> None:
    rng = np.random.default_rng(n)
    for g in range(4):
        env = BusEnv(n)
        env.reset(seed=100 * n + g)
        decisions: list[tuple[int, np.ndarray[Any, Any]]] = []
        while not env.done:
            decisions.append((env.current_seat, potentials(env.state.core)))
            env.step(int(rng.choice(env.legal_actions())))
        core = env.state.core
        phi_t = potentials(core)
        wlr = returns_for(core, "win_loss")
        sc = np.asarray(final_scores(core), float)
        for seat in range(n):
            ph = np.stack([p for s, p in decisions if s == seat])
            if not len(ph):
                continue
            vclip = [0, 0]
            rew, vt = _dense_targets(ph, phi_t, wlr, seat, mix, vclip)
            margin = (sc[seat] - (sc.sum() - sc[seat]) / (n - 1)) / 10
            assert rew.sum() == pytest.approx(mix * wlr[seat] + (1 - mix) * margin, abs=1e-5)
            assert vclip[1] == len(ph) * n
            assert np.abs(vt).max() <= 1.0


def test_value_target_entry0_equals_lambda1_return_and_clip() -> None:
    cfg = RolloutConfig(players=(3, 4, 5), mix=0.5, lam=1.0, reward="dense", envs_per_worker=3)
    res = play_games(_weights(), 6, 5, cfg)
    s = res["samples"]
    assert s is not None
    _, ret = (None, s["adv"] + s["value"])
    unclipped = np.abs(ret) < 1.0 - 1e-4
    assert unclipped.any()
    assert np.allclose(ret[unclipped], s["vtarget"][:, 0][unclipped], atol=1e-4)
    assert np.abs(s["vtarget"]).max() <= 1.0
    assert 0 <= res["stats"]["vclip"][0] <= res["stats"]["vclip"][1]


def test_terminal_mode_matches_independent_reconstruction() -> None:
    cfg = RolloutConfig(players=(3, 4, 5), mix=0.7, lam=0.99, envs_per_worker=3)
    s = play_games(_weights(), 6, 11, cfg)["samples"]
    assert s is not None
    for sl in _trajectories(s):
        vt = s["vtarget"][sl]
        assert (vt == vt[0]).all()  # one final return vector per trajectory
        rew = np.zeros(sl.stop - sl.start, np.float32)
        rew[-1] = vt[0, 0]
        adv, _ = gae(rew, s["value"][sl], 1.0, 0.99)
        assert np.array_equal(adv, s["adv"][sl])


def test_aux_targets_ties_and_ego_order() -> None:
    class S:
        def __init__(self, pts: int, st: int = 0, last: int | None = None) -> None:
            self.points, self.stones, self.last_score_move = pts, st, last

    class C:
        phase = "end"
        players = 4
        seats = [S(10, 0, 5), S(10, 0, 5), S(4, 1, 2), S(7, 0, 3)]

    sc, rk, wn = aux_targets(C, 2)  # ego order: seats 2,3,0,1
    assert sc[:4] == pytest.approx([3 / 20, 7 / 20, 10 / 20, 10 / 20]) and sc[4] == 0
    assert list(rk) == [3, 2, 0, 0, -1]
    assert wn[:4] == pytest.approx([0, 0, 0.5, 0.5]) and wn[4] == 0


def test_aux_samples_shapes() -> None:
    cfg = RolloutConfig(players=(3, 5), envs_per_worker=2)
    s = play_games(_weights(), 3, 2, cfg)["samples"]
    assert s is not None
    k = len(s["action"])
    assert s["ascore"].shape == (k, 5) and s["arank"].shape == (k, 5) and s["awin"].shape == (k, 5)
    assert np.allclose(s["awin"].sum(1), 1.0)


def test_stone_counts_use_clock_pause() -> None:
    cfg = RolloutConfig(players=(4,), envs_per_worker=2, collect=False)
    st = play_games(_weights(), 4, 3, cfg)["stats"]
    assert st["seats_cur"] == 16 and st["seats_oth"] == 0 and st["stones_oth"] == 0
    assert st["stones_cur"] >= 0
    assert CLOCK_PAUSE > 0


def _aux_data() -> dict[str, Any]:
    cfg = RolloutConfig(players=(3, 4), envs_per_worker=2, reward="dense", mix=0.5)
    s = play_games(_weights(), 6, 9, cfg)["samples"]
    assert s is not None
    return {k: torch.from_numpy(v) for k, v in s.items() if k not in ("value", "prog")}


def test_aux_update_trains_finite() -> None:
    torch.manual_seed(0)
    model = BusNet(hidden=(16, 16), aux=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    d = _aux_data()
    out = ppo_update(
        model, opt, d, PPOConfig(aux_coef=0.5, epochs=1, minibatches=2), 0.01, torch.Generator()
    )
    for k in ("aux_score", "aux_rank", "aux_win", "value_loss"):
        assert np.isfinite(out[k]) and out[k] >= 0
    # without aux heads / coefficient 0 nothing extra is reported
    plain = BusNet(hidden=(16, 16))
    out2 = ppo_update(
        plain, torch.optim.Adam(plain.parameters()), d, PPOConfig(epochs=1), 0.01, torch.Generator()
    )
    assert "aux_win" not in out2


def test_aux_heads_not_in_npz_export(tmp_path: Path) -> None:
    model = BusNet(hidden=(16, 16), aux=True)
    export_npz(model, tmp_path / "m.npz")
    plain = BusNet(hidden=(16, 16))
    export_npz(plain, tmp_path / "p.npz")
    a, b = np.load(tmp_path / "m.npz"), np.load(tmp_path / "p.npz")
    assert sorted(a.files) == sorted(b.files)
    assert not any("aux" in f for f in a.files)


def _ckpt(path: Path, hidden: tuple[int, ...]) -> BusNet:
    torch.manual_seed(5)
    net = BusNet(hidden=hidden)
    meta = {"hidden": list(hidden), "features_version": FEATURES_VERSION, "obs_size": net.obs_size}
    save_pt(path, {"model": net.state_dict(), "update": 3, "meta": meta})
    return net


def test_init_from_without_aux_heads(tmp_path: Path) -> None:
    src = _ckpt(tmp_path / "w.pt", (16, 16))
    args = build_parser().parse_args(
        ["--hidden", "16,16", "--aux-coef", "0.5", "--init-from", str(tmp_path / "w.pt")]
    )
    model = BusNet(hidden=(16, 16), aux=True)
    fresh = model.aux_win.weight.detach().clone()
    load_init_from(model, args)
    for k, v in src.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
    assert torch.equal(model.aux_win.weight, fresh)
    # a real mismatch is still an error
    args32 = build_parser().parse_args(
        ["--hidden", "32,32", "--aux-coef", "0.5", "--init-from", str(tmp_path / "w.pt")]
    )
    with pytest.raises(ResumeError):
        load_init_from(BusNet(hidden=(32, 32), aux=True), args32)


def test_cli_dense_run_and_resume_rules(tmp_path: Path, capsys: Any) -> None:
    common = ["--out-dir", str(tmp_path), "--games-per-update", "4", "--workers", "0",
              "--device", "cpu", "--eval-every", "0", "--envs-per-worker", "2",
              "--league", "1,0,0", "--ckpt-every", "100", "--hidden", "16,16"]  # fmt: skip
    argv = ["--name", "d", "--updates", "2", "--reward", "dense", "--aux-coef", "0.5",
            "--gae-lambda", "0.95", "--mix", "0.5", *common]  # fmt: skip
    assert main(["--stop-after", "1", *argv]) == 0
    run = str(tmp_path / "d")
    saved = json.loads((tmp_path / "d" / "args.json").read_text())
    assert saved["reward"] == "dense"
    rows = (tmp_path / "d" / "log.csv").read_text().splitlines()
    head, row = rows[0].split(","), dict(zip(rows[0].split(","), rows[1].split(","), strict=True))
    assert {"vclip", "stones_cur", "aux_win"} <= set(head)
    assert np.isfinite(float(row["aux_win"])) and row["vclip"] != ""
    capsys.readouterr()
    assert main(["--resume", run, "--reward", "terminal"]) == 2
    assert "--reward" in capsys.readouterr().err
    assert main(["--resume", run, "--aux-coef", "0"]) == 2
    assert "aux-coef" in capsys.readouterr().err
    assert main(["--resume", run, "--aux-coef", "0.25"]) == 0


def test_old_args_resume_as_terminal(tmp_path: Path) -> None:
    from train.runstate import merge_resume_args

    defaults = vars(build_parser().parse_args([]))
    saved = {k: v for k, v in defaults.items() if k not in ("reward", "aux_coef")}
    merged, _ = merge_resume_args(saved, {}, defaults)
    assert merged["reward"] == "terminal" and merged["aux_coef"] == 0.0
    with pytest.raises(ResumeError):
        merge_resume_args(saved, {"reward": "dense"}, defaults)
