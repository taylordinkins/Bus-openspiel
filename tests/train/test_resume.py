# pyright: basic, reportMissingImports=false
"""Phase 4 unit 8a: resume, atomic writes, retention, signals, worker death, shm hygiene."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tools.train_ppo import COLUMNS, EVAL_COLUMNS, Runner, WorkerFailure, main  # noqa: E402
from train import shm  # noqa: E402
from train.atomic import atomic_write, atomic_write_text  # noqa: E402
from train.runstate import (  # noqa: E402
    ResumeError,
    apply_retention,
    find_latest_checkpoint,
    merge_resume_args,
    truncate_csv,
)

ROOT = Path(__file__).resolve().parents[2]
TIMING = {
    "env_s", "learn_s", "decisions_per_s", "games_per_s", "worker_busy", "elapsed_s", "eval_s",
    "rss_mb",
}  # fmt: skip


def base_args(tmp: Path, name: str, *extra: str, workers: int = 0) -> list[str]:
    return [
        "--name", name, "--out-dir", str(tmp), "--games-per-update", "4", "--workers",
        str(workers), "--hidden", "32,32", "--device", "cpu", "--eval-every", "2",
        "--eval-games", "4", "--ckpt-every", "100", "--seed", "3", "--envs-per-worker", "2",
        "--lr-decay", "--ent-coef-final", "0.0", "--snapshot-every", "1", "--pool-size", "2",
        "--league", "0.3,0.4,0.3", *extra,
    ]  # fmt: skip


def read_csv(p: Path) -> list[dict[str, str]]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def det(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


# ---------------------------------------------------------------- (a) bit-exact resume


def test_resume_equals_straight_run(tmp_path: Path) -> None:
    assert main(base_args(tmp_path, "straight", "--updates", "5")) == 0
    assert main(base_args(tmp_path, "split", "--updates", "5", "--stop-after", "3")) == 0
    assert [r["update"] for r in read_csv(tmp_path / "split" / "log.csv")] == ["1", "2", "3"]
    # the stop-after override is itself a whitelisted change: lift it on resume
    assert main(["--resume", str(tmp_path / "split"), "--stop-after", "0"]) == 0

    a = read_csv(tmp_path / "straight" / "log.csv")
    b = read_csv(tmp_path / "split" / "log.csv")
    assert [r["update"] for r in b] == ["1", "2", "3", "4", "5"]
    assert det(a) == det(b)
    assert read_csv(tmp_path / "straight" / "eval.csv") == read_csv(tmp_path / "split" / "eval.csv")

    ca = torch.load(tmp_path / "straight" / "ckpt_5.pt", weights_only=False)
    cb = torch.load(tmp_path / "split" / "ckpt_5.pt", weights_only=False)
    for k in ca["model"]:
        assert torch.equal(ca["model"][k], cb["model"][k]), k
    sa, sb = ca["opt"]["state"], cb["opt"]["state"]
    assert float(sa[0]["step"]) > 5  # several optimizer steps per update
    assert sa.keys() == sb.keys()
    for i in sa:
        assert float(sa[i]["step"]) == float(sb[i]["step"])
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(sa[i][k], sb[i][k])
    assert ca["meta"]["pool_ids"] == cb["meta"]["pool_ids"] == [4, 5]
    assert torch.equal(ca["state"]["gen"], cb["state"]["gen"])
    assert ca["state"]["total_games"] == cb["state"]["total_games"] == 20
    pool = lambda r: sorted(p.name for p in (tmp_path / r / "pool").glob("snap_*.npz"))  # noqa: E731
    assert pool("straight") == pool("split") == ["snap_4.npz", "snap_5.npz"]


# ---------------------------------------------------------------- (b) CSV continuity


def test_csv_truncated_to_checkpoint_on_resume(tmp_path: Path) -> None:
    run = tmp_path / "c"
    args = base_args(tmp_path, "c", "--updates", "4", "--ckpt-every", "2")
    assert main(args) == 0
    # simulate a crash after update 3: the update-4 checkpoint never landed
    for f in (run / "latest.json", run / "ckpt_4.pt", run / "ckpt_4.npz"):
        f.unlink()
    with open(run / "log.csv", "a") as fh:
        fh.write("5,garbage")  # torn last line
    assert main(["--resume", str(run)]) == 0
    rows = read_csv(run / "log.csv")
    assert [r["update"] for r in rows] == ["1", "2", "3", "4"]
    assert (run / "log.csv").read_text().count("update,games") == 1
    ev = read_csv(run / "eval.csv")
    assert [e["update"] for e in ev if e["opponent"] == "random"] == ["2", "4"]
    assert list(ev[0]) == EVAL_COLUMNS and list(rows[0]) == COLUMNS


def test_truncate_csv_header_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ResumeError):
        truncate_csv(p, ["update", "b"], 5)
    q = tmp_path / "new.csv"
    assert truncate_csv(q, ["update", "b"], 0) == 0
    assert q.read_text().strip() == "update,b"


# ---------------------------------------------------------------- (c) resume arguments


def test_merge_resume_args_rules() -> None:
    defaults = {"hidden": "512,512", "seed": 0, "workers": 4, "players": (4,), "name": "run"}
    saved = {"hidden": "32", "seed": 3, "workers": 2, "players": [4], "name": "x"}
    merged, ch = merge_resume_args(saved, {"workers": 8, "players": (3, 4)}, defaults)
    assert merged["workers"] == 8 and merged["hidden"] == "32"
    assert ch == {"workers": {"old": 2, "new": 8}, "players": {"old": [4], "new": [3, 4]}}
    # the same value as saved is not a change, even for a locked option
    assert merge_resume_args(saved, {"hidden": "32", "players": (4,)}, defaults)[1] == {}
    for bad in ({"hidden": "64"}, {"seed": 9}, {"name": "y"}, {"out_dir": "z"}):
        with pytest.raises(ResumeError, match="cannot change --"):
            merge_resume_args(saved, bad, defaults)


def test_resume_cli_overrides_and_rejections(tmp_path: Path, capsys: Any) -> None:
    run = tmp_path / "r"
    assert main(base_args(tmp_path, "r", "--updates", "2")) == 0
    for bad in (["--hidden", "64,64"], ["--seed", "4"], ["--name", "other"]):
        assert main(["--resume", str(run), *bad]) == 2
    assert "cannot change --hidden" in capsys.readouterr().err
    # extend the horizon and override eval cadence
    assert main(["--resume", str(run), "--updates", "3", "--eval-every", "3"]) == 0
    log = json.loads((run / "resume_log.json").read_text())
    assert len(log) == 1 and log[0]["from_update"] == 2
    assert log[0]["overrides"]["updates"] == {"old": 2, "new": 3}
    assert log[0]["overrides"]["eval_every"] == {"old": 2, "new": 3}
    assert log[0]["time"]
    assert json.loads((run / "args.original.json").read_text())["updates"] == 2
    cur = json.loads((run / "args.json").read_text())
    assert cur["updates"] == 3 and cur["eval_every"] == 3 and cur["hidden"] == "32,32"
    # a finished run resumed again does nothing but is logged, original stays the first one
    assert main(["--resume", str(run)]) == 0
    assert len(json.loads((run / "resume_log.json").read_text())) == 2
    assert json.loads((run / "args.original.json").read_text())["updates"] == 2
    assert len(read_csv(run / "log.csv")) == 3


def test_fresh_run_refuses_existing_run_dir(tmp_path: Path, capsys: Any) -> None:
    assert main(base_args(tmp_path, "d", "--updates", "1")) == 0
    assert main(base_args(tmp_path, "d", "--updates", "1")) == 2
    assert "--resume" in capsys.readouterr().err


def test_resume_rejects_incompatible_checkpoint(tmp_path: Path, monkeypatch: Any) -> None:
    import tools.train_ppo as T

    assert main(base_args(tmp_path, "f", "--updates", "1")) == 0
    monkeypatch.setattr(T, "FEATURES_VERSION", T.FEATURES_VERSION + 1)
    assert main(["--resume", str(tmp_path / "f")]) == 2


# ---------------------------------------------------------------- (d) atomic writes


def _latest_update(run: Path) -> int:
    found = find_latest_checkpoint(run)
    assert found is not None
    return found[0]


def test_atomic_write_failure_leaves_no_partial_file(tmp_path: Path) -> None:
    p = tmp_path / "pool.json"
    atomic_write_text(p, "old")

    def boom(f: Any) -> None:
        f.write(b"par")
        raise OSError("disk full")

    with pytest.raises(OSError):
        atomic_write(p, boom)
    assert p.read_text() == "old"
    assert [x.name for x in tmp_path.iterdir()] == ["pool.json"]
    q = tmp_path / "new.bin"
    with pytest.raises(OSError):
        atomic_write(q, boom)
    assert not q.exists() and [x.name for x in tmp_path.iterdir()] == ["pool.json"]


def test_corrupt_latest_and_truncated_ckpt_fall_back(tmp_path: Path) -> None:
    assert main(base_args(tmp_path, "g", "--updates", "3", "--ckpt-every", "1")) == 0
    run = tmp_path / "g"
    assert json.loads((run / "latest.json").read_text())["update"] == 3
    (run / "latest.json").write_text("{not json")
    assert _latest_update(run) == 3
    data = (run / "ckpt_3.pt").read_bytes()
    (run / "ckpt_3.pt").write_bytes(data[: len(data) // 2])  # truncated newest
    assert _latest_update(run) == 2
    (run / "latest.json").unlink()
    assert _latest_update(run) == 2
    assert main(["--resume", str(run)]) == 0  # continues from 2, redoes 3
    assert [r["update"] for r in read_csv(run / "log.csv")] == ["1", "2", "3"]


# ---------------------------------------------------------------- (e) retention


def test_retention_keeps_last3_milestones_and_best(tmp_path: Path) -> None:
    for u in (10, 20, 250, 260, 270, 480, 500, 510, 520, 530):
        (tmp_path / f"ckpt_{u}.pt").write_bytes(b"x")
        (tmp_path / f"ckpt_{u}.npz").write_bytes(b"x")
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "best.npz").write_bytes(b"x")
    gone = apply_retention(tmp_path, keep_last=3, keep_every=250)
    left = sorted(int(p.stem.split("_")[1]) for p in tmp_path.glob("ckpt_*.pt"))
    assert left == [250, 500, 510, 520, 530]
    assert gone == [10, 20, 260, 270, 480]
    assert sorted(p.name for p in tmp_path.glob("ckpt_*.npz")) == [f"ckpt_{u}.npz" for u in left]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "best.npz").exists()


def test_run_keeps_best_and_retention(tmp_path: Path) -> None:
    args = base_args(tmp_path, "k", "--updates", "6", "--ckpt-every", "1", "--keep-every", "4")
    assert main(args) == 0
    run = tmp_path / "k"
    assert sorted(int(p.stem.split("_")[1]) for p in run.glob("ckpt_*.pt")) == [4, 5, 6]
    best = json.loads((run / "best.json").read_text())
    assert (run / "best.pt").exists() and (run / "best.npz").exists()
    assert best["update"] in (2, 4, 6) and 0.0 <= best["lo"] <= 1.0


# ---------------------------------------------------------------- (g) worker death


def test_worker_death_is_recovered(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    flag = tmp_path / "killed.flag"
    monkeypatch.setenv("BUS_TRAIN_KILL_ONCE", str(flag))
    args = base_args(tmp_path, "w", "--updates", "2", "--eval-every", "0", workers=2)
    assert main(args) == 0
    assert flag.exists()
    out = capsys.readouterr().out
    assert "rebuilding the worker pool" in out
    rows = read_csv(tmp_path / "w" / "log.csv")
    assert [r["update"] for r in rows] == ["1", "2"] and rows[0]["illegal"] == "0"
    assert not list(shm.spill_roots()[0].glob(f"{shm.PREFIX}{os.getpid()}_*"))


def test_worker_death_without_retries_raises(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BUS_TRAIN_KILL_ONCE", str(tmp_path / "flag"))
    from train.rollout import RolloutConfig

    from .test_ppo import _random_weights

    cfg = RolloutConfig(players=(4,), envs_per_worker=2)
    tasks = [(_random_weights(), 2, 1, cfg, None, 0, {})]
    r = Runner(1, timeout=60, max_retries=0)
    try:
        with pytest.raises(WorkerFailure):
            r.map(tasks)
    finally:
        r.kill()


# ---------------------------------------------------------------- (h) stale shm cleanup


def test_stale_shm_cleanup_spares_live_pids(tmp_path: Path) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    root = shm.spill_roots()[0]
    mine = os.getpid()
    names = {
        "dead": f"{shm.PREFIX}{dead.pid}_aaaa",
        "live": f"{shm.PREFIX}{mine}_bbbb",
        "parent": f"{shm.PREFIX}{os.getppid()}_cccc",
        "legacy_new": f"{shm.PREFIX}legacy1",
        "legacy_old": f"{shm.PREFIX}legacy2",
        "other": "something_else",
    }
    dirs = {k: root / v for k, v in names.items()}
    try:
        for d in dirs.values():
            d.mkdir()
            (d / "obs.npy").write_bytes(b"x")
        old = time.time() - 2 * shm.LEGACY_MAX_AGE_S
        os.utime(dirs["legacy_old"], (old, old))
        removed = shm.cleanup_stale()
        assert str(dirs["dead"]) in removed and str(dirs["legacy_old"]) in removed
        assert not dirs["dead"].exists() and not dirs["legacy_old"].exists()
        for k in ("live", "parent", "legacy_new", "other"):
            assert dirs[k].exists(), k
        assert shm.cleanup_own(mine) >= 1 and not dirs["live"].exists()
        assert dirs["parent"].exists()
    finally:
        for d in dirs.values():
            if d.exists():
                import shutil

                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- (f) SIGTERM


def test_sigterm_saves_cleans_and_resumes(tmp_path: Path) -> None:
    run = tmp_path / "s"
    cmd = [
        sys.executable, "-m", "tools.train_ppo", *base_args(tmp_path, "s", "--updates", "400",
        "--eval-every", "0", workers=2),
    ]  # fmt: skip
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if (run / "log.csv").exists() and len(read_csv(run / "log.csv")) >= 2:
                break
            assert proc.poll() is None, proc.stdout.read() if proc.stdout else ""
            time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out
    assert "--resume" in out
    latest = json.loads((run / "latest.json").read_text())
    ck = torch.load(run / latest["pt"], weights_only=False)
    n = latest["update"]
    assert ck["update"] == n >= 2
    assert [r["update"] for r in read_csv(run / "log.csv")][-1] == str(n)
    for root in shm.spill_roots():
        assert not list(root.glob(f"{shm.PREFIX}{proc.pid}_*"))
    r = subprocess.run(
        [sys.executable, "-m", "tools.train_ppo", "--resume", str(run), "--stop-after", "2",
         "--updates", str(n + 2)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    rows = read_csv(run / "log.csv")
    assert [int(x["update"]) for x in rows] == list(range(1, n + 3))
    assert json.loads((run / "latest.json").read_text())["update"] == n + 2


def test_stop_signals_debounce_and_second_signal() -> None:
    from tools.train_ppo import ForcedExit, StopSignals

    s = StopSignals()
    s._handle(2, None)
    s._handle(2, None)  # the duplicate from `uv run` forwarding: ignored
    assert s.requested and s.count == 1
    s.first_at -= 5.0
    with pytest.raises(ForcedExit):
        s._handle(15, None)
