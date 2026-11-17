# pyright: basic
"""tools.eval_search: agent parsing, rotation, seeding, prior temperature, CLI, resume."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from open_spiel.python.algorithms import mcts

import bus.game  # noqa: F401  (registers the game)
from bots.net_forward import NetWeights
from tools.eval_search import (
    AgentSpec,
    PathNetEvaluator,
    baseline_specs,
    build_jobs,
    main,
    make_puct,
    parse_agent,
    run_jobs,
    seat_types,
    summarize,
)

NET = str(Path(__file__).resolve().parents[2] / "models" / "ppo-smoke.npz")


def test_parse_agent() -> None:
    assert parse_agent("net") == AgentSpec("net")
    assert parse_agent("net_sample").sample
    assert parse_agent("heuristic").kind == "heuristic"
    s = parse_agent("mcts:50,c=2.5,t=1.5,fpu=-0.2")
    assert (s.kind, s.sims, s.uct_c, s.prior_temp, s.fpu) == ("mcts", 50, 2.5, 1.5, -0.2)
    for bad in ("mcts:", "mcts:0", "mcts:5,x=1", "mcts:5,t=0", "random", "net:foo"):
        with pytest.raises(ValueError):
            parse_agent(bad)


def test_baseline_tables() -> None:
    assert baseline_specs("ppo_greedy", 4) == ["net"] * 3
    assert baseline_specs("ppo_sample", 3) == ["net_sample"] * 2
    assert baseline_specs("mixed", 5) == ["net_sample", "heuristic", "net_sample", "heuristic"]
    with pytest.raises(ValueError):
        baseline_specs("nope", 4)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_candidate_rotates_through_every_seat(n: int) -> None:
    opp = [f"o{j}" for j in range(n - 1)]
    seats = Counter()
    for g in range(3 * n):
        types, cs = seat_types("C", opp, g)
        assert types[cs] == "C" and cs == g % n
        assert [types[(cs + 1 + j) % n] for j in range(n - 1)] == opp
        seats[cs] += 1
    assert set(seats) == set(range(n)) and set(seats.values()) == {3}


def test_jobs_round_up_and_share_seeds_across_candidates() -> None:
    jobs = build_jobs(["net", "mcts:3"], ["heuristic"], [3, 4], 5, 7, NET, NET)
    per = Counter((j["candidate"], j["players"]) for j in jobs)
    assert per[("net", 3)] == 6 and per[("net", 4)] == 8
    seeds = {(j["candidate"], j["players"]): j["seed"] for j in jobs}
    assert seeds[("net", 3)] == seeds[("mcts:3", 3)] != seeds[("net", 4)]


def test_prior_temperature() -> None:
    w = NetWeights.load(NET)
    state = bus.game.pyspiel.load_game("python_bus", {"players": 4}).new_initial_state()
    rng = random.Random(1)
    for _ in range(30):
        state.apply_action(rng.choice(state.legal_actions()))
    base = dict(PathNetEvaluator(w).prior(state))
    sharp = dict(PathNetEvaluator(w, 0.5).prior(state))
    flat = dict(PathNetEvaluator(w, 4.0).prior(state))
    assert set(base) == set(sharp) == set(flat) == set(state.legal_actions())
    for d in (base, sharp, flat):
        assert sum(d.values()) == pytest.approx(1.0, abs=1e-6)
    top = max(base, key=lambda a: base[a])
    assert sharp[top] >= base[top] >= flat[top]


def test_fpu_zero_matches_openspiel_puct() -> None:
    f = make_puct(0.0)
    for count, reward in ((0, 0.0), (3, 1.5), (5, -2.0)):
        node = mcts.SearchNode(1, 0, 0.3)
        node.explore_count, node.total_reward = count, reward
        assert f(node, 10, 1.5) == pytest.approx(node.puct_value(10, 1.5))
    node = mcts.SearchNode(1, 0, 0.3)
    assert make_puct(-0.5)(node, 4, 1.0) == pytest.approx(-0.5 + 0.3 * 2.0)


def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = sorted(rows, key=lambda r: (r["candidate"], r["baseline"], r["players"], r["game"]))
    return [{k: v for k, v in r.items() if k != "cand_seconds"} for r in keep]


def test_reproducible_across_worker_counts() -> None:
    jobs = build_jobs(["net_sample", "mcts:2"], ["mixed"], [3], 3, 1, NET, NET)
    a: list[dict[str, Any]] = []
    b: list[dict[str, Any]] = []
    run_jobs(jobs, 1, a.append)
    run_jobs(jobs, 2, b.append)
    assert _strip(a) == _strip(b) and len(a) == 6
    summ = summarize(a)
    assert {(r["candidate"], r["players"], r["games"]) for r in summ} == {
        ("net_sample", 3, 3), ("mcts:2", 3, 3),
    }  # fmt: skip
    for r in summ:
        assert r["lo"] <= r["win_share"] <= r["hi"] and r["chance"] == pytest.approx(1 / 3)


def test_cli_writes_outputs_and_resumes(tmp_path: Path) -> None:
    args = ["--candidates", "net;mcts:2,c=1.0", "--baselines", "heuristic", "--players", "3",
            "--games", "2", "--workers", "1", "--net", NET, "--out", str(tmp_path),
            "--quiet"]  # fmt: skip
    lines: list[str] = []
    assert main(args, out=lines.append) == 0
    assert "6 games to play" in lines[0]
    rows = (tmp_path / "games.jsonl").read_text().splitlines()
    assert len(rows) == 6
    data = json.loads((tmp_path / "summary.json").read_text())
    assert [r["candidate"] for r in data["summary"]] == ["net", "mcts:2,c=1.0"]
    assert (tmp_path / "summary.csv").read_text().startswith("candidate,baseline,players")
    lines.clear()
    assert main(args, out=lines.append) == 0
    assert "resuming: 6 games" in lines[0] and "0 games to play" in lines[1]
    assert len((tmp_path / "games.jsonl").read_text().splitlines()) == 6


def test_cli_rejects_bad_input(tmp_path: Path) -> None:
    base = ["--net", NET, "--out", str(tmp_path), "--quiet"]
    assert main(["--candidates", "bogus", *base]) == 2
    assert main(["--baselines", "nope", *base]) == 2
    assert main(["--players", "6", *base]) == 2
    assert main(["--net", str(tmp_path / "missing.npz"), "--out", str(tmp_path)]) == 2
