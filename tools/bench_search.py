"""Throughput spike for Gumbel AlphaZero (Phase 4 unit 8f): ``uv run python -m tools.bench_search``.

Measures, on this machine and per process: engine costs (``state.clone``, ``rules.apply``,
``rules.legal_actions``, ``encode``), batched net forward cost (numpy with one BLAS thread, and
torch on the GPU if available) at several batch sizes, and simulations/s, decisions/s and games/h
for self-play driven by a minimal Gumbel search at several simulation budgets, single process
and across N worker processes (numpy inference in each worker).

It is a spike, not the trainer. The search (``GumbelSearch``) is the smallest honest version of
Danihelka et al. 2022 adapted to N players: Gumbel top-m + sequential halving at the root,
deterministic "improved policy minus visit share" selection below the root, completed Q-values
with the mixed value estimate for unvisited actions, per-node min-max rescaling, and a per-seat
value vector backed up unchanged (each node's mover reads its own entry). Tree nodes hold the
immutable engine ``Core`` (``bus.rules.apply`` returns a new one), so search never clones a
pyspiel state; forced moves are applied while expanding, as in ``train/env.py``. Games are
batched across a worker: every live game contributes one leaf per wave and one forward serves
them all. No transposition table, no reanalyse, no training.

Options: ``--seconds`` per self-play cell (default 8), ``--sims`` list, ``--workers`` list for the
multi-process cells and ``--worker-sims`` their budgets, ``--net`` npz (default
``models/ppo-long.npz``), ``--players``, ``--quick``.
"""

# pyright: basic, reportMissingImports=false
from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

from bus import rules  # noqa: E402
from bus.actions import NUM_DISTINCT_ACTIONS  # noqa: E402
from train.features import encode  # noqa: E402

Array = np.ndarray[Any, Any]
# (obs (B, OBS), mask (B, A)) -> (masked logits (B, A), ego values (B, 5))
Forward = Callable[[Array, Array], tuple[Array, Array]]
C_VISIT = 50.0  # mctx qtransform_completed_by_mix_value defaults
C_SCALE = 0.1


# ------------------------------------------------------------------ search


def considered_visits(m: int, n: int) -> tuple[int, ...]:
    """Sequential-halving root visit schedule (mctx ``get_sequence_of_considered_visits``)."""
    if m <= 1:
        return tuple(range(n))
    log2max = math.ceil(math.log2(m))
    seq: list[int] = []
    visits = [0] * m
    k = m
    while len(seq) < n:
        extra = max(1, int(n / (log2max * k)))
        for _ in range(extra):
            seq.extend(visits[:k])
            for i in range(k):
                visits[i] += 1
        k = max(2, k // 2)
    return tuple(seq[:n])


def skip_forced(core: Any) -> tuple[Any, list[int]]:
    """Apply single-legal-action moves; return (decision or terminal core, its legal actions)."""
    while core.phase != rules.PHASE_END:
        legal = rules.legal_actions(core)
        if len(legal) != 1:
            return core, legal
        core = rules.apply(core, legal[0])
    return core, []


class Node:
    __slots__ = (
        "core", "mover", "actions", "logits", "prior", "value", "N", "W", "tot", "nmax",
        "children", "terminal",
    )  # fmt: skip

    def __init__(self, core: Any, actions: list[int], n: int) -> None:
        self.core = core
        self.terminal = core.phase == rules.PHASE_END
        self.mover = -1 if self.terminal else int(core.current)
        self.actions = np.asarray(actions, np.int64)
        k = len(actions)
        self.logits = np.zeros(k)
        self.prior = np.full(k, 1.0 / max(k, 1))
        self.tot = 0  # sum of N
        self.nmax = 0  # max of N
        self.value = np.zeros(n)  # absolute seats
        self.N = np.zeros(k, np.int64)
        self.W = np.zeros((k, n))
        self.children: list[Node | None] = [None] * k
        if self.terminal:
            self.value = np.asarray(rules.returns_for(core, "win_loss"), float)

    def set_eval(self, logits_full: Array, v_ego: Array) -> None:
        n = len(self.value)
        lg = logits_full[self.actions].astype(float)
        self.logits = lg - lg.max()
        e = np.exp(self.logits)
        self.prior = e / e.sum()
        self.value = np.array([v_ego[(s - self.mover) % n] for s in range(n)], float)


def _softmax(x: Array) -> Array:
    e = np.exp(x - x.max())
    return e / e.sum()


def completed_q(node: Node) -> Array:
    """Completed Q for the node's mover, min-max rescaled to [0, 1] (mixed value for unvisited)."""
    p = node.mover
    v = node.value[p]
    vis = node.N > 0
    tot = node.tot
    q = np.empty(len(node.N))
    if tot:
        qv = node.W[vis, p] / node.N[vis]
        pv = node.prior[vis]
        v_mix = (v + tot * float((pv * qv).sum() / pv.sum())) / (1 + tot)
        q[:] = v_mix
        q[vis] = qv
    else:
        q[:] = v
    lo, hi = q.min(), q.max()
    return (q - lo) / max(hi - lo, 1e-8)


def sigma(qhat: Array, node: Node) -> Array:
    return (C_VISIT + node.nmax) * C_SCALE * qhat


def improved_policy(node: Node) -> Array:
    """softmax(logits + sigma(completed Q)) over the node's legal actions (the policy target)."""
    return _softmax(node.logits + sigma(completed_q(node), node))


@dataclass
class GumbelSearch:
    """One search, driven by ``want`` (a core to evaluate, or None when done) and ``give``."""

    root_core: Any
    sims: int
    rng: np.random.Generator
    m: int = 16
    gumbel_scale: float = 1.0  # 0 = deterministic (evaluation)
    n: int = 0
    root: Node | None = None
    gumbel: Array = field(default_factory=lambda: np.zeros(0))
    schedule: tuple[int, ...] = ()
    done_sims: int = 0
    _path: list[tuple[Node, int]] = field(default_factory=list)
    _leaf: Node | None = None

    def __post_init__(self) -> None:
        self.n = int(self.root_core.players)
        _, legal = skip_forced(self.root_core)
        self.root = Node(self.root_core, legal, self.n)
        self._leaf = self.root  # the root itself is evaluated first
        k = len(legal)
        self.gumbel = self.gumbel_scale * self.rng.gumbel(size=k)
        self.schedule = considered_visits(min(self.m, k), self.sims)

    def _root_action(self, considered: int) -> int:
        root = self.root
        assert root is not None
        score = self.gumbel + root.logits + sigma(completed_q(root), root)
        score = np.where(considered == root.N, score, -np.inf)
        return int(np.argmax(score))

    def _interior_action(self, node: Node) -> int:
        pi = improved_policy(node)
        return int(np.argmax(pi - node.N / (1 + node.tot)))

    def want(self) -> Any:
        if self._leaf is not None:
            return self._leaf.core
        while self.done_sims < self.sims:
            root = self.root
            assert root is not None
            node, path = root, []
            a = self._root_action(self.schedule[self.done_sims])
            while True:
                path.append((node, a))
                child = node.children[a]
                if child is None:
                    core, legal = skip_forced(rules.apply(node.core, int(node.actions[a])))
                    child = Node(core, legal, self.n)
                    node.children[a] = child
                    if child.terminal:
                        self._backup(path, child.value)
                        break
                    self._path, self._leaf = path, child
                    return child.core
                if child.terminal:
                    self._backup(path, child.value)
                    break
                node = child
                a = self._interior_action(node)
        return None

    def give(self, logits_full: Array, v_ego: Array) -> None:
        leaf = self._leaf
        assert leaf is not None
        leaf.set_eval(logits_full, v_ego)
        self._leaf = None
        if leaf is not self.root:
            self._backup(self._path, leaf.value)

    def _backup(self, path: list[tuple[Node, int]], v: Array) -> None:
        for node, a in path:
            na = node.N[a] + 1
            node.N[a] = na
            node.tot += 1
            if na > node.nmax:
                node.nmax = int(na)
            node.W[a] += v
        self.done_sims += 1

    def result(self) -> tuple[int, Array, Array]:
        """(action id to play, improved-policy target over root legal actions, root actions)."""
        root = self.root
        assert root is not None
        a = self._root_action(int(root.N.max(initial=0)))
        return int(root.actions[a]), improved_policy(root), root.actions


# ------------------------------------------------------------------ backends


def numpy_forward(path: str) -> Forward:
    from bots.net_forward import NetWeights

    w = NetWeights.load(path)
    return w.forward


def torch_forward(path: str, device: str = "cuda") -> Forward:
    """The same weights in a torch ``BusNet`` (float32) on ``device``, copies to/from GPU timed."""
    import torch

    from bots.net_forward import NetWeights
    from train.model import BusNet

    w = NetWeights.load(path)
    hidden = tuple(int(b.shape[0]) for _, b in w.layers)
    net = BusNet(w.obs_size, hidden, w.n_actions, w.n_seats, separate_value=bool(w.v_layers))
    sd: dict[str, Any] = {}
    for i, (W, b) in enumerate(w.layers):
        sd[f"trunk.{i}.weight"], sd[f"trunk.{i}.bias"] = W.T, b
    for i, (W, b) in enumerate(w.v_layers):
        sd[f"v_trunk.{i}.weight"], sd[f"v_trunk.{i}.bias"] = W.T, b
    sd["pi.weight"], sd["pi.bias"] = w.pi[0].T, w.pi[1]
    sd["v.weight"], sd["v.bias"] = w.v[0].T, w.v[1]
    net.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()})
    net = net.to(device).eval()

    def fwd(obs: Array, mask: Array) -> tuple[Array, Array]:
        with torch.inference_mode():
            o = torch.from_numpy(obs).to(device, non_blocking=True)
            mk = torch.from_numpy(mask).to(device, non_blocking=True)
            lg, v = net(o, mk)
            return lg.cpu().numpy(), v.cpu().numpy()

    return fwd


# ------------------------------------------------------------------ measurements


def random_cores(players: int, count: int, seed: int) -> list[Any]:
    """Decision states sampled along random games (forced moves skipped)."""
    rng = np.random.default_rng(seed)
    out: list[Any] = []
    while len(out) < count:
        core, legal = skip_forced(rules.initial_core(players))
        while legal and len(out) < count:
            out.append(core)
            core, legal = skip_forced(rules.apply(core, int(rng.choice(legal))))
    return out


def net_cores(fwd: Forward, players: int, games: int, seed: int) -> tuple[list[Any], float]:
    """Decision states of ``games`` sampled-policy net self-play games, and decisions per game."""
    rng = np.random.default_rng(seed)
    live = [skip_forced(rules.initial_core(players)) for _ in range(games)]
    out: list[Any] = []
    while live:
        obs = np.stack([encode(c) for c, _ in live]).astype(np.float32)
        mask = np.zeros((len(live), NUM_DISTINCT_ACTIONS), np.bool_)
        for r, (_c, legal) in enumerate(live):
            mask[r, legal] = True
        lg, _v = fwd(obs, mask)
        nxt = []
        for r, (c, legal) in enumerate(live):
            out.append(c)
            z = lg[r, legal] - lg[r, legal].max()
            p = np.exp(z) / np.exp(z).sum()
            c2, l2 = skip_forced(rules.apply(c, int(legal[rng.choice(len(legal), p=p)])))
            if l2:
                nxt.append((c2, l2))
        live = nxt
    return out, len(out) / games


def bench_engine(players: int, count: int = 400) -> dict[str, float]:
    """Mean microseconds per call over decision states of random games."""
    import pyspiel

    from bus.game import BusState

    cores = random_cores(players, count, seed=1)
    clock = time.perf_counter
    res: dict[str, float] = {}
    t = clock()
    legals = [rules.legal_actions(c) for c in cores]
    res["legal_actions"] = (clock() - t) / count * 1e6
    rng = np.random.default_rng(2)
    acts = [int(rng.choice(lg)) for lg in legals]
    t = clock()
    for c, a in zip(cores, acts, strict=True):
        rules.apply(c, a)
    res["apply"] = (clock() - t) / count * 1e6
    t = clock()
    forced = 0
    for c, a in zip(cores, acts, strict=True):
        nxt = rules.apply(c, a)
        while nxt.phase != rules.PHASE_END:
            lg = rules.legal_actions(nxt)
            if len(lg) != 1:
                break
            nxt = rules.apply(nxt, lg[0])
            forced += 1
    res["expand (apply + forced skip + legal)"] = (clock() - t) / count * 1e6
    res["forced moves per decision"] = forced / count
    t = clock()
    for c in cores:
        encode(c)
    res["encode"] = (clock() - t) / count * 1e6
    game = pyspiel.load_game("python_bus", {"players": players})
    st = game.new_initial_state()
    assert isinstance(st, BusState)
    st.core = cores[len(cores) // 2]
    t = clock()
    for _ in range(count):
        st.clone()
    res["pyspiel state.clone (reference)"] = (clock() - t) / count * 1e6
    return res


def bench_forward(fwd: Forward, players: int, batches: tuple[int, ...]) -> dict[int, float]:
    """Microseconds per batched forward call, by batch size."""
    cores = random_cores(players, max(batches), seed=3)
    obs_all = np.stack([encode(c) for c in cores]).astype(np.float32)
    mask_all = np.zeros((len(cores), NUM_DISTINCT_ACTIONS), np.bool_)
    for i, c in enumerate(cores):
        mask_all[i, rules.legal_actions(c)] = True
    out: dict[int, float] = {}
    for b in batches:
        obs, mask = obs_all[:b], mask_all[:b]
        fwd(obs, mask)  # warm-up
        reps = max(5, int(2000 / b))
        t = time.perf_counter()
        for _ in range(reps):
            fwd(obs, mask)
        out[b] = (time.perf_counter() - t) / reps * 1e6
    return out


def selfplay(
    fwd: Forward, players: int, sims: int, batch: int, seconds: float, seed: int = 0, m: int = 16,
    starts: tuple[list[Any], float] | None = None,
) -> dict[str, float]:  # fmt: skip
    """Run ``batch`` concurrent self-play games under search for ``seconds``; return rates.

    Slots start from decision states drawn uniformly from net self-play games (``starts``, made
    here if None), so a short run sees the whole game's mix of phases, not only the setup.
    """
    rng = np.random.default_rng(seed)
    clock = time.perf_counter
    if starts is None:
        starts = net_cores(fwd, players, 16, seed + 10_000)
    pool, dpg = starts

    def new_search(core: Any) -> GumbelSearch:
        return GumbelSearch(core, sims, rng, m=m)

    cores: list[Any] = [pool[int(i)] for i in rng.integers(0, len(pool), batch)]
    searches = [new_search(c) for c in cores]
    t_fwd = 0.0
    sims_done = decisions = games = forwards = evals = 0
    start = clock()
    while clock() - start < seconds:
        reqs: list[tuple[int, Any]] = []
        for i, s in enumerate(searches):
            c = s.want()
            while c is None:
                sims_done += s.done_sims
                a, _pi, _acts = s.result()
                decisions += 1
                nxt, legal = skip_forced(rules.apply(cores[i], a))
                if not legal:
                    games += 1
                    nxt, legal = skip_forced(rules.initial_core(players))
                cores[i] = nxt
                searches[i] = s = new_search(nxt)
                c = s.want()
            reqs.append((i, c))
        obs = np.stack([encode(c) for _, c in reqs]).astype(np.float32)
        mask = np.zeros((len(reqs), NUM_DISTINCT_ACTIONS), np.bool_)
        for r, (i, _c) in enumerate(reqs):
            node = searches[i]._leaf
            assert node is not None
            mask[r, node.actions] = True
        t0 = clock()
        lg, v = fwd(obs, mask)
        t_fwd += clock() - t0
        forwards += 1
        evals += len(reqs)
        for r, (i, _c) in enumerate(reqs):
            searches[i].give(lg[r], v[r])
    wall = clock() - start
    sims_done += sum(s.done_sims for s in searches)
    return {
        "sims/s": sims_done / wall,
        "evals/s": evals / wall,
        # steady state: one decision per ``sims`` simulations (completed decisions undercount
        # short runs with large batches, where a whole wave of searches may still be open)
        "decisions/s": sims_done / wall / sims,
        "games/h": sims_done / wall / sims * 3600 / dpg,
        "fwd share": t_fwd / wall,
        "mean batch": evals / max(forwards, 1),
        "finished games": float(games),
        "decisions/game": dpg,
    }


def _worker(args: tuple[str, int, int, int, float, int]) -> dict[str, float]:
    path, players, sims, batch, seconds, seed = args
    r = selfplay(numpy_forward(path), players, sims, batch, seconds, seed)
    r["rss MB"] = peak_rss_mb()
    return r


def peak_rss_mb() -> float:
    """Peak resident set of this process (VmHWM; ``ru_maxrss`` survives exec in spawned workers)."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def multi_worker(
    path: str, players: int, sims: int, batch: int, seconds: float, workers: int
) -> dict[str, float]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        rs = pool.map(_worker, [(path, players, sims, batch, seconds, s) for s in range(workers)])
    keys = ("sims/s", "decisions/s", "games/h")
    out = {k: sum(r[k] for r in rs) for k in keys}
    out["max rss MB"] = max(r["rss MB"] for r in rs)
    out["fwd share"] = sum(r["fwd share"] for r in rs) / workers
    return out


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="models/ppo-long.npz")
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--sims", default="16,32,64,128")
    ap.add_argument("--workers", default="4,8", help="multi-process cells ('' = none)")
    ap.add_argument("--worker-sims", default="16,64")
    ap.add_argument("--quick", action="store_true", help="tiny run (smoke test)")
    a = ap.parse_args()
    sims_list = [int(x) for x in a.sims.split(",")]
    workers_list = [int(x) for x in a.workers.split(",") if x]
    secs = 1.0 if a.quick else a.seconds
    batches = (1, 16, 64, 256)

    print(f"# players={a.players} net={a.net}")
    print("## engine (us per call, mean over decision states of random games)")
    for k, v in bench_engine(a.players, 100 if a.quick else 400).items():
        print(f"  {k}: {v:.2f}")

    np_fwd = numpy_forward(a.net)
    print("## forward (us per call / us per obs)")
    for b, us in bench_forward(np_fwd, a.players, batches).items():
        print(f"  numpy 1-thread  B={b:<4} {us:9.0f} {us / b:8.1f}")
    t_fwd: Forward | None = None
    try:
        import torch

        if torch.cuda.is_available():
            t_fwd = torch_forward(a.net)
            for b, us in bench_forward(t_fwd, a.players, batches).items():
                print(f"  torch cuda      B={b:<4} {us:9.0f} {us / b:8.1f}")
    except ImportError:
        print("  torch not installed: GPU rows skipped")

    starts = net_cores(np_fwd, a.players, 4 if a.quick else 24, 10_000)
    print(f"## self-play under Gumbel search, one process ({starts[1]:.1f} decisions/game)")
    cols = ("sims/s", "evals/s", "decisions/s", "games/h", "fwd share", "mean batch")
    print("  backend         B    sims " + " ".join(f"{c:>11}" for c in cols))
    cells: list[tuple[str, Forward, int]] = [("numpy", np_fwd, 1), ("numpy", np_fwd, 32)]
    if t_fwd is not None:
        cells += [("torch-cuda", t_fwd, 64), ("torch-cuda", t_fwd, 256)]
    for name, f, b in cells:
        for s in sims_list:
            r = selfplay(f, a.players, s, b, secs, starts=starts)
            print(f"  {name:<14} {b:<4} {s:5d} " + " ".join(f"{r[c]:11.2f}" for c in cols))

    rss = peak_rss_mb()
    print(f"  main process max RSS: {rss:.0f} MB (includes torch/CUDA if loaded)")
    if workers_list and not a.quick:
        print("## self-play, N worker processes, numpy B=32 each")
        for w in workers_list:
            for s in [int(x) for x in a.worker_sims.split(",")]:
                r = multi_worker(a.net, a.players, s, 32, secs, w)
                print(
                    f"  workers={w:<3} sims={s:<4} "
                    + " ".join(f"{k}={v:.1f}" for k, v in r.items())
                )


if __name__ == "__main__":
    main()
