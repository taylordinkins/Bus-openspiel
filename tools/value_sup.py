# pyright: basic, reportMissingImports=false
"""Supervised value training on K-mean labels (Phase 4 interaction work). Needs torch.

A cheap test of the value directions without self-play: does a value trunk trained on the
conditional mean E[R | s] (estimated by K continuations, ``tools.noise_ceiling --save-obs``) get
closer to the noise ceiling with observation v3 than with v2, and with the leader-relative target
than with the mean-opponent one?

Protocol per (features, target) cell:

1. **Labels:** one or more ``positions.npz`` from ``tools.noise_ceiling --save-obs`` (positions
   of ``ppo-long`` self-play, K continuations each). The target for each present seat is the
   K-mean of the chosen quantity (``R`` = 0.5 wl + 0.5 mean-opponent margin, ``R_leader``,
   ``R_rank``, ``wl``), rotated to ego order (index 0 = the mover) like the value head.
2. **Split** by sampling game: games with ``game % 5 == 0`` are held out (``--holdout`` file
   replaces this with a separate labels file).
3. **Net:** ``--init`` (an ``.npz``; default ``models/ppo-long.npz``), widened to v3 with zero
   columns when ``--features`` asks for v3 (``NetWeights.widened``: the start net equals the
   v2 one), or a fresh value trunk with ``--reset-value``. Only the value trunk and value head
   train (policy trunk and head frozen); MSE on all present seats, AdamW, early stopping on the
   held-out R² of the ego seat.
4. **Metrics** on held-out positions, ego seat, by game quartile and overall
   (``tools.noise_ceiling.net_ev`` / ``ceiling``): EV against the K single outcomes, R² against
   the K-mean (finite-K corrected), and the ceiling EV_max for that quantity; for the start net
   and the best epoch.

    uv run --group train python -m tools.value_sup --labels runs/value_labels/a/positions.npz \\
        --features v2,v3 --targets R,R_leader --out runs/value_sup/a

Writes ``summary.json`` / ``summary.md`` and, per cell, ``<features>_<target>.npz`` (the
trained net, loadable as a v2/v3 ``NetWeights``).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bots.net_forward import NetWeights
from tools.noise_ceiling import ceiling, net_ev
from train.features import FEATURES_V2, FEATURES_V3, MAX_SEATS
from train.league import save_weights
from train.model import BusNet
from train.ppo import net_to_weights

Array = np.ndarray[Any, Any]
FEATURES = {"v2": FEATURES_V2, "v3": FEATURES_V3}
TARGETS = ("R", "R_leader", "R_rank", "wl")
QUARTERS = ("q1", "q2", "q3", "q4", "all")


# ------------------------------------------------------------------ data


def load_labels(paths: Sequence[str]) -> dict[str, Array]:
    """Concatenate label files; game ids are made unique across files."""
    parts: list[dict[str, Array]] = []
    base = 0
    for p in paths:
        with np.load(p) as z:
            if "obs_v2" not in z.files:
                raise SystemExit(f"{p} has no observations: rerun noise_ceiling with --save-obs")
            d = {k: z[k] for k in z.files if not k.startswith("pred_")}
        d["game"] = d["game"] + base
        base = int(d["game"].max()) + 1
        parts.append(d)
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


def ego_outcomes(data: dict[str, Array], target: str) -> tuple[Array, Array]:
    """(P, 5, K) single outcomes and (P, 5) seat mask, in ego order (index 0 = the mover)."""
    r = data[target]
    p, _, k = r.shape
    out = np.zeros((p, MAX_SEATS, k))
    mask = np.zeros((p, MAX_SEATS))
    for i in range(p):
        n, m = int(data["players"][i]), int(data["mover"][i])
        for j in range(n):
            out[i, j] = r[i, (m + j) % n]
        mask[i, :n] = 1.0
    return out, mask


def split(data: dict[str, Array], holdout: dict[str, Array] | None) -> tuple[Array, Array]:
    """Index arrays (train, held-out) into ``data`` (held-out empty if ``holdout`` is given)."""
    idx = np.arange(len(data["game"]))
    if holdout is not None:
        return idx, idx[:0]
    held = data["game"] % 5 == 0
    return idx[~held], idx[held]


# ------------------------------------------------------------------ net


def weights_to_net(w: NetWeights) -> BusNet:
    """A torch ``BusNet`` holding the numpy weights ``w`` (inverse of ``net_to_weights``)."""
    if w.activation != BusNet.ACTIVATION:
        raise SystemExit(f"BusNet is {BusNet.ACTIVATION}; the net uses {w.activation}")
    hidden = tuple(int(x[0].shape[1]) for x in w.layers)
    net = BusNet(obs_size=w.obs_size, hidden=hidden, separate_value=bool(w.v_layers))
    sd = net.state_dict()

    def put(prefix: str, layer: tuple[Array, Array]) -> None:
        sd[prefix + ".weight"] = torch.from_numpy(np.ascontiguousarray(layer[0].T)).float()
        sd[prefix + ".bias"] = torch.from_numpy(np.asarray(layer[1])).float()

    for i, layer in enumerate(w.layers):
        put(f"trunk.{i}", layer)
    for i, layer in enumerate(w.v_layers):
        put(f"v_trunk.{i}", layer)
    put("pi", w.pi)
    put("v", w.v)
    net.load_state_dict(sd)
    return net


def value_params(net: BusNet) -> list[torch.nn.Parameter]:
    if not net.separate_value:
        raise SystemExit("value-only training needs a separate value trunk (format 2 nets)")
    return [*net.v_trunk.parameters(), *net.v.parameters()]


def reset_value(net: BusNet, seed: int) -> None:
    from train.model import layer_init

    torch.manual_seed(seed)
    for layer in net.v_trunk:
        assert isinstance(layer, torch.nn.Linear)
        layer_init(layer)
    layer_init(net.v, std=1.0)


@torch.no_grad()
def predict(net: BusNet, obs: torch.Tensor, batch: int = 4096) -> Array:
    """(P, 5) value head outputs (the policy mask is irrelevant for the value)."""
    net.eval()
    mask = torch.ones((1, net.n_actions), dtype=torch.bool, device=obs.device)
    out = []
    for a in range(0, len(obs), batch):
        x = obs[a : a + batch]
        _, v = net(x, mask.expand(len(x), -1))
        out.append(v.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, MAX_SEATS))


def metrics(pred_ego: Array, r_ego: Array, quartile: Array) -> dict[str, dict[str, float]]:
    """Per quartile: net EV single / R² vs K-mean (``net_ev``) and the ceiling EV_max."""
    out: dict[str, dict[str, float]] = {}
    for q in QUARTERS:
        sel = np.ones(len(quartile), bool) if q == "all" else quartile == int(q[1]) - 1
        if sel.sum() < 3:
            out[q] = {"units": int(sel.sum())}
            continue
        e = net_ev(pred_ego[sel], r_ego[sel])
        out[q] = {"units": int(sel.sum()), **e, "ev_max": ceiling(r_ego[sel])["ev_max"]}
    return out


# ------------------------------------------------------------------ one cell


def train_cell(
    data: dict[str, Array], held: dict[str, Array], tr: Array, ho: Array, feat: str, target: str,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:  # fmt: skip
    t0 = time.perf_counter()
    w = NetWeights.load(args.init)
    if FEATURES[feat] != w.features_version:
        if (w.features_version, FEATURES[feat]) != (FEATURES_V2, FEATURES_V3):
            raise SystemExit(f"cannot start a {feat} net from features v{w.features_version}")
        w = w.widened(FEATURES[feat])
    net = weights_to_net(w).to(device)
    if args.reset_value:
        reset_value(net, args.seed)
        net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    params = value_params(net)
    for p in params:
        p.requires_grad_(True)
    frozen = {k: v.clone() for k, v in net.state_dict().items() if not k.startswith(("v_", "v."))}

    key = f"obs_{feat}"
    r_tr, m_tr = ego_outcomes(data, target)
    r_ho, _ = ego_outcomes(held, target)
    x_tr = torch.from_numpy(data[key][tr]).float().to(device)
    y_tr = torch.from_numpy(r_tr[tr].mean(axis=2)).float().to(device)
    mk_tr = torch.from_numpy(m_tr[tr]).float().to(device)
    x_ho = torch.from_numpy(held[key][ho]).float().to(device)
    rh, qh = r_ho[ho, 0], held["quartile"][ho]

    def evaluate() -> dict[str, dict[str, float]]:
        return metrics(predict(net, x_ho)[:, 0], rh, qh)

    start = evaluate()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best, best_ep, best_state, bad = start, 0, None, 0
    history = []
    full = torch.ones((1, net.n_actions), dtype=torch.bool, device=device)
    for ep in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(x_tr))
        loss_sum = 0.0
        for a in range(0, len(order), args.batch):
            b = torch.from_numpy(order[a : a + args.batch]).to(device)
            _, v = net(x_tr[b], full.expand(len(b), -1))
            loss = (((v - y_tr[b]) ** 2) * mk_tr[b]).sum() / mk_tr[b].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(b)
        m = evaluate()
        history.append(
            {
                "epoch": ep,
                "loss": loss_sum / max(len(order), 1),
                "r2_all": m["all"].get("r2_mean"),
                "ev_all": m["all"].get("ev_single"),
            }
        )
        if m["all"].get("r2_mean", -np.inf) > best["all"].get("r2_mean", -np.inf):
            best, best_ep, bad = m, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    for k, v in net.state_dict().items():  # the policy side never moved
        if k in frozen:
            assert torch.equal(v, frozen[k]), k
    out_w = net_to_weights(net.cpu())
    extra = {"features_version": FEATURES[feat], "value_sup": target, "epoch": best_ep}
    save_weights(out_w, Path(args.out) / f"{feat}_{target}.npz", extra)
    return {
        "features": feat, "target": target, "train_positions": int(len(tr)),
        "heldout_positions": int(len(ho)), "best_epoch": best_ep, "start": start, "best": best,
        "history": history, "seconds": round(time.perf_counter() - t0, 1),
    }  # fmt: skip


# ------------------------------------------------------------------ report


def _f(x: Any) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:+.3f}" if x < 0 else f"{x:.3f}"


def markdown(cells: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# Supervised value training on K-mean labels", "", "```", json.dumps(meta), "```", ""]
    lines += [
        "Held-out positions, ego seat. EV = explained variance against single outcomes; R² is "
        "against the K-mean (finite-K corrected); ceiling = EV_max of that quantity.", "",
    ]  # fmt: skip
    head = "| features | target | epoch | metric |" + "".join(f" {q} |" for q in QUARTERS)
    sep = "|---|---|---|---|" + "---|" * len(QUARTERS)
    lines += [head, sep]
    for c in cells:
        for label, src, key in (
            ("start EV", "start", "ev_single"), ("best EV", "best", "ev_single"),
            ("start R²", "start", "r2_mean"), ("best R²", "best", "r2_mean"),
            ("ceiling EV", "best", "ev_max"),
        ):  # fmt: skip
            row = f"| {c['features']} | {c['target']} | {c['best_epoch']} | {label} |"
            row += "".join(f" {_f(c[src][q].get(key))} |" for q in QUARTERS)
            lines.append(row)
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--labels", required=True, help="comma list of noise_ceiling positions.npz")
    ap.add_argument("--holdout", default=None, help="separate held-out labels (comma list)")
    ap.add_argument("--features", default="v2,v3", help="comma list of v2, v3")
    ap.add_argument("--targets", default="R,R_leader", help=f"comma list of {', '.join(TARGETS)}")
    ap.add_argument("--init", default="models/ppo-long.npz", help="start net (.npz)")
    ap.add_argument("--reset-value", action="store_true", help="fresh value trunk and head")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8, help="epochs without held-out gain")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--out", default="runs/value_sup/default")
    args = ap.parse_args(argv)
    for f in args.features.split(","):
        if f not in FEATURES:
            ap.error(f"unknown features {f!r}")
    for t in args.targets.split(","):
        if t not in TARGETS:
            ap.error(f"unknown target {t!r}")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    data = load_labels(args.labels.split(","))
    held = load_labels(args.holdout.split(",")) if args.holdout else None
    tr, ho = split(data, held)
    if held is None:
        held = data
    else:
        ho = np.arange(len(held["game"]))
    if len(ho) == 0 or len(tr) == 0:
        raise SystemExit("need both training and held-out positions")
    cells = [
        train_cell(data, held, tr, ho, f, t, args, device)
        for f in args.features.split(",") for t in args.targets.split(",")
    ]  # fmt: skip
    meta = {
        "labels": args.labels, "holdout": args.holdout, "init": args.init,
        "reset_value": args.reset_value, "k": int(data["R"].shape[2]), "epochs": args.epochs,
        "lr": args.lr, "batch": args.batch, "seed": args.seed, "device": dev,
    }  # fmt: skip
    res = {"meta": meta, "cells": cells}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    (out / "summary.md").write_text(markdown(cells, meta))
    print(markdown(cells, meta), flush=True)
    return res


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
