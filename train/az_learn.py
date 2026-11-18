# pyright: basic, reportMissingImports=false
"""Gumbel AlphaZero learner (Phase 4 unit 8h). Needs torch.

Loss per minibatch (``docs/GUMBEL_AZ.md`` §3):

- policy: cross-entropy of the net's masked policy against the completed-Q improved policy
  (``pi``, zero outside the legal actions); skipped during the value-only warm-up;
- value: MSE of the 5-wide ego value head against the per-seat final R (``z``), present seats only
  (``vmask``); with ``value_target="zq"`` the target is (z + q) / 2, q = root search value;
- aux (optional): the unit 8e training-only heads (score, rank, winner);
- weight decay through AdamW (decoupled; parameters without a gradient, e.g. the policy side during
  the warm-up, are left alone).

Minibatches are drawn uniformly from the replay with a numpy generator the caller seeds per
generation, so a run is reproducible and resumable without saving sampler state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from train.az_replay import Replay
from train.model import BusNet
from train.ppo import aux_losses, ev_by_quartile, explained_variance

Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class LearnConfig:
    batch: int = 1024
    vf_coef: float = 1.0
    aux_coef: float = 0.25
    max_grad_norm: float = 1.0
    value_target: str = "z"  # "z" or "zq"


def to_tensors(b: dict[str, Array], device: torch.device) -> dict[str, torch.Tensor]:
    out = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}
    out["arank"] = out["arank"].long()
    return out


def value_target(d: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
    return d["z"] if kind == "z" else 0.5 * (d["z"] + d["q"])


def losses(
    model: BusNet, d: dict[str, torch.Tensor], cfg: LearnConfig, value_only: bool
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, v = model(d["obs"], d["mask"])
    vm = d["vmask"]
    v_loss = (((v - value_target(d, cfg.value_target)) ** 2) * vm).sum() / vm.sum().clamp(min=1.0)
    logp = torch.log_softmax(logits, dim=-1)
    pi_loss = -(d["pi"] * torch.where(d["mask"], logp, torch.zeros_like(logp))).sum(-1).mean()
    loss = cfg.vf_coef * v_loss
    if not value_only:
        loss = loss + pi_loss
    info = {"loss_pi": pi_loss.item(), "loss_v": v_loss.item(), "loss_aux": 0.0}
    if cfg.aux_coef > 0 and model.has_aux:
        idx = torch.arange(len(vm), device=vm.device)
        a_s, a_r, a_w = aux_losses(model, d, idx)
        aux = a_s + a_r + a_w
        loss = loss + cfg.aux_coef * aux
        info["loss_aux"] = aux.item()
    return loss, info


def train_steps(
    model: BusNet, opt: torch.optim.Optimizer, replay: Replay, steps: int,
    rng: np.random.Generator, cfg: LearnConfig, device: torch.device, value_only: bool,
) -> dict[str, float]:  # fmt: skip
    """``steps`` minibatch updates on uniform replay samples; mean losses."""
    acc = {"loss_pi": 0.0, "loss_v": 0.0, "loss_aux": 0.0}
    if steps <= 0 or replay.size == 0:
        return acc
    model.train()
    params = [p for p in model.parameters()]
    for _ in range(steps):
        idx = rng.integers(0, replay.size, size=min(cfg.batch, replay.size))
        d = to_tensors(replay.batch(idx), device)
        loss, info = losses(model, d, cfg, value_only)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        for k in acc:
            acc[k] += info[k]
    return {k: v / steps for k, v in acc.items()}


@torch.no_grad()
def diagnostics(
    model: BusNet, parts: list[dict[str, Array]], device: torch.device, max_samples: int = 8192,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:  # fmt: skip
    """Value EV (own seat, overall and by game quartile) and policy CE on fresh sample chunks.

    Called on a generation's samples before the learner sees them, so they are held out.
    """
    rep = Replay(capacity=1 << 62)
    for p in parts:
        rep.add(p)
    n = rep.size
    if n == 0:
        return {}
    idx = np.arange(n)
    if n > max_samples:
        idx = np.sort((rng or np.random.default_rng(0)).choice(n, max_samples, replace=False))
    model.eval()
    preds, zs, progs, ces = [], [], [], []
    for a in range(0, len(idx), 2048):
        b = rep.batch(idx[a : a + 2048])
        d = to_tensors(b, device)
        logits, v = model(d["obs"], d["mask"])
        logp = torch.log_softmax(logits, dim=-1)
        ce = -(d["pi"] * torch.where(d["mask"], logp, torch.zeros_like(logp))).sum(-1)
        preds.append(v[:, 0].cpu().numpy())
        zs.append(b["z"][:, 0])
        progs.append(b["prog"])
        ces.append(ce.cpu().numpy())
    pred, z, prog = np.concatenate(preds), np.concatenate(zs), np.concatenate(progs)
    q = ev_by_quartile(pred, z, prog)
    return {
        "ev": explained_variance(pred, z), "ev_q1": q[0], "ev_q2": q[1], "ev_q3": q[2],
        "ev_q4": q[3], "policy_ce": float(np.concatenate(ces).mean()),
    }  # fmt: skip
