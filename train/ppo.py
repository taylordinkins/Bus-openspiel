# pyright: basic, reportMissingImports=false
"""PPO learner pieces (Phase 4 unit 3): schedules, weight conversion, clipped-loss update.

The loss block is copy-adapted from OpenSpiel's ``python/pytorch/ppo.py`` (clipped surrogate,
masked MSE value loss, entropy bonus), changed for a masked, multi-seat value head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from bots.net_forward import NetWeights
from train.model import BusNet


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    lr_decay: bool = False  # linear to 10% of lr over the planned updates
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float | None = None  # linear schedule target; None = constant
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 8
    normalize_adv: bool = True
    target_kl: float | None = None  # stop epochs early when exceeded (None = off)


def linear(start: float, end: float, step: int, total: int) -> float:
    if total <= 0:
        return start
    return start + (end - start) * min(1.0, max(0.0, step / total))


def cosine(start: float, end: float, step: int, total: int) -> float:
    """Half-cosine from ``start`` to ``end`` over ``total`` steps (flat after)."""
    if total <= 0:
        return start
    f = min(1.0, max(0.0, step / total))
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * f))


def lr_at(
    step: int, lr: float, decay: bool, schedule: str, total: int, final_frac: float = 0.1
) -> float:
    """Learning rate for update ``step`` (0-based): constant, or ``linear``/``cosine`` down to
    ``final_frac * lr`` over the fixed ``total``-update horizon."""
    if not decay:
        return lr
    fn = cosine if schedule == "cosine" else linear
    return fn(lr, lr * final_frac, step, total)


def ent_at(
    step: int, start: float, final: float | None, total: int, floor_at: float = 1.0
) -> float:
    """Entropy coefficient: ``start`` falling linearly to the constant floor ``final`` at
    ``floor_at * total`` updates, then constant. ``final=None`` keeps ``start``."""
    if final is None:
        return start
    return linear(start, final, step, int(total * floor_at))


def mix_at(update: int, mix0: float, anneal_updates: int) -> float:
    """Reward mix for ``update`` (0-based): ``mix0`` rising linearly to 1.0 over the anneal.

    ``anneal_updates <= 0`` disables annealing (constant ``mix0``).
    """
    if anneal_updates <= 0:
        return mix0
    return linear(mix0, 1.0, update, anneal_updates)


def net_to_weights(model: BusNet) -> NetWeights:
    """Numpy copy of the model for the rollout workers (same layout as the npz export)."""

    def lin(layer: Any) -> tuple[Any, Any]:
        w = layer.weight.detach().cpu().numpy().T.astype(np.float32).copy()
        return w, layer.bias.detach().cpu().numpy().astype(np.float32).copy()

    return NetWeights(
        tuple(lin(layer) for layer in model.trunk), lin(model.pi), lin(model.v), model.ACTIVATION,
        model.obs_size, model.n_actions, model.n_seats, {},
        tuple(lin(layer) for layer in model.v_trunk),
    )  # fmt: skip


def explained_variance(pred: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> float:
    var = float(np.var(target))
    return float("nan") if var < 1e-12 else float(1.0 - np.var(target - pred) / var)


def ev_by_quartile(
    pred: np.ndarray[Any, Any], target: np.ndarray[Any, Any], prog: np.ndarray[Any, Any]
) -> list[float]:
    """Explained variance of the value head on samples in each quarter of game progress
    (``prog`` in [0, 1): decision index within the seat's trajectory / its length); NaN if a
    quarter has no samples or no target variance."""
    out: list[float] = []
    for q in range(4):
        m = (prog >= q / 4) & (prog < (q + 1) / 4)
        out.append(explained_variance(pred[m], target[m]) if m.any() else float("nan"))
    return out


def ppo_update(
    model: BusNet, opt: torch.optim.Optimizer, data: dict[str, torch.Tensor], cfg: PPOConfig,
    ent_coef: float, gen: torch.Generator,
) -> dict[str, float]:  # fmt: skip
    """Several epochs of minibatch PPO over ``data`` (all tensors already on the model device).

    data: obs, mask, action, logp, adv, vtarget, vmask. Returns mean diagnostics.
    """
    n = data["obs"].shape[0]
    adv_all = data["adv"]
    if cfg.normalize_adv:
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
    mb = max(1, n // cfg.minibatches)
    acc = dict.fromkeys(("entropy", "approx_kl", "clip_frac", "pg_loss", "value_loss"), 0.0)
    count = 0
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen, device="cpu").to(data["obs"].device)
        for start in range(0, n, mb):
            idx = perm[start : start + mb]
            _, newlogp, ent, values = model.act(
                data["obs"][idx], data["mask"][idx], data["action"][idx]
            )
            logratio = newlogp - data["logp"][idx]
            ratio = logratio.exp()
            a = adv_all[idx]
            pg = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
            vm = data["vmask"][idx]
            vl = (((values - data["vtarget"][idx]) ** 2) * vm).sum() / vm.sum().clamp(min=1.0)
            ent_m = ent.mean()
            loss = pg + cfg.vf_coef * vl - ent_coef * ent_m
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - logratio).mean().item()
                acc["entropy"] += ent_m.item()
                acc["approx_kl"] += kl
                acc["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                acc["pg_loss"] += pg.item()
                acc["value_loss"] += vl.item()
                count += 1
            if cfg.target_kl is not None and kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    return {k: v / max(count, 1) for k, v in acc.items()}
