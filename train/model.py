# pyright: basic, reportMissingImports=false
"""BusNet: policy/value MLP for PPO self-play (Phase 4 unit 2). Needs torch.

Shape: MLP trunk (tanh, as in OpenSpiel's PPO and robust for PPO value learning), a policy
head giving one logit per action id, and an N-player value head giving one value per ego-ordered
seat (seat 0 = player to move, padded to ``n_seats`` = 5; the caller masks absent seats), tanh-
bounded to [-1, 1] because shaped returns lie there. Orthogonal init after OpenSpiel's
``open_spiel/python/pytorch/ppo.py`` (copy-adapted: ``layer_init``, masked categorical).

``separate_value=True`` (default from unit 8b, export format 2) gives the value head its own trunk
of the same shape, so value-loss gradients (``vf_coef`` 1.0) cannot disturb the policy features;
``False`` keeps the original shared trunk (export format 1 layout).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from bus.actions import NUM_DISTINCT_ACTIONS
from train.features import MAX_SEATS, OBS_SIZE

MASK_FILL = -1e8


def layer_init(
    layer: nn.Linear, std: float = float(np.sqrt(2)), bias_const: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), logits, torch.full_like(logits, MASK_FILL))


class BusNet(nn.Module):
    ACTIVATION = "tanh"

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        hidden: tuple[int, ...] = (512, 512),
        n_actions: int = NUM_DISTINCT_ACTIONS,
        n_seats: int = MAX_SEATS,
        separate_value: bool = True,
        aux: bool = False,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.obs_size, self.hidden, self.n_actions, self.n_seats = (
            obs_size, tuple(hidden), n_actions, n_seats,
        )  # fmt: skip
        sizes = (obs_size, *self.hidden)
        self.trunk = nn.ModuleList(
            layer_init(nn.Linear(a, b)) for a, b in zip(sizes[:-1], sizes[1:], strict=True)
        )
        self.separate_value = separate_value
        self.v_trunk = (
            nn.ModuleList(
                layer_init(nn.Linear(a, b)) for a, b in zip(sizes[:-1], sizes[1:], strict=True)
            )
            if separate_value
            else nn.ModuleList()
        )
        self.pi = layer_init(nn.Linear(sizes[-1], n_actions), std=0.01)
        self.v = layer_init(nn.Linear(sizes[-1], n_seats), std=1.0)
        # Training-only auxiliary heads on the value trunk output (unit 8e); never exported.
        self.has_aux = aux
        self._vfeat: torch.Tensor | None = None
        if aux:
            self.aux_score = layer_init(nn.Linear(sizes[-1], n_seats), std=1.0)
            self.aux_rank = layer_init(nn.Linear(sizes[-1], n_seats * n_seats), std=1.0)
            self.aux_win = layer_init(nn.Linear(sizes[-1], n_seats), std=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(masked logits (B, n_actions), values (B, n_seats) in [-1, 1])."""
        x = obs
        for layer in self.trunk:
            x = torch.tanh(layer(x))
        xv = obs if self.separate_value else x
        if self.separate_value:
            for layer in self.v_trunk:
                xv = torch.tanh(layer(xv))
        if self.has_aux:
            self._vfeat = xv  # read by ``aux_outputs`` right after this call
        return mask_logits(self.pi(x), mask), torch.tanh(self.v(xv))

    def aux_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(score (B, S), rank logits (B, S, S), winner logits (B, S)) of the last ``forward``."""
        assert self.has_aux and self._vfeat is not None
        f = self._vfeat
        return (
            self.aux_score(f),
            self.aux_rank(f).view(-1, self.n_seats, self.n_seats),
            self.aux_win(f),
        )

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(action, log_prob, entropy, values); samples unless ``action`` is given or greedy."""
        logits, values = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if greedy else dist.sample()
        # Entropy over legal actions only (masked ones have ~0 probability anyway).
        p = dist.probs
        ent = -(p * torch.where(mask.bool(), dist.logits, torch.zeros_like(p))).sum(-1)
        logp = cast(torch.Tensor, dist.log_prob(action))
        return action, logp, ent, values


FIRST_LAYERS = ("trunk.0.weight", "v_trunk.0.weight")


def widen_model(src: BusNet, dst: BusNet, src_version: int, dst_version: int) -> None:
    """Function-preserving warm start of ``dst`` (features ``dst_version``) from ``src``.

    Every parameter is copied; the first layer of each trunk gets the old weights in the input
    columns of the matching blocks (``train.features.input_columns``) and zeros in the new
    columns, so ``dst`` on a v3 observation computes exactly what ``src`` computes on the v2
    observation it contains (up to float summation order). The new columns still get gradients.
    """
    from train.features import input_columns, obs_size

    if src.obs_size != obs_size(src_version) or dst.obs_size != obs_size(dst_version):
        raise ValueError("obs sizes do not match the given features versions")
    cols = torch.as_tensor(input_columns(src_version, dst_version), dtype=torch.long)
    sd = src.state_dict()
    out = dst.state_dict()
    if set(sd) != set(out):
        raise ValueError(f"parameter sets differ: {sorted(set(sd) ^ set(out))}")
    for k, v in sd.items():
        if k in FIRST_LAYERS:
            w = torch.zeros_like(out[k])
            w[:, cols.to(w.device)] = v.to(w.device)
            out[k] = w
        else:
            if out[k].shape != v.shape:
                raise ValueError(f"{k}: shape {tuple(v.shape)} vs {tuple(out[k].shape)}")
            out[k] = v.clone().to(out[k].device)
    dst.load_state_dict(out)
