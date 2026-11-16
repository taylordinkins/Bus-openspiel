# pyright: basic, reportMissingImports=false
"""BusNet: policy/value MLP for PPO self-play (Phase 4 unit 2). Needs torch.

Shape: shared MLP trunk (tanh, as in OpenSpiel's PPO and robust for PPO value learning), a policy
head giving one logit per action id, and an N-player value head giving one value per ego-ordered
seat (seat 0 = player to move, padded to ``n_seats`` = 5; the caller masks absent seats), tanh-
bounded to [-1, 1] because shaped returns lie there. Orthogonal init after OpenSpiel's
``open_spiel/python/pytorch/ppo.py`` (copy-adapted: ``layer_init``, masked categorical).
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
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.obs_size, self.hidden, self.n_actions, self.n_seats = (
            obs_size, tuple(hidden), n_actions, n_seats,
        )  # fmt: skip
        sizes = (obs_size, *self.hidden)
        self.trunk = nn.ModuleList(
            layer_init(nn.Linear(a, b)) for a, b in zip(sizes[:-1], sizes[1:], strict=True)
        )
        self.pi = layer_init(nn.Linear(sizes[-1], n_actions), std=0.01)
        self.v = layer_init(nn.Linear(sizes[-1], n_seats), std=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(masked logits (B, n_actions), values (B, n_seats) in [-1, 1])."""
        x = obs
        for layer in self.trunk:
            x = torch.tanh(layer(x))
        return mask_logits(self.pi(x), mask), torch.tanh(self.v(x))

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
