"""Trained-net seat: `net:<name>`, with `<name>` resolved only through `models/manifest.yaml`.

The name is looked up in the allow-list; nothing user-supplied is ever used as a path. Weights are
loaded once per name (numpy only, no torch). Observation comes from `train.features.encode`, which
is imported lazily and is torch-free. Play is greedy (argmax of the masked logits) unless
`sample=True`. Any runtime failure or illegal result falls back to the heuristic bot (logged).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyspiel
import yaml

from bots.heuristic import HeuristicBot
from bots.net_forward import NetWeights, StaleModelError

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.yaml"
_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    file: str
    label: str
    description: str
    players: tuple[int, ...]
    features_version: int

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.file


@lru_cache(maxsize=1)
def manifest() -> dict[str, ModelInfo]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    out: dict[str, ModelInfo] = {}
    for e in raw.get("models", []):
        info = ModelInfo(
            e["name"], e["file"], e["label"], e.get("description", ""),
            tuple(int(p) for p in e["players"]), int(e["features_version"]),
        )  # fmt: skip
        if not _NAME_RE.fullmatch(info.name) or Path(info.file).name != info.file:
            raise ValueError(f"bad manifest entry {info.name!r}")
        out[info.name] = info
    return out


def resolve_model(name: str, players: int | None = None) -> ModelInfo:
    """The manifest entry for `name`; ValueError if unknown, malformed or not for `players`."""
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid net model name {name!r}")
    info = manifest().get(name)
    if info is None:
        raise ValueError(f"unknown net model {name!r}")
    if players is not None and players not in info.players:
        raise ValueError(f"net model {name!r} supports {list(info.players)} players, not {players}")
    return info


@cache
def load_weights(name: str) -> NetWeights:
    """Load (once per name) the allow-listed model; StaleModelError if it does not fit."""
    from train.features import FEATURES_VERSION

    info = resolve_model(name)
    if info.features_version != FEATURES_VERSION:
        raise StaleModelError(
            f"net model {name!r} was built for features v{info.features_version}; "
            f"current are v{FEATURES_VERSION}: retrain or re-export"
        )
    return NetWeights.load(info.path)


class NetBot:
    def __init__(self, name: str, sample: bool = False) -> None:
        self.name = name
        self.sample = sample
        self.weights = load_weights(name)  # raises ValueError/StaleModelError at seat creation
        self._fallback = HeuristicBot()

    def _pick(self, state: pyspiel.State, legal: list[int], rng: random.Random) -> int:
        from train.features import encode

        mask = np.zeros(self.weights.n_actions, np.bool_)
        mask[legal] = True
        obs = encode(state.core)  # type: ignore[attr-defined]
        if not self.sample:
            logits, _ = self.weights.forward(obs, mask)
            return int(np.argmax(logits))
        probs, _ = self.weights.policy(obs, mask)
        return int(rng.choices(range(len(probs)), weights=probs.tolist())[0])

    def choose(self, state: pyspiel.State, rng: random.Random) -> int:
        legal = list(state.legal_actions())
        if len(legal) == 1:
            return legal[0]
        try:
            a = self._pick(state, legal, rng)
            if a in legal:
                return a
            log.warning("net %r returned illegal action %s; using heuristic", self.name, a)
        except Exception:
            log.exception("net %r failed; using heuristic", self.name)
        return self._fallback.choose(state, rng)


def parse_net_param(param: str | None) -> NetBot:
    if param is None:
        raise ValueError("net needs a model name: net:<name>")
    resolve_model(param)
    return NetBot(param)


def check_players(seat_type: str, players: int) -> None:
    """ValueError if a net-backed seat type does not support `players`; no-op otherwise."""
    kind, _, name = seat_type.partition(":")
    if kind == "net":
        resolve_model(name, players)
    elif kind in ("mcts_net", "gumbel"):
        resolve_model(name.partition("-")[2], players)
