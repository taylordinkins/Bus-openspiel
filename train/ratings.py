"""Multiplayer Bradley-Terry / Elo ratings from game placements (Phase 4 unit 8b). Numpy only.

A game with ``n`` seats yields ``n (n - 1) / 2`` pairwise results: seat ``i`` beats seat ``j`` when
its final score is higher (half a win each on equal scores). Pairs of seats with the same entrant
label are skipped. The pairwise win matrix is fitted with the Bradley-Terry model by the
Zermelo/MM iteration; a small prior (``prior`` virtual half-wins for every ordered pair) keeps the
fit finite for entrants that never lost or never won. Ratings are ``400 * log10(strength)`` shifted
so that the anchor entrant sits at ``anchor_rating`` (Elo-like: a 400 point gap is a 10:1 pairwise
odds). Pairwise treatment ignores that seat scores in one game are correlated; it is a ranking
tool, not a calibrated probability model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]


def pair_wins(
    games: list[tuple[list[str], list[float]]], entrants: list[str]
) -> tuple[Array, Array]:
    """(W, A): ``W[i, j]`` = results of entrant i over j (win 1, tie 0.5); ``A[i]`` = seats of i.

    ``games`` = [(label per seat, final score per seat)]. Unknown labels are ignored.
    """
    idx = {e: k for k, e in enumerate(entrants)}
    w = np.zeros((len(entrants), len(entrants)))
    appear = np.zeros(len(entrants))
    for labels, scores in games:
        for a, la in enumerate(labels):
            if la in idx:
                appear[idx[la]] += 1
            for b in range(a + 1, len(labels)):
                lb = labels[b]
                if la not in idx or lb not in idx or la == lb:
                    continue
                ia, ib = idx[la], idx[lb]
                if scores[a] > scores[b]:
                    w[ia, ib] += 1.0
                elif scores[a] < scores[b]:
                    w[ib, ia] += 1.0
                else:
                    w[ia, ib] += 0.5
                    w[ib, ia] += 0.5
    return w, appear


def bradley_terry(w: Array, prior: float = 0.5, iters: int = 500) -> Array:
    """Strengths (geometric mean 1) maximising the BT likelihood of the win matrix ``w``."""
    n = len(w)
    ww = w + prior * (1.0 - np.eye(n))
    tot = ww + ww.T
    wins = ww.sum(axis=1)
    p = np.ones(n)
    for _ in range(iters):
        denom = (tot / (p[:, None] + p[None, :] + np.eye(n))).sum(axis=1)
        new = wins / np.maximum(denom, 1e-12)
        new = new / np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < 1e-9:
            p = new
            break
        p = new
    return p


def elo_ratings(
    games: list[tuple[list[str], list[float]]],
    entrants: list[str],
    anchor: str,
    anchor_rating: float = 1000.0,
    prior: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Per entrant: rating, seats played and mean pairwise score (wins / decided pairs)."""
    w, appear = pair_wins(games, entrants)
    p = bradley_terry(w, prior)
    elo = 400.0 * np.log10(p)
    elo = elo - elo[entrants.index(anchor)] + anchor_rating
    pairs = w.sum(axis=1) + w.sum(axis=0)
    return {
        e: {
            "rating": float(elo[k]),
            "games": float(appear[k]),
            "score_rate": float(w[k].sum() / pairs[k]) if pairs[k] > 0 else float("nan"),
        }
        for k, e in enumerate(entrants)
    }
