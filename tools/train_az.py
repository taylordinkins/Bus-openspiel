"""Gumbel AlphaZero trainer CLI (Phase 4 unit 8h): ``python -m tools.train_az`` (train group).

Options and outputs: see ``train.az_trainer`` and ``docs/TRAINING.md``. This module stays light on
purpose: spawn workers re-import the main module, and importing torch there would cost every
actor process ~0.5 GB of RSS; the actors themselves (``train.az_selfplay``) are numpy only.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from train.az_trainer import main as run

    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
