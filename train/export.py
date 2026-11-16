# pyright: basic, reportMissingImports=false
"""Export a trained BusNet to the numpy ``.npz`` format read by ``bots.net_forward``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from train.model import BusNet


def export_npz(model: BusNet, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    """Write weights, activation, sizes and the observation layout hash; ``meta`` is extra info.

    Format 2 (``bots.net_forward.EXPORT_FORMAT``); a model without a separate value trunk is
    written without ``vw*`` arrays and loads identically.
    """
    from train.league import save_weights
    from train.ppo import net_to_weights

    save_weights(net_to_weights(model), Path(path), meta)
