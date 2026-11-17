"""Training tests: skip, rather than fail, when torch (the optional ``train`` group) is absent.

CI runs ``uv sync --locked`` without ``--group train``. Most torch tests guard themselves with
``pytest.importorskip("torch")``; this hook applies the same rule to any test that reaches torch
indirectly (for example by importing ``train.ppo`` or ``tools.train_ppo``), so a new test cannot
turn CI red just because torch is not installed. Any other import error still fails.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    try:
        return (yield)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            pytest.skip("torch not installed (install the train group: uv sync --group train)")
        raise
