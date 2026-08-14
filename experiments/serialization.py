"""Persistence for experiment outputs used to regenerate reports."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any


__all__ = ["load_experiment_outputs", "save_experiment_outputs"]


_FORMAT = "celestial-nav-sim-outputs-v1"


def save_experiment_outputs(
    path: str | Path,
    outputs: Mapping[str, Any],
) -> Path:
    """Save all pipeline outputs in one pickle bundle."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(
            {"format": _FORMAT, "outputs": dict(outputs)},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return destination


def load_experiment_outputs(path: str | Path) -> dict[str, Any]:
    """Load a trusted pickle bundle created by this project."""

    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format") != _FORMAT:
        raise ValueError("unsupported experiment-output format")
    return payload["outputs"]
