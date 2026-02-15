from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib


def save_bundle(path: str, bundle: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict[str, Any]:
    return joblib.load(path)


def bundle_exists(path: str) -> bool:
    return Path(path).exists()
