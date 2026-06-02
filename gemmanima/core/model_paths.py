from __future__ import annotations

import os
from pathlib import Path


def default_model_root() -> Path:
    override = os.environ.get("GEMMANIMA_MODEL_ROOT")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "GemmAnima" / "models"
    return Path.home() / ".gemmanima" / "models"


def model_path(component: str, filename: str, legacy_path: str | Path) -> Path:
    legacy = Path(legacy_path)
    if "GEMMANIMA_MODEL_ROOT" not in os.environ and legacy.exists():
        return legacy
    return default_model_root() / component / filename
