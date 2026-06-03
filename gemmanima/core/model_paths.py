from __future__ import annotations

import os
from pathlib import Path


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def allow_legacy_model_paths() -> bool:
    return _truthy_env("GEMMANIMA_ALLOW_LEGACY_MODEL_PATHS")


def default_model_root() -> Path:
    override = os.environ.get("GEMMANIMA_MODEL_ROOT")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "GemmAnima" / "models"
    return Path.home() / ".gemmanima" / "models"


def model_path(component: str, filename: str, legacy_path: str | Path | None = None) -> Path:
    if legacy_path is not None and allow_legacy_model_paths():
        legacy = Path(legacy_path)
        if legacy.exists():
            return legacy
    return default_model_root() / component / filename
