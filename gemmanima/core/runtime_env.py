from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def configure_local_render_runtime() -> dict[str, object]:
    """Populate optional local render runtime env vars from common source-tree layouts."""

    changes: dict[str, str] = {}
    _set_first_existing(
        "GEMMANIMA_COMFY_ROOT",
        (
            Path("ComfyUI"),
            Path("..") / "ComfyUI",
            Path("E:/ComfyUI_sage/ComfyUI"),
            Path("E:/ComfyUI/ComfyUI"),
            Path("D:/ComfyUI/ComfyUI"),
        ),
        required=("folder_paths.py", "nodes.py"),
        changes=changes,
    )
    _set_first_existing(
        "GEMMANIMA_COMFY_SITE_PACKAGES",
        (
            Path(".venv/Lib/site-packages"),
            Path("E:/ComfyUI_sage/python_embeded/Lib/site-packages"),
            Path("E:/ComfyUI/python_embeded/Lib/site-packages"),
            Path("D:/ComfyUI/python_embeded/Lib/site-packages"),
        ),
        required=("comfy_aimdo",),
        changes=changes,
    )
    _set_first_existing(
        "GEMMANIMA_RENDER_PYTHON",
        (
            Path(".venv/Scripts/python.exe"),
            Path("E:/ComfyUI_sage/python_embeded/python.exe"),
            Path("E:/ComfyUI/python_embeded/python.exe"),
            Path("D:/ComfyUI/python_embeded/python.exe"),
        ),
        changes=changes,
    )
    _set_first_existing(
        "GEMMANIMA_SWAP_PROJECT_ROOT",
        (
            Path("."),
            Path("..") / "anima_gemma_swap",
            Path("E:/anima_gemma_swap"),
            Path("D:/anima_gemma_swap"),
        ),
        required=("scripts/core",),
        changes=changes,
    )
    _set_first_existing(
        "GEMMANIMA_GEMMA_HF_DIR",
        (
            Path("models/gemma_core_hf"),
            Path("D:/Projects/training/out/merged"),
            Path("E:/Projects/training/out/merged"),
        ),
        required=("model.safetensors", "tokenizer.json"),
        changes=changes,
    )
    os.environ.setdefault("GEMMANIMA_GEMMA_HIDDEN_DEVICE", "cpu")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return {
        "configured": changes,
        "env": {
            key: os.environ.get(key, "")
            for key in (
                "GEMMANIMA_COMFY_ROOT",
                "GEMMANIMA_COMFY_SITE_PACKAGES",
                "GEMMANIMA_RENDER_PYTHON",
                "GEMMANIMA_SWAP_PROJECT_ROOT",
                "GEMMANIMA_GEMMA_HF_DIR",
                "GEMMANIMA_GEMMA_HIDDEN_DEVICE",
            )
        },
    }


def _set_first_existing(
    name: str,
    candidates: Iterable[Path],
    *,
    required: tuple[str, ...] = (),
    changes: dict[str, str],
) -> None:
    if os.environ.get(name):
        return
    for candidate in candidates:
        resolved = candidate.expanduser()
        if _candidate_ready(resolved, required=required):
            os.environ[name] = str(resolved)
            changes[name] = str(resolved)
            return


def _candidate_ready(path: Path, *, required: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    for rel in required:
        if not (path / rel).exists():
            return False
    return True
