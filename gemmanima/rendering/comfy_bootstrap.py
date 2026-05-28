from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComfyBootstrapConfig:
    comfy_root: Path = Path(r"E:\ComfyUI_anima_exp")
    models_root: Path = Path(r"E:\ComfyUI_sage\ComfyUI\models")
    project_root: Path = Path(r"E:\anima_gemma_swap")
    model_folders: tuple[str, ...] = ("diffusion_models", "unet", "text_encoders", "clip", "vae", "loras")


@dataclass(frozen=True)
class ComfyBootstrapResult:
    comfy_root: Path
    models_root: Path
    project_core: Path
    model_folders: dict[str, Path]
    imported_folder_paths: bool


def bootstrap_comfy(
    config: ComfyBootstrapConfig | None = None,
    *,
    import_folder_paths: bool = True,
) -> ComfyBootstrapResult:
    resolved = config or ComfyBootstrapConfig()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if str(resolved.comfy_root) not in sys.path:
        sys.path.insert(0, str(resolved.comfy_root))
    project_core = resolved.project_root / "scripts" / "core"
    if str(project_core) not in sys.path:
        sys.path.insert(0, str(project_core))

    model_folders = {name: resolved.models_root / name for name in resolved.model_folders}
    if not import_folder_paths:
        return ComfyBootstrapResult(
            comfy_root=resolved.comfy_root,
            models_root=resolved.models_root,
            project_core=project_core,
            model_folders=model_folders,
            imported_folder_paths=False,
        )

    saved_argv = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]]
        import folder_paths

        for name, path in model_folders.items():
            folder_paths.add_model_folder_path(name, str(path))
        folder_paths.add_model_folder_path(
            "loras",
            str(resolved.project_root / "lora_tests" / "anima_loras"),
        )
    finally:
        sys.argv = saved_argv

    return ComfyBootstrapResult(
        comfy_root=resolved.comfy_root,
        models_root=resolved.models_root,
        project_core=project_core,
        model_folders=model_folders,
        imported_folder_paths=True,
    )
