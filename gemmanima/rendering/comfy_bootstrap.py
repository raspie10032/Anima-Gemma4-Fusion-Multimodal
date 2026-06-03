from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from gemmanima.core.model_paths import default_model_root


_TORCHVISION_LIBRARY_SHIMS: list[object] = []


class NativeAttentionImportBlocker:
    """Prevents optional native attention extensions from crashing mismatched Python/Torch runtimes."""

    blocked_roots = ("flash_attn", "sageattention", "sageattn3", "xformers")

    def find_spec(
        self,
        fullname: str,
        path: object | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root in self.blocked_roots:
            raise ModuleNotFoundError(f"{root} is disabled for GemmAnima local rendering stability")
        return None


@dataclass(frozen=True)
class ComfyBootstrapConfig:
    comfy_root: Path = field(default_factory=lambda: _env_path("GEMMANIMA_COMFY_ROOT", "ComfyUI"))
    models_root: Path = field(default_factory=lambda: _env_path("GEMMANIMA_COMFY_MODELS_ROOT", str(default_model_root() / "comfy")))
    embedded_site_packages: Path = field(
        default_factory=lambda: _env_path("GEMMANIMA_COMFY_SITE_PACKAGES", ".venv/Lib/site-packages")
    )
    project_root: Path = field(default_factory=lambda: _env_path("GEMMANIMA_SWAP_PROJECT_ROOT", "."))
    model_folders: tuple[str, ...] = ("diffusion_models", "unet", "text_encoders", "clip", "vae", "loras")


@dataclass(frozen=True)
class ComfyBootstrapResult:
    comfy_root: Path
    models_root: Path
    embedded_site_packages: Path
    project_core: Path
    model_folders: dict[str, Path]
    imported_folder_paths: bool


def bootstrap_comfy(
    config: ComfyBootstrapConfig | None = None,
    *,
    import_folder_paths: bool = True,
    comfy_args: tuple[str, ...] = (),
) -> ComfyBootstrapResult:
    resolved = config or ComfyBootstrapConfig()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if resolved.embedded_site_packages.exists() and str(resolved.embedded_site_packages) not in sys.path:
        sys.path.append(str(resolved.embedded_site_packages))
    if str(resolved.comfy_root) not in sys.path:
        sys.path.insert(0, str(resolved.comfy_root))
    project_core = resolved.project_root / "scripts" / "core"
    if str(project_core) not in sys.path:
        sys.path.insert(0, str(project_core))
    _install_native_attention_import_blocker()
    _ensure_torchvision_nms_operator()

    model_folders = {name: resolved.models_root / name for name in resolved.model_folders}
    if not import_folder_paths:
        return ComfyBootstrapResult(
            comfy_root=resolved.comfy_root,
            models_root=resolved.models_root,
            embedded_site_packages=resolved.embedded_site_packages,
            project_core=project_core,
            model_folders=model_folders,
            imported_folder_paths=False,
        )

    saved_argv = list(sys.argv)
    try:
        sys.argv = [sys.argv[0], *_resolved_comfy_args(comfy_args)]
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
        embedded_site_packages=resolved.embedded_site_packages,
        project_core=project_core,
        model_folders=model_folders,
        imported_folder_paths=True,
    )


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _resolved_comfy_args(comfy_args: tuple[str, ...]) -> list[str]:
    if comfy_args:
        return [str(arg) for arg in comfy_args]
    raw = os.environ.get("GEMMANIMA_COMFY_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _ensure_torchvision_nms_operator() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except RuntimeError:
        pass
    try:
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_LIBRARY_SHIMS.append(library)
    except Exception:
        return


def _install_native_attention_import_blocker() -> None:
    if any(isinstance(finder, NativeAttentionImportBlocker) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, NativeAttentionImportBlocker())
