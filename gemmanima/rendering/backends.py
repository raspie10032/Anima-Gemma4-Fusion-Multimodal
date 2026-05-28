from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gemmanima.core.config import EngineConfig
from gemmanima.training.real_render import (
    DEFAULT_CHAT_RENDER_SCRIPT,
    DEFAULT_EMBEDDED_PYTHON,
    audit_real_render_dependencies,
)
from gemmanima.rendering.comfy_bootstrap import ComfyBootstrapConfig
from gemmanima.rendering.gemma_hidden import gemma_hidden_environment
from gemmanima.rendering.t5_tokenizer import t5_tokenizer_environment
from gemmanima.rendering.anima_sampler import anima_sampler_environment


RendererBackendName = Literal["external_script", "in_process"]


@dataclass(frozen=True)
class RendererBackendProfile:
    name: RendererBackendName
    execution: str
    ready: bool
    dependency_ready: bool
    checks: dict[str, bool]
    next_steps: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "execution": self.execution,
            "ready": self.ready,
            "dependency_ready": self.dependency_ready,
            "checks": self.checks,
            "next_steps": list(self.next_steps),
        }


def renderer_backend_profile(
    name: RendererBackendName,
    *,
    config: EngineConfig | None = None,
) -> RendererBackendProfile:
    if name == "external_script":
        return _external_script_profile(config=config)
    if name == "in_process":
        return _in_process_profile(config=config)
    raise ValueError(f"unknown renderer backend: {name}")


def audit_renderer_backend(config: EngineConfig | None = None) -> dict[str, dict[str, object]]:
    return {
        "external_script": renderer_backend_profile("external_script", config=config).to_json_dict(),
        "in_process": renderer_backend_profile("in_process", config=config).to_json_dict(),
    }


def _external_script_profile(config: EngineConfig | None = None) -> RendererBackendProfile:
    deps = audit_real_render_dependencies(config=config)
    checks = {key: bool(value) for key, value in deps["checks"].items()}
    return RendererBackendProfile(
        name="external_script",
        execution="subprocess",
        ready=bool(deps["ready"]),
        dependency_ready=bool(deps["ready"]),
        checks=checks,
        next_steps=("legacy_script_removed",),
    )


def _in_process_profile(config: EngineConfig | None = None) -> RendererBackendProfile:
    resolved_config = config or EngineConfig()
    bootstrap = ComfyBootstrapConfig()
    gemma_env = gemma_hidden_environment()
    t5_env = t5_tokenizer_environment(load_tokenizer=False)
    sampler_env = anima_sampler_environment()
    checks = {
        "embedded_python": DEFAULT_EMBEDDED_PYTHON.exists(),
        "comfy_bootstrap_module": True,
        "comfy_root": bootstrap.comfy_root.exists(),
        "comfy_import": _can_import_from_comfy_root(bootstrap.comfy_root),
        "gemma_hidden_provider_module": True,
        "gemma_model_safetensors": bool(gemma_env["model_safetensors"]),
        "gemma_tokenizer_json": bool(gemma_env["tokenizer_json"]),
        "t5_tokenizer_provider_module": bool(t5_env["provider_module"]),
        "sampler_runtime_module": bool(sampler_env["sampler_module"]),
        "adapter_attach_module": True,
        "hiddenstage_bridge": resolved_config.models.hiddenstage_bridge.exists(),
        "anima_diffusion_model": resolved_config.models.anima_diffusion_model.exists(),
        "anima_vae": resolved_config.models.anima_vae.exists(),
        "legacy_script_reference": DEFAULT_CHAT_RENDER_SCRIPT.exists(),
    }
    dependency_ready = all(
        checks[key]
        for key in (
            "embedded_python",
            "comfy_root",
            "comfy_import",
            "gemma_hidden_provider_module",
            "gemma_model_safetensors",
            "gemma_tokenizer_json",
            "t5_tokenizer_provider_module",
            "sampler_runtime_module",
            "adapter_attach_module",
            "hiddenstage_bridge",
            "anima_diffusion_model",
            "anima_vae",
        )
    )
    return RendererBackendProfile(
        name="in_process",
        execution="in_process",
        ready=dependency_ready,
        dependency_ready=dependency_ready,
        checks=checks,
        next_steps=(
            "legacy_script_removed",
        ),
    )


def _can_import_from_comfy_root(comfy_root: Path) -> bool:
    return (comfy_root / "comfy").is_dir() and (comfy_root / "nodes.py").exists()
