from __future__ import annotations

from pathlib import Path
from typing import Any

from gemmanima import GemmAnimaConductor
from gemmanima.core.config import EngineConfig, ModelConfig
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.schemas import ChatTurn
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.modules.hiddenstage_exit import HiddenStageExit
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter
from gemmanima.modules.real_anima_renderer import ExternalAnimaRendererAdapter
from gemmanima.rendering.backends import audit_renderer_backend


def response_to_dict(response) -> dict[str, Any]:
    return {
        "mode": response.mode.value,
        "status": response.status.value,
        "message": response.message,
        "prompt": response.prompt,
        "manifest_path": str(response.manifest_path) if response.manifest_path else None,
        "output_path": str(response.output_path) if response.output_path else None,
        "progress": list(response.progress),
        "job_id": response.job_id,
    }


def build_plan_overrides(payload: dict[str, Any]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if payload.get("steps") is not None:
        overrides["steps"] = int(payload["steps"])
    if payload.get("size") is not None:
        size = int(payload["size"])
        overrides["width"] = size
        overrides["height"] = size
    if payload.get("cfg") is not None:
        overrides["cfg"] = float(payload["cfg"])
    if payload.get("seed") is not None and str(payload["seed"]).strip():
        overrides["seed"] = int(payload["seed"])
    return overrides


def build_config(payload: dict[str, Any]) -> EngineConfig:
    config = EngineConfig()
    anima_dm = str(payload.get("anima_dm") or "").strip()
    if not anima_dm:
        return config
    models = ModelConfig(
        gemma_planner_adapter=config.models.gemma_planner_adapter,
        gemma_vision_embedding=config.models.gemma_vision_embedding,
        anima_diffusion_model=Path(anima_dm),
        anima_text_encoder=config.models.anima_text_encoder,
        anima_vae=config.models.anima_vae,
        hiddenstage_bridge=config.models.hiddenstage_bridge,
    )
    return EngineConfig(models=models, hardware=config.hardware, renderer_profiles=config.renderer_profiles)


def build_renderer(payload: dict[str, Any], *, image_root: Path, config: EngineConfig) -> AnimaRendererAdapter | None:
    renderer_name = str(payload.get("renderer") or "dry-run")
    if renderer_name == "dry-run":
        return None
    if renderer_name == "external-script":
        return ExternalAnimaRendererAdapter(image_root)
    if renderer_name in {"real", "in-process"}:
        return InProcessAnimaRendererAdapter(
            image_root,
            config=config,
            unet_dtype=str(payload.get("unet_dtype") or "fp8_e4m3fn_fast"),
        )
    raise ValueError(f"unknown renderer: {renderer_name}")


def handle_chat_payload(payload: dict[str, Any], *, base_dir: str | Path = "runs") -> dict[str, Any]:
    message = str(payload.get("message", "")).strip()
    if not message:
        return {"error": "message is required", "status": "failed"}

    session_id = payload.get("session_id")
    base = Path(base_dir)
    config = build_config(payload)
    try:
        renderer = build_renderer(payload, image_root=base / "images", config=config)
    except ValueError as exc:
        return {"error": str(exc), "status": "failed"}
    conductor = GemmAnimaConductor(
        session_id=str(session_id) if session_id else None,
        manifest_root=base / "manifests",
        image_root=base / "images",
        renderer=renderer,
        config=config,
        plan_overrides=build_plan_overrides(payload),
    )
    for item in payload.get("history", ()):
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role and content:
            conductor.history.append(ChatTurn(role=role, content=content))
    return response_to_dict(conductor.handle_user_message(message))


def handle_health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "models": ModelRegistry().health(),
        "hiddenstage_bridge": HiddenStageExit().audit_bridge().to_json_dict(),
        "renderers": audit_renderer_backend(),
    }
