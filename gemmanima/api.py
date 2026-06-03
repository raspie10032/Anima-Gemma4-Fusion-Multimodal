from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from gemmanima import GemmAnimaConductor
from gemmanima.core.config import EngineConfig, ModelConfig
from gemmanima.core.generation_presets import (
    apply_generation_preset,
    generation_preset_options,
    normalize_anima_cfg,
    resolution_preset_options,
    sampler_options,
    scheduler_options,
)
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.schemas import ChatTurn, GenerationPlan
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.modules.hiddenstage_exit import HiddenStageExit
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter
from gemmanima.modules.local_worker_anima_renderer import LocalWorkerAnimaRendererAdapter
from gemmanima.modules.real_anima_renderer import ExternalAnimaRendererAdapter
from gemmanima.modules.tipo_runtime import (
    TipoTextConfig,
    clean_vision_tags,
    normalize_chat_mode,
    output_contract_for_mode,
    parse_image_generation_contract,
    run_tipo_text_chat,
    run_tipo_vision_tag,
    tipo_text_health,
    tipo_vision_health,
)
from gemmanima.rendering.backends import audit_renderer_backend


ARTIST_TAG_MARKERS = ("@", "artist:", "by ")
TEXT_RENDERING_HINTS = (
    "caption",
    "label",
    "letters",
    "logo",
    "readable",
    "sign",
    "text",
    "typography",
    "word",
    "간판",
    "글자",
    "로고",
    "문구",
    "텍스트",
)


def response_to_dict(response) -> dict[str, Any]:
    return {
        "mode": response.mode.value,
        "status": response.status.value,
        "message": response.message,
        "prompt": response.prompt,
        "plan": response.plan.to_json_dict() if response.plan else None,
        "manifest_path": str(response.manifest_path) if response.manifest_path else None,
        "output_path": str(response.output_path) if response.output_path else None,
        "progress": list(response.progress),
        "clarification_required": response.clarification_required,
        "conflict": response.conflict,
        "job_id": response.job_id,
    }


def build_generation_preset_overrides(payload: dict[str, Any]) -> dict[str, object]:
    plan = apply_payload_generation_preset(GenerationPlan(prompt="preset"), payload)
    return {
        "width": plan.width,
        "height": plan.height,
        "steps": plan.steps,
        "cfg": plan.cfg,
        "sampler": plan.sampler,
        "scheduler": plan.scheduler,
        "renderer_profile": plan.renderer_profile,
        "lora_stack": plan.lora_stack,
    }


def build_plan_overrides(payload: dict[str, Any]) -> dict[str, object]:
    overrides: dict[str, object] = build_generation_preset_overrides(payload)
    if _has_payload_value(payload, "steps"):
        overrides["steps"] = int(payload["steps"])
    if _has_payload_value(payload, "size"):
        size = int(payload["size"])
        overrides["width"] = size
        overrides["height"] = size
    if _has_payload_value(payload, "cfg"):
        overrides["cfg"] = normalize_anima_cfg(
            float(payload["cfg"]),
            allow_low_cfg=bool(payload.get("allow_low_cfg")),
        )
    if payload.get("seed") is not None and str(payload["seed"]).strip():
        overrides["seed"] = int(payload["seed"])
    if _has_payload_value(payload, "width"):
        overrides["width"] = int(payload["width"])
    if _has_payload_value(payload, "height"):
        overrides["height"] = int(payload["height"])
    if payload.get("sampler"):
        overrides["sampler"] = str(payload["sampler"])
    if payload.get("scheduler"):
        overrides["scheduler"] = str(payload["scheduler"])
    if payload.get("renderer_profile"):
        overrides["renderer_profile"] = str(payload["renderer_profile"])
    if payload.get("lora_stack") is not None:
        overrides["lora_stack"] = tuple(str(item) for item in payload.get("lora_stack") or ())
    reference_image_path = str(payload.get("reference_image_path") or payload.get("image_path") or "").strip()
    if reference_image_path:
        overrides["reference_image_path"] = reference_image_path
    return overrides


def _has_payload_value(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    return value is not None and str(value).strip() != ""


def bridge_profile_options(config: EngineConfig | None = None) -> dict[str, dict[str, str]]:
    models = (config or EngineConfig()).models
    return {
        name: {
            "label": profile.label,
            "checkpoint": str(profile.checkpoint),
            "role": profile.role,
            "notes": profile.notes,
        }
        for name, profile in models.bridge_profiles().items()
    }


def select_bridge_profile_name(payload: dict[str, Any], *, config: EngineConfig | None = None) -> str:
    selected = str(payload.get("bridge_profile") or "").strip()
    profiles = (config or EngineConfig()).models.bridge_profiles()
    if selected:
        if selected not in profiles:
            known = ", ".join(sorted(profiles))
            raise ValueError(f"unknown bridge_profile {selected!r}; known profiles: {known}")
        return selected

    text = str(payload.get("message") or payload.get("prompt") or "").lower()
    if any(hint in text for hint in TEXT_RENDERING_HINTS) or _contains_quoted_text(text):
        return "text_exact"
    if any(marker in text for marker in ARTIST_TAG_MARKERS):
        return "style_artist"
    return "balanced_pose"


def _contains_quoted_text(text: str) -> bool:
    return ('"' in text and text.count('"') >= 2) or ("'" in text and text.count("'") >= 2)


def build_config(payload: dict[str, Any]) -> EngineConfig:
    config = EngineConfig()
    anima_dm = str(payload.get("anima_dm") or "").strip()
    hiddenstage_bridge = str(payload.get("hiddenstage_bridge") or "").strip()
    profile_name = select_bridge_profile_name(payload, config=config)
    profile_bridge = config.models.bridge_profiles()[profile_name].checkpoint
    selected_bridge = Path(hiddenstage_bridge) if hiddenstage_bridge else profile_bridge
    if not anima_dm and selected_bridge == config.models.hiddenstage_bridge:
        return config
    models = ModelConfig(
        gemma_planner_adapter=config.models.gemma_planner_adapter,
        gemma_vision_embedding=config.models.gemma_vision_embedding,
        anima_diffusion_model=Path(anima_dm) if anima_dm else config.models.anima_diffusion_model,
        anima_vae=config.models.anima_vae,
        hiddenstage_bridge=selected_bridge,
        hiddenstage_bridge_balanced_pose=config.models.hiddenstage_bridge_balanced_pose,
        hiddenstage_bridge_style_artist=config.models.hiddenstage_bridge_style_artist,
        hiddenstage_bridge_text_exact=config.models.hiddenstage_bridge_text_exact,
    )
    return EngineConfig(models=models, hardware=config.hardware, renderer_profiles=config.renderer_profiles)


def build_renderer(payload: dict[str, Any], *, image_root: Path, config: EngineConfig) -> AnimaRendererAdapter | None:
    renderer_name = str(payload.get("renderer") or "dry-run")
    if renderer_name == "dry-run":
        return None
    if renderer_name in {"local-worker", "worker"}:
        return LocalWorkerAnimaRendererAdapter(
            image_root,
            config=config,
            unet_dtype=str(payload.get("unet_dtype") or "fp8_e4m3fn_fast"),
            tiled_vae=_optional_bool(payload.get("tiled_vae"), default=True),
            comfy_args=_comfy_memory_args(payload),
        )
    if renderer_name == "external-script":
        return ExternalAnimaRendererAdapter(image_root)
    if renderer_name in {"real", "in-process"}:
        return InProcessAnimaRendererAdapter(
            image_root,
            config=config,
            unet_dtype=str(payload.get("unet_dtype") or "fp8_e4m3fn_fast"),
            tiled_vae=_optional_bool(payload.get("tiled_vae"), default=True),
            comfy_args=_comfy_memory_args(payload),
        )
    raise ValueError(f"unknown renderer: {renderer_name}")


def generation_plan_from_chat_contract(contract: dict[str, Any]) -> GenerationPlan:
    data: dict[str, Any] = {
        "prompt": contract["prompt"],
        "negative_prompt": contract.get("negative_prompt", ""),
    }
    for key in (
        "width",
        "height",
        "steps",
        "cfg",
        "seed",
        "sampler",
        "scheduler",
        "lora_stack",
        "renderer_profile",
    ):
        value = contract.get(key)
        if value not in (None, ""):
            data[key] = value
    return GenerationPlan.from_dict(data)


def apply_payload_generation_preset(plan: GenerationPlan, payload: dict[str, Any]) -> GenerationPlan:
    return apply_generation_preset(
        plan,
        preset_name=str(payload.get("generation_preset") or "anima_balanced"),
        resolution_name=str(payload.get("resolution_preset") or "square_1024"),
        orientation=str(payload.get("orientation") or ""),
        custom_width=_optional_int(payload.get("custom_width") or payload.get("width")),
        custom_height=_optional_int(payload.get("custom_height") or payload.get("height")),
        overrides=_explicit_plan_overrides(payload),
        allow_low_cfg=bool(payload.get("allow_low_cfg")),
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_bool(value: Any, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def build_tipo_text_config(payload: dict[str, Any]) -> TipoTextConfig:
    config = TipoTextConfig()
    if "headroom_enabled" in payload:
        config = replace(
            config,
            headroom_enabled=_optional_bool(payload.get("headroom_enabled"), default=config.headroom_enabled),
        )
    if payload.get("headroom_model"):
        config = replace(config, headroom_model=str(payload.get("headroom_model")).strip())
    if _has_payload_value(payload, "headroom_timeout_seconds"):
        config = replace(config, headroom_timeout_seconds=float(payload["headroom_timeout_seconds"]))
    return config


def _comfy_memory_args(payload: dict[str, Any]) -> tuple[str, ...]:
    args: list[str] = []
    memory_mode = str(payload.get("memory_mode") or "").strip().lower()
    if memory_mode in {"lowvram", "novram"}:
        args.append(f"--{memory_mode}")
    if _optional_bool(payload.get("cpu_vae"), default=False):
        args.append("--cpu-vae")
    reserve_vram = payload.get("reserve_vram")
    if reserve_vram not in (None, ""):
        args.extend(["--reserve-vram", str(float(reserve_vram))])
    return tuple(args)


def _explicit_plan_overrides(payload: dict[str, Any]) -> dict[str, object]:
    explicit: dict[str, object] = {}
    for key in ("steps", "cfg", "seed", "sampler", "scheduler", "renderer_profile", "width", "height"):
        if payload.get(key) not in (None, ""):
            value = payload[key]
            if key in {"steps", "seed", "width", "height"}:
                value = int(value)
            elif key == "cfg":
                value = normalize_anima_cfg(float(value), allow_low_cfg=bool(payload.get("allow_low_cfg")))
            else:
                value = str(value)
            explicit[key] = value
    if payload.get("lora_stack") is not None:
        explicit["lora_stack"] = tuple(str(item) for item in payload.get("lora_stack") or ())
    return explicit


def handle_chat_payload(payload: dict[str, Any], *, base_dir: str | Path = "runs") -> dict[str, Any]:
    task = str(payload.get("task") or payload.get("mode") or "auto").strip().lower()
    language = str(payload.get("language") or "ko").strip() or "ko"
    chat_mode = normalize_chat_mode(str(payload.get("chat_mode") or payload.get("intent") or "general_chat"))
    output_contract = output_contract_for_mode(chat_mode)
    if task in {"tag", "tag_image", "vision_tag", "image_tag"}:
        return handle_tag_payload(payload)

    message = str(payload.get("message", "")).strip()
    if not message:
        return {"error": "message is required", "status": "failed"}
    route_as_text_chat = task in {"chat", "talk"}
    auto_route: dict[str, Any] | None = None
    if task == "auto":
        if chat_mode != "general_chat":
            route_as_text_chat = True
        else:
            auto_route = classify_auto_intent(payload, message=message, language=language)
            intent = str(auto_route.get("intent") or "chat")
            if intent == "tag_image" and str(payload.get("image_path") or "").strip():
                return handle_tag_payload(payload)
            if intent == "generate_image":
                chat_mode = "image_generation_request"
                output_contract = output_contract_for_mode(chat_mode)
                route_as_text_chat = True
            elif intent == "chat":
                route_as_text_chat = True
            else:
                routing_conductor = GemmAnimaConductor(config=build_config(payload))
                route_as_text_chat = not routing_conductor.planner.is_image_request(message)
                if route_as_text_chat and _should_resume_generation_from_history(payload, routing_conductor):
                    route_as_text_chat = False
    if route_as_text_chat:
        result = run_tipo_text_chat(
            message=message,
            language=language,
            chat_mode=chat_mode,
            config=build_tipo_text_config(payload),
            history=[
                {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
                for item in payload.get("history", ())
                if isinstance(item, dict)
            ],
        )
        if result.get("status") != "completed":
            if auto_route and auto_route.get("intent") == "generate_image" and chat_mode == "image_generation_request":
                fallback = handle_direct_generation_payload(payload, message=message, base_dir=base_dir, force_plan=True)
                fallback["auto_route"] = auto_route
                fallback["progress"] = [*fallback.get("progress", ()), "contract:fallback"]
                fallback["chat_mode"] = chat_mode
                fallback["output_contract"] = output_contract
                fallback["contract_error"] = result.get("error", "image generation contract failed")
                return fallback
            return {
                "mode": "chat",
                "status": "failed",
                "error": result.get("error", "chat generation failed"),
                "error_code": result.get("error_code", "chat_generation_failed"),
                "preflight": result.get("preflight"),
                "headroom": result.get("headroom"),
                "warnings": result.get("warnings", []),
                "chat_mode": result.get("chat_mode") or chat_mode,
                "output_contract": result.get("output_contract") or output_contract,
                "progress": ["route:chat", "tipo_text:failed"],
            }
        if chat_mode == "image_generation_request" and output_contract == "image_generation_json":
            contract = result.get("image_generation")
            if not isinstance(contract, dict) or contract.get("status") != "completed":
                contract = parse_image_generation_contract(
                    str(result.get("raw") or result.get("message") or "")
                )
            if contract.get("status") != "completed":
                if auto_route and auto_route.get("intent") == "generate_image":
                    fallback = handle_direct_generation_payload(payload, message=message, base_dir=base_dir, force_plan=True)
                    fallback["auto_route"] = auto_route
                    fallback["progress"] = [*fallback.get("progress", ()), "contract:fallback"]
                    fallback["chat_mode"] = chat_mode
                    fallback["output_contract"] = output_contract
                    fallback["contract_error"] = contract.get("error", "image generation contract failed")
                    return fallback
                return {
                    "mode": "chat",
                    "status": "failed",
                    "error": contract.get("error", "image generation contract failed"),
                    "error_code": "image_generation_contract_failed",
                    "chat_mode": chat_mode,
                    "output_contract": output_contract,
                    "image_generation": contract,
                    "progress": ["route:chat", "tipo:text", "contract:failed"],
                }
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
            plan = apply_payload_generation_preset(generation_plan_from_chat_contract(contract), payload)
            response = response_to_dict(conductor.handle_generation_plan(message, plan))
            response.update(
                {
                    "chat_mode": chat_mode,
                    "output_contract": output_contract,
                    "generation_preset": str(payload.get("generation_preset") or "anima_balanced"),
                    "resolution_preset": str(payload.get("resolution_preset") or "square_1024"),
                    "orientation": str(payload.get("orientation") or ""),
                    "chat_notes": contract.get("notes", ""),
                    "image_generation": contract,
                    "plan": plan.to_json_dict(),
                    "auto_route": auto_route,
                    "tipo_model": result.get("model"),
                    "tipo_device": result.get("device"),
                    "tipo_seconds": result.get("seconds"),
                    "headroom": result.get("headroom"),
                    "warnings": result.get("warnings", []),
                }
            )
            return response
        return {
            "mode": "chat",
            "status": "completed",
            "message": result.get("message", ""),
            "raw": result.get("raw", ""),
            "stderr_tail": result.get("stderr_tail", ""),
            "seconds": result.get("seconds"),
            "model": result.get("model"),
            "device": result.get("device"),
            "language": result.get("language", language),
            "chat_mode": result.get("chat_mode") or chat_mode,
            "output_contract": result.get("output_contract") or output_contract,
            "prompt": None,
            "manifest_path": None,
            "output_path": None,
            "progress": ["route:chat", "tipo:text"],
            "clarification_required": False,
            "conflict": None,
            "job_id": None,
            "auto_route": auto_route,
            "headroom": result.get("headroom"),
            "warnings": result.get("warnings", []),
        }

    force_plan = task in {"generate", "generate_image", "image", "img"}
    return handle_direct_generation_payload(payload, message=message, base_dir=base_dir, force_plan=force_plan)


def handle_direct_generation_payload(
    payload: dict[str, Any],
    *,
    message: str,
    base_dir: str | Path,
    force_plan: bool = False,
) -> dict[str, Any]:
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
    if not force_plan:
        return response_to_dict(conductor.handle_user_message(message))
    reference_image_path = str(payload.get("reference_image_path") or payload.get("image_path") or "").strip()
    plan = apply_payload_generation_preset(
        GenerationPlan(
            prompt=message,
            negative_prompt=str(payload.get("negative_prompt") or ""),
            reference_image_path=reference_image_path,
        ),
        payload,
    )
    return response_to_dict(conductor.handle_generation_plan(message, plan))


def classify_auto_intent(payload: dict[str, Any], *, message: str, language: str) -> dict[str, Any]:
    classifier_message = (
        "Classify this GemmAnima user message for routing. Return JSON only.\n"
        f"attached_image: {bool(str(payload.get('image_path') or '').strip())}\n"
        f"user_message: {message}"
    )
    result = run_tipo_text_chat(
        message=classifier_message,
        language=language,
        chat_mode="intent_classification",
        history=[
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in payload.get("history", ())
            if isinstance(item, dict)
        ],
    )
    parsed = _parse_intent_classifier_result(result)
    parsed["classifier_status"] = result.get("status")
    parsed["classifier_seconds"] = result.get("seconds")
    return parsed


def _parse_intent_classifier_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "completed":
        return {
            "intent": "fallback",
            "confidence": 0.0,
            "reason": str(result.get("error") or "intent classifier failed"),
        }
    raw = str(result.get("raw") or result.get("message") or "")
    data = _first_json_object(raw)
    intent = str(data.get("intent") or "").strip().lower()
    aliases = {
        "image": "generate_image",
        "generation": "generate_image",
        "generate": "generate_image",
        "tag": "tag_image",
        "tags": "tag_image",
    }
    intent = aliases.get(intent, intent)
    if intent not in {"chat", "generate_image", "tag_image"}:
        intent = "fallback"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "intent": intent,
        "confidence": confidence,
        "reason": str(data.get("reason") or "").strip(),
    }


def _first_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = raw.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def handle_tag_payload(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        return {"error": "image_path is required for tag task", "status": "failed"}
    prompt = str(payload.get("message") or "").strip() or None
    result = run_tipo_vision_tag(image_path=image_path, prompt=prompt)
    status = result.get("status", "failed")
    if status != "completed":
        return {
            "mode": "tag_image",
            "status": "failed",
            "error": result.get("error", "tag generation failed"),
            "tags": result.get("tags", ""),
            "progress": ["route:tag", "tipo:failed"],
        }
    tags = clean_vision_tags(str(result.get("tags") or ""))
    return {
        "mode": "tag_image",
        "status": "completed",
        "message": tags,
        "tags": tags,
        "raw": result.get("raw", ""),
        "stderr_tail": result.get("stderr_tail", ""),
        "seconds": result.get("seconds"),
        "model": result.get("model"),
        "mmproj": result.get("mmproj"),
        "device": result.get("device"),
        "prompt": prompt,
        "manifest_path": None,
        "output_path": None,
        "progress": ["route:tag", "tipo:vision"],
    }


def _should_resume_generation_from_history(payload: dict[str, Any], conductor: GemmAnimaConductor) -> bool:
    history = [item for item in payload.get("history", ()) if isinstance(item, dict)]
    has_prior_image_request = any(
        str(item.get("role", "")).strip() == "user"
        and conductor.planner.is_image_request(str(item.get("content", "")))
        for item in history
    )
    has_clarification_prompt = any(
        str(item.get("role", "")).strip() == "assistant"
        and _looks_like_generation_clarification(str(item.get("content", "")))
        for item in history
    )
    return has_prior_image_request and has_clarification_prompt


def _looks_like_generation_clarification(text: str) -> bool:
    lowered = text.lower()
    return (
        "should i" in lowered
        or "preserve" in lowered
        or "change it" in lowered
        or "reference has" in lowered
        or "충돌" in text
        or "유지" in text
        or "변경" in text
    )


def handle_health_payload() -> dict[str, Any]:
    tipo_text = tipo_text_health()
    tipo_vision = tipo_vision_health()
    issues = []
    issues.extend(tipo_text.get("issues", ()))
    issues.extend(tipo_vision.get("issues", ()))
    return {
        "status": "ok",
        "ready": not issues,
        "preflight": {
            "ready": not issues,
            "blocking": bool(issues),
            "issues": issues,
        },
        "models": ModelRegistry().health(),
        "bridge_profiles": bridge_profile_options(),
        "hiddenstage_bridge": HiddenStageExit().audit_bridge().to_json_dict(),
        "renderers": audit_renderer_backend(),
        "generation_presets": generation_preset_options(),
        "resolution_presets": resolution_preset_options(),
        "samplers": sampler_options(),
        "schedulers": scheduler_options(),
        "tipo_text": tipo_text,
        "tipo_vision": tipo_vision,
    }
