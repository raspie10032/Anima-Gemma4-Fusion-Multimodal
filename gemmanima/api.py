from __future__ import annotations

from pathlib import Path
from typing import Any

from gemmanima import GemmAnimaConductor
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.schemas import ChatTurn
from gemmanima.modules.hiddenstage_exit import HiddenStageExit


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


def handle_chat_payload(payload: dict[str, Any], *, base_dir: str | Path = "runs") -> dict[str, Any]:
    message = str(payload.get("message", "")).strip()
    if not message:
        return {"error": "message is required", "status": "failed"}

    session_id = payload.get("session_id")
    conductor = GemmAnimaConductor(
        session_id=str(session_id) if session_id else None,
        manifest_root=Path(base_dir) / "manifests",
        image_root=Path(base_dir) / "images",
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
    }
