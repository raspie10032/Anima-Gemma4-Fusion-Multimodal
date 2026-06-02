from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from generator import BuiltinImageConfig, compile_image_prompt, create_builtin_generation_job
from model_prototype import DEFAULT_CHAT_CLI, DEFAULT_CHAT_MODEL, DEFAULT_PLANNER_MODEL, DEFAULT_TAG_CLI, DEFAULT_TAG_MMPROJ, DEFAULT_TAG_MODEL
from model_prototype import model_health
from purpose import project_contract

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHAT_PROMPT = (
    "You are a concise Korean/English assistant. Answer naturally. "
    "Do not output image tags unless explicitly asked."
)
DEFAULT_TAG_PROMPT = (
    "Output ONLY a comma-separated list of English Danbooru tags for this image. "
    "No thinking, no explanation, no sentences. Capture the image and plausibly "
    "expand it TIPO style. About 50 tags."
)


Runner = Callable[..., Any]
Timer = Callable[[], float]


@dataclass(frozen=True)
class ChatConfig:
    cli: Path = DEFAULT_CHAT_CLI
    model: Path = DEFAULT_CHAT_MODEL
    visible_devices: str = "0"
    device: str = "CUDA0"
    max_new_tokens: int = 256
    temperature: float = 0.7
    timeout_seconds: int = 300
    seed: int = 42
    chat_template: str = "gemma"


@dataclass(frozen=True)
class TagConfig:
    cli: Path = DEFAULT_TAG_CLI
    model: Path = DEFAULT_TAG_MODEL
    mmproj: Path = DEFAULT_TAG_MMPROJ
    visible_devices: str = "0"
    device: str = "CUDA0"
    max_new_tokens: int = 140
    temperature: float = 0.7
    timeout_seconds: int = 300
    seed: int = 42
    chat_template: str = "gemma"


@dataclass(frozen=True)
class PlannerConfig:
    cli: Path = DEFAULT_CHAT_CLI
    model: Path = DEFAULT_PLANNER_MODEL
    visible_devices: str = "0"
    device: str = "CUDA0"
    max_new_tokens: int = 96
    temperature: float = 0.7
    timeout_seconds: int = 180
    seed: int = 42


def health() -> dict[str, Any]:
    payload = model_health()
    payload["project_contract"] = project_contract()
    return payload


def route_request(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload.get("task") or "auto").strip().lower()
    image_path = str(payload.get("image_path") or payload.get("image") or "").strip()
    message = str(payload.get("message") or "").strip()
    history = payload.get("history") if isinstance(payload.get("history"), list) else []

    if task in {"image", "generate", "txt2img", "text_to_image"} or (
        task == "auto" and _looks_like_image_request(message)
    ):
        if not message:
            return {"mode": "image", "status": "failed", "error": "message is required"}
        return create_image_job(
            message=message,
            style=str(payload.get("style") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
        )

    if task in {"tag", "tag_image", "vision_tag"} or (task == "auto" and image_path):
        if not image_path:
            return {"mode": "tag", "status": "failed", "error": "image_path is required"}
        return tag_image(image_path=image_path, prompt=message or None)

    if task in {"chat", "talk", "auto"}:
        if not message:
            return {"mode": "chat", "status": "failed", "error": "message is required"}
        return chat(message=message, history=history)

    return {"mode": task, "status": "failed", "error": f"unknown task: {task}"}


def create_image_job(
    *,
    message: str,
    style: str = "",
    negative_prompt: str = "",
    config: BuiltinImageConfig | None = None,
) -> dict[str, Any]:
    if config:
        return create_builtin_generation_job(message=message, config=config)
    return create_builtin_generation_job(
        message=message,
        style=style or BuiltinImageConfig.style,
        negative_prompt=negative_prompt or BuiltinImageConfig.negative_prompt,
    )


def chat(
    *,
    message: str,
    history: Iterable[dict[str, str]] = (),
    config: ChatConfig | None = None,
    runner: Runner = subprocess.run,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    cfg = config or ChatConfig()
    missing = _missing({"cli": cfg.cli, "model": cfg.model})
    if missing:
        return _failed("chat", "; ".join(missing), model=str(cfg.model), device=cfg.device)

    prompt = _chat_prompt(message, history)
    cmd = [
        str(cfg.cli),
        "-m",
        str(cfg.model),
        "--chat-template",
        cfg.chat_template,
        "--single-turn",
        "--simple-io",
        "--reasoning",
        "off",
        "--no-display-prompt",
        "--no-warmup",
        "-dev",
        cfg.device,
        "-ngl",
        "auto",
        "-n",
        str(cfg.max_new_tokens),
        "--temp",
        str(cfg.temperature),
        "--seed",
        str(cfg.seed),
        "-p",
        prompt,
    ]
    return _run_text_command("chat", cmd, cfg.visible_devices, cfg.timeout_seconds, runner, timer, str(cfg.model), cfg.device)


def tag_image(
    *,
    image_path: str | Path,
    prompt: str | None = None,
    config: TagConfig | None = None,
    runner: Runner = subprocess.run,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    cfg = config or TagConfig()
    image = Path(image_path)
    missing = _missing({"cli": cfg.cli, "model": cfg.model, "mmproj": cfg.mmproj, "image": image})
    if missing:
        return _failed("tag", "; ".join(missing), model=str(cfg.model), mmproj=str(cfg.mmproj), device=cfg.device)

    tag_prompt = (prompt or DEFAULT_TAG_PROMPT).strip() or DEFAULT_TAG_PROMPT
    cmd = [
        str(cfg.cli),
        "-m",
        str(cfg.model),
        "--mmproj",
        str(cfg.mmproj),
        "--image",
        str(image),
        "--chat-template",
        cfg.chat_template,
        "--no-warmup",
        "-dev",
        cfg.device,
        "-ngl",
        "auto",
        "-n",
        str(cfg.max_new_tokens),
        "--temp",
        str(cfg.temperature),
        "--seed",
        str(cfg.seed),
        "-p",
        tag_prompt,
    ]
    result = _run_text_command("tag", cmd, cfg.visible_devices, cfg.timeout_seconds, runner, timer, str(cfg.model), cfg.device)
    result["tags"] = _parse_text(result.get("raw", ""))
    result["message"] = result["tags"]
    result["mmproj"] = str(cfg.mmproj)
    return result


def _looks_like_image_request(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "image",
        "picture",
        "draw",
        "generate",
        "txt2img",
        "그림",
        "이미지",
        "생성",
        "그려",
        "만들어",
        "뽑아",
    )
    return any(marker in lowered for marker in markers)


def _run_text_command(
    mode: str,
    cmd: list[str],
    visible_devices: str,
    timeout_seconds: int,
    runner: Runner,
    timer: Timer,
    model: str,
    device: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    start = timer()
    try:
        proc = runner(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _failed(mode, f"timeout after {timeout_seconds}s", model=model, device=device)
    seconds = round(timer() - start, 3)
    raw = proc.stdout or ""
    stderr = proc.stderr or ""
    text = _parse_text(raw)
    status = "completed" if proc.returncode == 0 else "failed"
    return {
        "mode": mode,
        "status": status,
        "message": text,
        "raw": raw,
        "stderr_tail": stderr[-2000:],
        "seconds": seconds,
        "model": model,
        "device": device,
        "error": "" if status == "completed" else (stderr or raw)[-1000:],
    }


def _chat_prompt(message: str, history: Iterable[dict[str, str]]) -> str:
    lines = [DEFAULT_CHAT_PROMPT]
    for turn in list(history)[-8:]:
        role = str(turn.get("role", "")).strip() or "user"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"user: {message.strip()}")
    lines.append("assistant:")
    return "\n".join(lines)


def _parse_text(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip().strip("`").strip()
        if not stripped:
            continue
        if stripped.lower().startswith(("main:", "load:", "llama", "clip", "image decoded")):
            continue
        lines.append(stripped.rstrip(":."))
    return "\n".join(lines).strip()


def _missing(assets: dict[str, Path]) -> list[str]:
    return [f"missing {name}: {path}" for name, path in assets.items() if not path.is_file()]


def _failed(mode: str, error: str, **extra: Any) -> dict[str, Any]:
    return {"mode": mode, "status": "failed", "error": error, "message": "", **extra}


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
