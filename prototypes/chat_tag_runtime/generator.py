from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from purpose import INTERNAL_GENERATOR_ID, REQUIRED_IMAGE_ENGINE

DEFAULT_IMAGE_STYLE = "anime illustration, clean line art, detailed lighting"
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, bad anatomy, extra fingers, "
    "watermark, signature, text artifacts"
)


@dataclass(frozen=True)
class BuiltinImageConfig:
    generator_id: str = INTERNAL_GENERATOR_ID
    style: str = DEFAULT_IMAGE_STYLE
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_dir: Path = Path("runs/prototypes/chat_tag_runtime/images")


@dataclass(frozen=True)
class BuiltinGenerationJob:
    job_id: str
    prompt: str
    negative_prompt: str
    planner_prompt: str
    output_path: Path

    def to_json_dict(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "planner_prompt": self.planner_prompt,
            "output_path": str(self.output_path.as_posix()),
        }


class BuiltinImageGenerator:
    generator_id = INTERNAL_GENERATOR_ID

    def generate(self, job: BuiltinGenerationJob) -> dict[str, Any]:
        return {
            "mode": "image",
            "status": "engine_missing",
            "generator": self.generator_id,
            "required_engine": REQUIRED_IMAGE_ENGINE,
            "required_conditioning": "Gemma4 hidden states",
            "external_backend": False,
            "job": job.to_json_dict(),
            "error": "Gemma4 hidden-state to Anima synthesis bridge is not implemented yet",
        }


def create_builtin_generation_job(
    *,
    message: str,
    style: str = DEFAULT_IMAGE_STYLE,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    config: BuiltinImageConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BuiltinImageConfig(style=style, negative_prompt=negative_prompt)
    job = build_generation_job(message=message, config=cfg)
    return {
        "mode": "image",
        "status": "generator_required",
        "generator": cfg.generator_id,
        "external_backend": False,
        "message": "builtin image generator is not wired yet",
        "job_id": job.job_id,
        "prompt": job.prompt,
        "planner_prompt": job.planner_prompt,
        "planner_contract": "TIPO Partial tags continuation",
        "negative_prompt": job.negative_prompt,
        "source_message": message.strip(),
        "output_path": str(job.output_path.as_posix()),
        "next_engine_slot": "connect Gemma4 hidden states directly into Anima/GEMMANIMA synthesis",
    }


def build_generation_job(*, message: str, config: BuiltinImageConfig | None = None) -> BuiltinGenerationJob:
    cfg = config or BuiltinImageConfig()
    prompt = compile_image_prompt(message=message, style=cfg.style)
    planner_prompt = compile_tipo_planner_prompt(message=message)
    job_id = _job_id(prompt)
    return BuiltinGenerationJob(
        job_id=job_id,
        prompt=prompt,
        negative_prompt=cfg.negative_prompt,
        planner_prompt=planner_prompt,
        output_path=cfg.output_dir / f"{job_id}.png",
    )


def build_generation_job_with_planner_tags(
    *,
    message: str,
    planner_tags: list[str],
    config: BuiltinImageConfig | None = None,
) -> BuiltinGenerationJob:
    from planner import merge_user_prompt_and_planner_tags

    cfg = config or BuiltinImageConfig()
    prompt = merge_user_prompt_and_planner_tags(
        user_prompt=message,
        planner_tags=planner_tags,
        style=cfg.style,
    )
    planner_prompt = compile_tipo_planner_prompt(message=message)
    job_id = _job_id(prompt)
    return BuiltinGenerationJob(
        job_id=job_id,
        prompt=prompt,
        negative_prompt=cfg.negative_prompt,
        planner_prompt=planner_prompt,
        output_path=cfg.output_dir / f"{job_id}.png",
    )


def compile_image_prompt(*, message: str, style: str = DEFAULT_IMAGE_STYLE) -> str:
    subject = " ".join(message.strip().split())
    style_text = " ".join(style.strip().split()) if style.strip() else DEFAULT_IMAGE_STYLE
    if not subject:
        subject = "a character portrait"
    if "," in subject:
        return f"{subject}, {style_text}"
    return f"{subject}, {style_text}, high detail, coherent composition"


def compile_tipo_planner_prompt(*, message: str) -> str:
    subject = " ".join(message.strip().split()) or "character portrait"
    if "," in subject:
        return f"rating: safe, {subject}, Partial tags:"
    return f"rating: safe, {subject}, detailed, anime style, Partial tags:"


def _job_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
