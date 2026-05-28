from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class Mode(str, Enum):
    CHAT = "chat"
    GENERATE_IMAGE = "generate_image"
    EDIT_IMAGE = "edit_image"


class JobStatus(str, Enum):
    PLANNED = "planned"
    ASK_CLARIFY = "ask_clarify"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class ContextCapsule:
    capsule_id: str
    user_request: str
    visual_facts: tuple[str, ...] = ()
    user_preferences: tuple[str, ...] = ()
    generation_constraints: tuple[str, ...] = ()
    omitted_technical_context: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationPlan:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 28
    cfg: float = 4.0
    seed: int | None = None
    lora_stack: tuple[str, ...] = ()
    renderer_profile: str = "anima_fp16_final"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationPlan":
        allowed = {
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
            "cfg",
            "seed",
            "lora_stack",
            "renderer_profile",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown generation plan keys: {', '.join(unknown)}")
        plan = cls(
            prompt=str(data.get("prompt", "")),
            negative_prompt=str(data.get("negative_prompt", "")),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 28)),
            cfg=float(data.get("cfg", 4.0)),
            seed=None if data.get("seed") is None else int(data["seed"]),
            lora_stack=tuple(str(item) for item in data.get("lora_stack", ())),
            renderer_profile=str(data.get("renderer_profile", "anima_fp16_final")),
        )
        plan.validate()
        return plan

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg": self.cfg,
            "seed": self.seed,
            "lora_stack": list(self.lora_stack),
            "renderer_profile": self.renderer_profile,
        }

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("generation plan requires a non-empty prompt")
        if self.width < 256 or self.height < 256:
            raise ValueError("width and height must be at least 256")
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be multiples of 8")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.cfg <= 0:
            raise ValueError("cfg must be positive")


@dataclass(frozen=True)
class ConditioningBundle:
    source: str
    shape: tuple[int, int, int] = (1, 512, 1024)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.shape != (1, 512, 1024):
            raise ValueError(f"expected conditioning shape (1, 512, 1024), got {self.shape}")


@dataclass(frozen=True)
class RenderResult:
    image_id: str
    output_path: Path
    seed: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EngineResponse:
    mode: Mode
    status: JobStatus
    message: str
    prompt: str | None = None
    manifest_path: Path | None = None
    output_path: Path | None = None
    progress: tuple[str, ...] = ()
    job_id: str = field(default_factory=lambda: uuid4().hex)
