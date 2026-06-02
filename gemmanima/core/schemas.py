from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from gemmanima.core.protocol import PROTOCOL_VERSION, GemmanimaProtocol


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
    protocol: GemmanimaProtocol = field(default_factory=GemmanimaProtocol)


@dataclass(frozen=True)
class GenerationPlan:
    prompt: str
    negative_prompt: str = ""
    reference_image_path: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 28
    cfg: float = 4.0
    seed: int | None = None
    sampler: str = "euler_ancestral"
    scheduler: str = "sgm_uniform"
    lora_stack: tuple[str, ...] = ()
    renderer_profile: str = "anima_fp16_final"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationPlan":
        allowed = {
            "prompt",
            "negative_prompt",
            "reference_image_path",
            "width",
            "height",
            "steps",
            "cfg",
            "seed",
            "sampler",
            "scheduler",
            "lora_stack",
            "renderer_profile",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown generation plan keys: {', '.join(unknown)}")
        plan = cls(
            prompt=str(data.get("prompt", "")),
            negative_prompt=str(data.get("negative_prompt", "")),
            reference_image_path=str(data.get("reference_image_path") or ""),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 28)),
            cfg=float(data.get("cfg", 4.0)),
            seed=None if data.get("seed") is None else int(data["seed"]),
            sampler=str(data.get("sampler", "euler_ancestral")),
            scheduler=str(data.get("scheduler", "sgm_uniform")),
            lora_stack=tuple(str(item) for item in data.get("lora_stack", ())),
            renderer_profile=str(data.get("renderer_profile", "anima_fp16_final")),
        )
        plan.validate()
        return plan

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "reference_image_path": self.reference_image_path,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg": self.cfg,
            "seed": self.seed,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
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
        if not self.sampler.strip():
            raise ValueError("sampler must be non-empty")
        if not self.scheduler.strip():
            raise ValueError("scheduler must be non-empty")


@dataclass(frozen=True)
class ConditioningBundle:
    source: str
    shape: tuple[int, int, int] = (1, 512, 1024)
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION
    semantic_conditioning: dict[str, Any] = field(default_factory=dict)
    reference_conditioning: dict[str, Any] = field(default_factory=dict)
    style_conditioning: dict[str, Any] = field(default_factory=dict)
    mood_conditioning: dict[str, Any] = field(default_factory=dict)
    spatial_conditioning: dict[str, Any] = field(default_factory=dict)
    negative_conditioning: dict[str, Any] = field(default_factory=dict)
    lora_hints: tuple[str, ...] = ()
    strength_weights: dict[str, Any] = field(default_factory=dict)
    conflict_state: dict[str, Any] = field(default_factory=dict)
    renderer_profile: str = ""

    def validate(self) -> None:
        if self.shape != (1, 512, 1024):
            raise ValueError(f"expected conditioning shape (1, 512, 1024), got {self.shape}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditioningBundle":
        allowed = {
            "source",
            "shape",
            "metadata",
            "protocol_version",
            "semantic_conditioning",
            "reference_conditioning",
            "style_conditioning",
            "mood_conditioning",
            "spatial_conditioning",
            "negative_conditioning",
            "lora_hints",
            "strength_weights",
            "conflict_state",
            "renderer_profile",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown conditioning bundle keys: {', '.join(unknown)}")
        bundle = cls(
            source=str(data.get("source", "")),
            shape=tuple(int(item) for item in data.get("shape", (1, 512, 1024))),  # type: ignore[arg-type]
            metadata=dict(data.get("metadata") or {}),
            protocol_version=str(data.get("protocol_version") or PROTOCOL_VERSION),
            semantic_conditioning=dict(data.get("semantic_conditioning") or {}),
            reference_conditioning=dict(data.get("reference_conditioning") or {}),
            style_conditioning=dict(data.get("style_conditioning") or {}),
            mood_conditioning=dict(data.get("mood_conditioning") or {}),
            spatial_conditioning=dict(data.get("spatial_conditioning") or {}),
            negative_conditioning=dict(data.get("negative_conditioning") or {}),
            lora_hints=tuple(str(item) for item in data.get("lora_hints", ())),
            strength_weights=dict(data.get("strength_weights") or {}),
            conflict_state=dict(data.get("conflict_state") or {}),
            renderer_profile=str(data.get("renderer_profile") or ""),
        )
        bundle.validate()
        return bundle

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "shape": list(self.shape),
            "metadata": self.metadata,
            "protocol_version": self.protocol_version,
            "semantic_conditioning": self.semantic_conditioning,
            "reference_conditioning": self.reference_conditioning,
            "style_conditioning": self.style_conditioning,
            "mood_conditioning": self.mood_conditioning,
            "spatial_conditioning": self.spatial_conditioning,
            "negative_conditioning": self.negative_conditioning,
            "lora_hints": list(self.lora_hints),
            "strength_weights": self.strength_weights,
            "conflict_state": self.conflict_state,
            "renderer_profile": self.renderer_profile,
        }


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
    plan: GenerationPlan | None = None
    manifest_path: Path | None = None
    output_path: Path | None = None
    progress: tuple[str, ...] = ()
    clarification_required: bool = False
    conflict: dict[str, Any] | None = None
    job_id: str = field(default_factory=lambda: uuid4().hex)
