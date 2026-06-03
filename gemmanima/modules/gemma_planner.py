from __future__ import annotations

from dataclasses import dataclass
import re

from gemmanima.core.config import EngineConfig
from gemmanima.core.model_paths import model_path
from gemmanima.core.schemas import ContextCapsule, GenerationPlan


IMAGE_REQUEST_TERMS = (
    "그려",
    "그림",
    "이미지",
    "만들어줘",
    "생성",
    "draw",
    "image",
    "generate",
    "render",
)

IMAGE_REQUEST_PATTERNS = (
    r"\b(draw|generate|render|create|make)\b.+\b(image|picture|illustration|anime|art|scene|portrait)\b",
    r"\b(image|picture|illustration|anime|art|scene|portrait)\b.+\b(draw|generate|render|create|make)\b",
    r"\b(draw|generate|render)\b\s+.+",
    r"[\uadf8\ub9bc\uc744\uc774]?\s*\uadf8\ub824\s*(\uc918|\uc8fc|\uc8fc\uc138\uc694|\ub2ec\ub77c|\ubd10)",
    r"(\uc774\ubbf8\uc9c0|\uadf8\ub9bc)\s*(\uc0dd\uc131|\ub9cc\ub4e4)\s*(\ud574|\ud574\uc918|\ud574\uc8fc|\ud574\uc8fc\uc138\uc694)",
    r"(\uc774\ubbf8\uc9c0|\uadf8\ub9bc)(\uc744|\ub97c)?\s*(\ub9cc\ub4e4\uc5b4|\uc0dd\uc131\ud574|\uadf8\ub824)\s*(\uc918|\uc8fc|\uc8fc\uc138\uc694)?",
    r"\uc774\ubbf8\uc9c0\ub85c\s*(\ub9cc\ub4e4\uc5b4|\uc0dd\uc131\ud574)\s*(\uc918|\uc8fc|\uc8fc\uc138\uc694)?",
)

VAGUE_IMAGE_REQUESTS = {
    "\uadf8\ub824\uc918",
    "\uc774\ubbf8\uc9c0\ub85c",
    "\uc774\ubbf8\uc9c0\ub85c \ub9cc\ub4e4\uc5b4\uc918",
    "\uadf8\uac78 \uadf8\ub824\uc918",
    "\uadf8\uac70 \uadf8\ub824\uc918",
}

NEGATIVE_IMAGE_REQUEST_PATTERNS = (
    r"\bnot\s+(an?\s+)?image\s+request\b",
    r"\bdo\s+not\s+(draw|generate|render|create|make)\b",
    r"\b(don't|dont)\s+(draw|generate|render|create|make)\b",
    r"(\uc774\ubbf8\uc9c0|\uadf8\ub9bc)\s*\uc694\uccad\uc774\s*\uc544\ub2c8",
    r"(\uc774\ubbf8\uc9c0|\uadf8\ub9bc)\s*(\uc0dd\uc131|\uc694\uccad)\s*(\ub9d0\uace0|\uc544\ub2c8)",
    r"(\uadf8\ub824|\uc0dd\uc131|generate|draw|render).{0,12}\s*(\ub9d0\uace0|\uc544\ub2c8)",
)


@dataclass(frozen=True)
class PlannerArtifacts:
    adapter_model: str = str(model_path("hiddenstage_bridge", "hiddenstage-planner-adapter.safetensors"))
    vision_embedding: str = str(model_path("hiddenstage_bridge", "hiddenstage-planner-embed-vision.pt"))
    eval_loss: float = 1.0061092711985111
    eval_threshold: float = 1.5


class GemmaPlannerAdapter:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def is_image_request(self, text: str) -> bool:
        lowered = text.lower()
        if any(re.search(pattern, lowered) for pattern in NEGATIVE_IMAGE_REQUEST_PATTERNS):
            return False
        if any(re.search(pattern, lowered) for pattern in IMAGE_REQUEST_PATTERNS):
            return True
        return lowered.strip() in VAGUE_IMAGE_REQUESTS

    def needs_clarification(self, capsule: ContextCapsule) -> bool:
        request = capsule.user_request.strip()
        has_subject = len(request) >= 12 or capsule.visual_facts or capsule.user_preferences
        vague_only = request in {"그려줘", "이미지로", "이미지로 만들어줘", "그걸 그려줘", "그거 그려줘"}
        return vague_only and not has_subject

    def clarification_question(self, capsule: ContextCapsule) -> str:
        return "어떤 대상을 중심으로 그릴까?"

    def make_plan(self, capsule: ContextCapsule) -> GenerationPlan:
        parts: list[str] = []
        if capsule.visual_facts:
            parts.extend(capsule.visual_facts[-3:])
        if capsule.user_preferences:
            parts.extend(capsule.user_preferences[-2:])
        parts.append(capsule.user_request)
        if capsule.generation_constraints:
            parts.extend(capsule.generation_constraints)

        prompt = ", ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))
        profile = self.config.profile("anima_fp16_final")
        return GenerationPlan(
            prompt=prompt,
            negative_prompt="low quality, distorted anatomy, unreadable text",
            width=profile.width,
            height=profile.height,
            steps=profile.steps,
            cfg=profile.cfg,
            renderer_profile=profile.name,
        )
