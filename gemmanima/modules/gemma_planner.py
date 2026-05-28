from __future__ import annotations

from dataclasses import dataclass

from gemmanima.core.config import EngineConfig
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


@dataclass(frozen=True)
class PlannerArtifacts:
    adapter_model: str = r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\adapter_model.safetensors"
    vision_embedding: str = r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\embed_vision.pt"
    eval_loss: float = 1.0061092711985111
    eval_threshold: float = 1.5


class GemmaPlannerAdapter:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def is_image_request(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in IMAGE_REQUEST_TERMS)

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
