from __future__ import annotations

from uuid import uuid4

from gemmanima.core.schemas import ChatTurn, ContextCapsule


TECHNICAL_TERMS = (
    "gpu",
    "vram",
    "4070",
    "5060",
    "checkpoint",
    "eval_loss",
    "hiddenstage",
    "backend",
    "학습",
    "체크포인트",
    "그래픽카드",
    "전력",
    "일반 채팅 경로",
    "이미지 컴포넌트",
    "dry-run",
    "manifest",
)

VISUAL_HINTS = (
    "그림",
    "이미지",
    "장면",
    "캐릭터",
    "배경",
    "스타일",
    "색감",
    "구도",
    "portrait",
    "illustration",
    "anime",
)


class ContextRelevanceFilter:
    """Builds a small generation capsule while keeping engine talk out of image prompts."""

    def build(self, history: list[ChatTurn], user_request: str, max_items: int = 8) -> ContextCapsule:
        visual_facts: list[str] = []
        preferences: list[str] = []
        constraints: list[str] = []
        omitted: list[str] = []

        for turn in history[-32:]:
            text = turn.content.strip()
            if not text:
                continue
            lowered = text.lower()
            if any(term.lower() in lowered for term in TECHNICAL_TERMS):
                omitted.append(text)
                continue
            if any(hint.lower() in lowered for hint in VISUAL_HINTS):
                visual_facts.append(text)
            elif turn.role == "user" and len(text) < 160:
                preferences.append(text)

        if "세로" in user_request or "portrait" in user_request.lower():
            constraints.append("portrait orientation preferred")
        if "가로" in user_request or "landscape" in user_request.lower():
            constraints.append("landscape orientation preferred")

        return ContextCapsule(
            capsule_id=uuid4().hex,
            user_request=user_request,
            visual_facts=tuple(visual_facts[-max_items:]),
            user_preferences=tuple(preferences[-max_items:]),
            generation_constraints=tuple(constraints[-max_items:]),
            omitted_technical_context=tuple(omitted[-max_items:]),
        )
