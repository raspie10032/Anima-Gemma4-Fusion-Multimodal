from __future__ import annotations

from uuid import uuid4

from gemmanima.core.protocol_parser import ProtocolParser
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
    "forest",
    "garden",
    "character",
    "fairy",
    "light",
    "portrait",
    "illustration",
    "anime",
    "image",
    "reference",
    "ref",
)

CHAT_ONLY_TERMS = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "yes",
    "no",
    "안녕",
    "안녕하세요",
    "고마워",
    "고맙습니다",
    "응",
    "ㅇㅇ",
    "ㄴㄴ",
)

PREFERENCE_HINTS = (
    "prefer",
    "preference",
    "keep",
    "same",
    "like",
    "style",
    "color",
    "palette",
    "lighting",
    "mood",
    "pose",
    "composition",
    "camera",
    "angle",
    "hair",
    "outfit",
    "background",
    "soft",
    "cute",
    "선호",
    "좋아",
    "유지",
    "같이",
    "같은",
    "스타일",
    "색",
    "색감",
    "조명",
    "분위기",
    "포즈",
    "구도",
    "머리",
    "의상",
    "배경",
)


class ContextRelevanceFilter:
    """Builds a small generation capsule while keeping engine talk out of image prompts."""

    def __init__(self, protocol_parser: ProtocolParser | None = None) -> None:
        self.protocol_parser = protocol_parser or ProtocolParser()

    def build(self, history: list[ChatTurn], user_request: str, max_items: int = 8) -> ContextCapsule:
        visual_facts: list[str] = []
        preferences: list[str] = []
        constraints: list[str] = []
        omitted: list[str] = []

        for turn in history[-32:]:
            if turn.role != "user":
                continue
            text = turn.content.strip()
            if not text:
                continue
            lowered = text.lower()
            if any(term.lower() in lowered for term in TECHNICAL_TERMS):
                omitted.append(text)
                continue
            if _is_chat_only_text(lowered):
                omitted.append(text)
                continue
            if any(hint.lower() in lowered for hint in VISUAL_HINTS):
                visual_facts.append(text)
            elif turn.role == "user" and len(text) < 160 and any(
                hint.lower() in lowered for hint in PREFERENCE_HINTS
            ):
                preferences.append(text)
            else:
                omitted.append(text)

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
            protocol=self.protocol_parser.parse(
                user_request=user_request,
                visual_facts=tuple(visual_facts[-max_items:]),
            ),
        )


def _is_chat_only_text(lowered_text: str) -> bool:
    normalized = lowered_text.strip().strip(".!?。！？")
    if normalized in CHAT_ONLY_TERMS:
        return True
    return len(normalized) <= 4 and normalized.isascii() and normalized.isalpha()
