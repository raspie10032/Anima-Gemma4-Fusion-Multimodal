from __future__ import annotations

import re

from gemmanima.core.protocol import (
    CharacterCapsule,
    GemmanimaProtocol,
    InstructionCapsule,
    MoodCapsule,
    ReferenceCapsule,
    SceneCapsule,
)


HAIR_COLORS = (
    "silver",
    "black",
    "white",
    "blonde",
    "brown",
    "red",
    "blue",
    "green",
    "pink",
    "purple",
)

MOOD_KEYWORDS = {
    "calm": ("calm", "", ""),
    "peaceful": ("peaceful", "", ""),
    "tense": ("tense", "", "high"),
    "dramatic": ("dramatic", "", "high"),
    "melancholic": ("melancholic", "melancholic", ""),
    "eerie": ("eerie", "eerie", "medium"),
}

LIGHTING_KEYWORDS = (
    "moonlit",
    "moonlight",
    "neon",
    "soft lighting",
    "rim light",
    "backlit",
    "golden hour",
    "low light",
)

PRESERVE_ALIASES = {
    "hair_color": ("hair color", "hair"),
    "outfit": ("outfit", "clothes", "clothing", "dress", "cloak"),
    "face_impression": ("face", "expression", "facial impression"),
    "style": ("style", "art style", "rendering style"),
    "identity": ("identity", "same character", "character"),
}


class ProtocolParser:
    def parse(self, *, user_request: str, visual_facts: tuple[str, ...] = ()) -> GemmanimaProtocol:
        reference_text = " ".join(visual_facts).lower()
        request_text = user_request.lower()
        preserve = self._dedupe(
            self.reference_preserve_fields(visual_facts) + self.explicit_preserve_fields(request_text)
        )
        mood = self.mood_fields(request_text=request_text, reference_text=reference_text)
        return GemmanimaProtocol(
            scene=SceneCapsule(
                action=user_request,
                mood=mood["emotion"],
                visual_anchors=visual_facts,
                negative_constraints=self.negative_constraints(request_text),
            ),
            character=CharacterCapsule(identity=self.reference_identity(reference_text)),
            mood=MoodCapsule(
                emotion=mood["emotion"],
                atmosphere=mood["atmosphere"],
                tension=mood["tension"],
                lighting=mood["lighting"],
            ),
            reference=ReferenceCapsule(
                source_image_id="history_visual_reference" if visual_facts else "",
                preserve=preserve,
                strength=0.6 if visual_facts else 0.0,
            ),
            instruction=InstructionCapsule(
                user_intent=user_request,
                explicit_changes=self.explicit_changes(request_text),
                preserve_requests=preserve,
                forbidden_changes=preserve,
            ),
        )

    def reference_preserve_fields(self, visual_facts: tuple[str, ...]) -> tuple[str, ...]:
        joined = " ".join(visual_facts).lower()
        fields: list[str] = []
        if "hair" in joined:
            fields.append("hair_color")
        if "cloak" in joined or "outfit" in joined or "dress" in joined:
            fields.append("outfit")
        if "face" in joined or "expression" in joined:
            fields.append("face_impression")
        if "style" in joined or "illustration" in joined or "line art" in joined or "anime" in joined:
            fields.append("style")
        if "identity" in joined or "same character" in joined:
            fields.append("identity")
        return tuple(dict.fromkeys(fields))

    def reference_identity(self, reference_text: str) -> str:
        marker = "identity "
        if marker in reference_text:
            after = reference_text.split(marker, 1)[1].strip()
            return after.split(",", 1)[0].split()[0]
        return ""

    def explicit_preserve_fields(self, request_text: str) -> tuple[str, ...]:
        preserve_requested = any(token in request_text for token in ("preserve", "keep", "same as reference"))
        if not preserve_requested:
            return ()

        fields: list[str] = []
        for field, aliases in PRESERVE_ALIASES.items():
            if any(alias in request_text for alias in aliases):
                fields.append(field)
        return tuple(fields)

    def mood_fields(self, *, request_text: str, reference_text: str) -> dict[str, str]:
        emotion = ""
        atmosphere = ""
        tension = ""
        for text in (request_text, reference_text):
            for keyword, values in MOOD_KEYWORDS.items():
                if keyword in text:
                    emotion, atmosphere, tension = values
                    break
            if emotion:
                break

        lighting = ""
        for text in (request_text, reference_text):
            for keyword in LIGHTING_KEYWORDS:
                if keyword in text:
                    lighting = "moonlit" if keyword == "moonlight" else keyword
                    break
            if lighting:
                break

        return {
            "emotion": emotion,
            "atmosphere": atmosphere,
            "tension": tension,
            "lighting": lighting,
        }

    def negative_constraints(self, request_text: str) -> tuple[str, ...]:
        constraints: list[str] = []
        for match in re.finditer(r"\b(?:no|without|avoid|exclude)\s+([^,.;]+)", request_text):
            value = self._clean_phrase(match.group(1))
            if value:
                constraints.append(value)
        return self._dedupe(tuple(constraints))

    def explicit_changes(self, request_text: str) -> tuple[str, ...]:
        changes: list[str] = []

        for color in HAIR_COLORS:
            if f"{color} hair" in request_text or f"{color}-haired" in request_text:
                changes.append(f"hair_color_to_{color}")
                break

        background = self._match_change_target(request_text, "background")
        if background:
            changes.append(f"background_to_{self._slug(background)}")

        if "photorealistic" in request_text:
            changes.append("style_to_photorealistic")

        outfit = self._match_after(request_text, "wearing")
        if outfit:
            changes.append(f"outfit_to_{self._slug(outfit)}")

        identity = self._requested_identity(request_text)
        if identity:
            changes.append(f"identity_to_{identity}")

        return self._dedupe(tuple(changes))

    def _match_change_target(self, request_text: str, field: str) -> str:
        match = re.search(rf"\bchange\s+(?:the\s+)?{field}\s+to\s+([^,.;]+)", request_text)
        if not match:
            return ""
        return self._clean_phrase(match.group(1))

    def _match_after(self, request_text: str, marker: str) -> str:
        match = re.search(rf"\b{marker}\s+([^,.;]+)", request_text)
        if not match:
            return ""
        return self._clean_phrase(match.group(1))

    def _requested_identity(self, request_text: str) -> str:
        for marker in ("draw ", "make her ", "make him "):
            if marker in request_text:
                after = request_text.split(marker, 1)[1].strip()
                candidate = after.split()[0].strip(" ,.").lower()
                if candidate and candidate not in {"the", "same", "a", "an"}:
                    return candidate
        return ""

    def _clean_phrase(self, value: str) -> str:
        value = value.strip(" ,.")
        for article in ("a ", "an ", "the "):
            if value.startswith(article):
                return value[len(article) :].strip(" ,.")
        return value

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _dedupe(self, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for value in values if value))
