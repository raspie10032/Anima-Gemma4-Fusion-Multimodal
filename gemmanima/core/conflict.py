from __future__ import annotations

from gemmanima.core.protocol import ConflictItem, ConflictReport
from gemmanima.core.schemas import ContextCapsule, GenerationPlan


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

OUTFIT_TERMS = (
    "blue cloak",
    "red dress",
    "green outfit",
    "white dress",
    "black coat",
)

STYLE_TERMS = (
    "anime illustration",
    "line art",
    "photorealistic",
    "watercolor",
    "oil painting",
)

UNSAFE_MINOR_TERMS = ("minor", "child", "underage")
UNSAFE_SEXUAL_TERMS = ("sexualized", "nude", "nsfw", "erotic")

FIELD_ALIASES = {
    "hair_color": ("hair color", "hair"),
    "outfit": ("outfit", "clothes", "clothing", "dress", "cloak"),
    "style": ("style", "art style", "rendering style"),
    "identity": ("identity", "character", "person"),
}


class ConflictResolver:
    def resolve(self, capsule: ContextCapsule, plan: GenerationPlan) -> ConflictReport:
        reference_text = " ".join(capsule.visual_facts).lower()
        request_text = capsule.user_request.lower()
        unsafe = self._unsafe_content_conflict(request_text)
        if unsafe:
            return unsafe

        conflicts: list[ConflictItem] = []
        modify: list[str] = []
        questions: list[str] = []

        reference_hair = self._hair_color(reference_text)
        requested_hair = self._hair_color(request_text)
        if (
            reference_hair
            and requested_hair
            and reference_hair != requested_hair
            and "hair_color" in capsule.protocol.reference.preserve
        ):
            resolution = self._resolution_for_field(capsule, "hair_color", requested_hair)
            if resolution == "modify":
                modify.append(f"hair_color_to_{requested_hair}")
            if resolution:
                pass
            else:
                self._add_conflict(
                    conflicts,
                    modify,
                    questions,
                    field="hair_color",
                    image_value=reference_hair,
                    text_value=requested_hair,
                    modify_value=f"hair_color_to_{requested_hair}",
                    question=(
                        f"The reference has {reference_hair} hair, but the request asks for "
                        f"{requested_hair} hair. Should I preserve the reference hair color or change it?"
                    ),
                )

        reference_outfit = self._first_term(reference_text, OUTFIT_TERMS)
        requested_outfit = self._first_term(request_text, OUTFIT_TERMS)
        if (
            reference_outfit
            and requested_outfit
            and reference_outfit != requested_outfit
            and "outfit" in capsule.protocol.reference.preserve
        ):
            modify_value = f"outfit_to_{requested_outfit.replace(' ', '_')}"
            resolution = self._resolution_for_field(capsule, "outfit", requested_outfit)
            if resolution == "modify":
                modify.append(modify_value)
            if resolution:
                pass
            else:
                self._add_conflict(
                    conflicts,
                    modify,
                    questions,
                    field="outfit",
                    image_value=reference_outfit,
                    text_value=requested_outfit,
                    modify_value=modify_value,
                    question=(
                        f"The reference outfit is {reference_outfit}, but the request asks for "
                        f"{requested_outfit}. Should I preserve the reference outfit or change it?"
                    ),
                )

        reference_style = self._first_term(reference_text, STYLE_TERMS)
        requested_style = self._first_term(request_text, STYLE_TERMS)
        if (
            reference_style
            and requested_style
            and reference_style != requested_style
            and "style" in capsule.protocol.reference.preserve
        ):
            modify_value = f"style_to_{requested_style.replace(' ', '_')}"
            resolution = self._resolution_for_field(capsule, "style", requested_style)
            if resolution == "modify":
                modify.append(modify_value)
            if resolution:
                pass
            else:
                self._add_conflict(
                    conflicts,
                    modify,
                    questions,
                    field="style",
                    image_value=reference_style,
                    text_value=requested_style,
                    modify_value=modify_value,
                    question=(
                        f"The reference style is {reference_style}, but the request asks for "
                        f"{requested_style}. Should I preserve the reference style or change it?"
                    ),
                )

        identity = capsule.protocol.character.identity
        requested_identity = self._requested_identity(request_text)
        if identity and requested_identity and identity != requested_identity:
            modify_value = f"identity_to_{requested_identity}"
            resolution = self._resolution_for_field(capsule, "identity", requested_identity)
            if resolution == "modify":
                modify.append(modify_value)
            if resolution:
                pass
            else:
                self._add_conflict(
                    conflicts,
                    modify,
                    questions,
                    field="identity",
                    image_value=identity,
                    text_value=requested_identity,
                    modify_value=modify_value,
                    question=(
                        f"The reference identity is {identity}, but the request asks for "
                        f"{requested_identity}. Should I preserve the reference identity or change it?"
                    ),
                )

        if conflicts:
            return ConflictReport(
                preserve=tuple(capsule.protocol.reference.preserve),
                modify=tuple(modify),
                conflicts=tuple(conflicts),
                proposed_questions=tuple(questions),
            )
        return ConflictReport()

    def _hair_color(self, text: str) -> str:
        for color in HAIR_COLORS:
            if f"{color} hair" in text or f"{color}-haired" in text:
                return color
        return ""

    def _first_term(self, text: str, terms: tuple[str, ...]) -> str:
        for term in terms:
            if term in text:
                return term
        return ""

    def _requested_identity(self, text: str) -> str:
        for marker in ("draw ", "make her ", "make him "):
            if marker in text:
                after = text.split(marker, 1)[1].strip()
                candidate = after.split()[0].strip(" ,.").lower()
                if candidate and candidate not in {"the", "same", "a", "an"}:
                    return candidate
        return ""

    def _unsafe_content_conflict(self, text: str) -> ConflictReport | None:
        has_minor = any(term in text for term in UNSAFE_MINOR_TERMS)
        has_sexual = any(term in text for term in UNSAFE_SEXUAL_TERMS)
        if not (has_minor and has_sexual):
            return None
        return ConflictReport(
            conflicts=(
                ConflictItem(
                    field="unsafe_content",
                    image_value="safety_policy",
                    text_value="sexualized_minor",
                    severity="critical",
                ),
            ),
            proposed_questions=(
                "This request has an unresolved unsafe-content risk, so generation cannot continue.",
            ),
        )

    def _resolution_for_field(self, capsule: ContextCapsule, field: str, requested_value: str) -> str:
        aliases = FIELD_ALIASES.get(field, (field.replace("_", " "),))
        for statement in capsule.user_preferences + capsule.visual_facts:
            text = statement.lower()
            field_hint = any(alias in text for alias in aliases) or requested_value in text
            if not field_hint:
                continue
            preserve_hint = any(
                token in text
                for token in ("preserve", "keep", "same as reference", "use reference", "reference value")
            )
            change_hint = any(
                token in text
                for token in ("change", "change it", "change to", "use the request", "use requested", "apply requested")
            )

            if preserve_hint and not change_hint:
                return "preserve"
            if change_hint:
                return "modify"
        return ""

    def _add_conflict(
        self,
        conflicts: list[ConflictItem],
        modify: list[str],
        questions: list[str],
        *,
        field: str,
        image_value: str,
        text_value: str,
        modify_value: str,
        question: str,
    ) -> None:
        conflicts.append(
            ConflictItem(
                field=field,
                image_value=image_value,
                text_value=text_value,
                severity="high",
            )
        )
        modify.append(modify_value)
        questions.append(question)
