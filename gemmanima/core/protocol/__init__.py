from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = "0.1"

_SEVERITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def _list(values: tuple[str, ...]) -> list[str]:
    return list(values)


@dataclass(frozen=True)
class SceneCapsule:
    location: str = ""
    subjects: tuple[str, ...] = ()
    action: str = ""
    mood: str = ""
    visual_anchors: tuple[str, ...] = ()
    negative_constraints: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "subjects": _list(self.subjects),
            "action": self.action,
            "mood": self.mood,
            "visual_anchors": _list(self.visual_anchors),
            "negative_constraints": _list(self.negative_constraints),
        }


@dataclass(frozen=True)
class CharacterCapsule:
    identity: str = ""
    appearance: tuple[str, ...] = ()
    outfit: str = ""
    accessories: tuple[str, ...] = ()
    facial_impression: str = ""
    consistency_priority: str = "medium"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "appearance": _list(self.appearance),
            "outfit": self.outfit,
            "accessories": _list(self.accessories),
            "facial_impression": self.facial_impression,
            "consistency_priority": self.consistency_priority,
        }


@dataclass(frozen=True)
class StyleCapsule:
    medium: str = ""
    line_quality: str = ""
    shading: str = ""
    palette: tuple[str, ...] = ()
    detail_density: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "medium": self.medium,
            "line_quality": self.line_quality,
            "shading": self.shading,
            "palette": _list(self.palette),
            "detail_density": self.detail_density,
        }


@dataclass(frozen=True)
class MoodCapsule:
    emotion: str = ""
    atmosphere: str = ""
    tension: str = ""
    lighting: str = ""
    palette_direction: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "atmosphere": self.atmosphere,
            "tension": self.tension,
            "lighting": self.lighting,
            "palette_direction": self.palette_direction,
        }


@dataclass(frozen=True)
class ReferenceCapsule:
    source_image_id: str = ""
    preserve: tuple[str, ...] = ()
    allow_modify: tuple[str, ...] = ()
    strength: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_image_id": self.source_image_id,
            "preserve": _list(self.preserve),
            "allow_modify": _list(self.allow_modify),
            "strength": self.strength,
        }


@dataclass(frozen=True)
class InstructionCapsule:
    user_intent: str = ""
    explicit_changes: tuple[str, ...] = ()
    preserve_requests: tuple[str, ...] = ()
    forbidden_changes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "user_intent": self.user_intent,
            "explicit_changes": _list(self.explicit_changes),
            "preserve_requests": _list(self.preserve_requests),
            "forbidden_changes": _list(self.forbidden_changes),
        }


@dataclass(frozen=True)
class ConflictItem:
    field: str
    image_value: str
    text_value: str
    severity: str = "medium"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "image_value": self.image_value,
            "text_value": self.text_value,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ConflictReport:
    preserve: tuple[str, ...] = ()
    modify: tuple[str, ...] = ()
    conflicts: tuple[ConflictItem, ...] = ()
    proposed_questions: tuple[str, ...] = ()

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(item.field for item in self.conflicts)

    @property
    def severity(self) -> str:
        if not self.conflicts:
            return "none"
        return max(self.conflicts, key=lambda item: _SEVERITY_RANK.get(item.severity, 0)).severity

    @property
    def requires_user_confirmation(self) -> bool:
        return self.has_conflict and _SEVERITY_RANK.get(self.severity, 0) >= _SEVERITY_RANK["high"]

    def blocks_generation(self) -> bool:
        return self.requires_user_confirmation

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "has_conflict": self.has_conflict,
            "severity": self.severity,
            "fields": _list(self.fields),
            "preserve": _list(self.preserve),
            "modify": _list(self.modify),
            "conflicts": [item.to_json_dict() for item in self.conflicts],
            "requires_user_confirmation": self.requires_user_confirmation,
            "proposed_questions": _list(self.proposed_questions),
        }


@dataclass(frozen=True)
class ConditioningWeights:
    semantic_weight: float = 1.0
    reference_weight: float = 0.6
    style_weight: float = 0.4
    mood_weight: float = 0.2
    spatial_weight: float = 0.0

    def to_json_dict(self) -> dict[str, float]:
        return {
            "semantic_weight": self.semantic_weight,
            "reference_weight": self.reference_weight,
            "style_weight": self.style_weight,
            "mood_weight": self.mood_weight,
            "spatial_weight": self.spatial_weight,
        }


@dataclass(frozen=True)
class GemmanimaProtocol:
    scene: SceneCapsule = field(default_factory=SceneCapsule)
    character: CharacterCapsule = field(default_factory=CharacterCapsule)
    style: StyleCapsule = field(default_factory=StyleCapsule)
    mood: MoodCapsule = field(default_factory=MoodCapsule)
    reference: ReferenceCapsule = field(default_factory=ReferenceCapsule)
    instruction: InstructionCapsule = field(default_factory=InstructionCapsule)
    conflict: ConflictReport = field(default_factory=ConflictReport)
    conditioning: ConditioningWeights = field(default_factory=ConditioningWeights)
    version: str = PROTOCOL_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scene": self.scene.to_json_dict(),
            "character": self.character.to_json_dict(),
            "style": self.style.to_json_dict(),
            "mood": self.mood.to_json_dict(),
            "reference": self.reference.to_json_dict(),
            "instruction": self.instruction.to_json_dict(),
            "conflict": self.conflict.to_json_dict(),
            "conditioning": self.conditioning.to_json_dict(),
        }
