from pathlib import Path

import pytest

from gemmanima.context.capsule import ContextRelevanceFilter
from gemmanima.core.conflict import ConflictResolver
from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.manifest import Manifest, ManifestSchemaError, ManifestStore
from gemmanima.core.protocol import ConflictReport, GemmanimaProtocol
from gemmanima.core.protocol_parser import ProtocolParser
from gemmanima.core.schemas import ChatTurn, ConditioningBundle, GenerationPlan, JobStatus, Mode, RenderResult
from gemmanima.core.validation import (
    SchemaValidationError,
    validate_conditioning_bundle_payload,
    validate_conflict_report_payload,
    validate_protocol_payload,
)


class CountingRenderer:
    dry_run = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        self.calls += 1
        return RenderResult(
            image_id="counting-render",
            output_path=Path("counting-render.txt"),
            seed=123,
        )


def test_context_filter_builds_protocol_from_visual_reference_and_request() -> None:
    capsule = ContextRelevanceFilter().build(
        [ChatTurn(role="user", content="Reference image: silver hair, blue cloak, calm face")],
        "draw the same character in a ruined cathedral",
    )

    assert capsule.protocol.scene.action == "draw the same character in a ruined cathedral"
    assert capsule.protocol.instruction.user_intent == "draw the same character in a ruined cathedral"
    assert "hair_color" in capsule.protocol.reference.preserve
    assert "outfit" in capsule.protocol.reference.preserve
    assert capsule.protocol.reference.strength == 0.6


def test_protocol_parser_extracts_reference_axes_without_context_filter() -> None:
    protocol = ProtocolParser().parse(
        user_request="draw Furina instead as a photorealistic portrait",
        visual_facts=("Reference image: identity Nahida, anime illustration style, blue cloak",),
    )

    assert protocol.character.identity == "nahida"
    assert protocol.scene.action == "draw Furina instead as a photorealistic portrait"
    assert protocol.reference.preserve == ("outfit", "style", "identity")
    assert protocol.reference.strength == 0.6


def test_protocol_parser_extracts_mood_lighting_negative_and_change_signals() -> None:
    protocol = ProtocolParser().parse(
        user_request=(
            "keep the same character and preserve the hair color, change the background "
            "to a moonlit cathedral, make her black-haired, no watermark, avoid blurry details"
        ),
        visual_facts=("Reference image: silver hair, blue cloak, calm face",),
    )

    assert protocol.scene.mood == "calm"
    assert protocol.mood.emotion == "calm"
    assert protocol.mood.lighting == "moonlit"
    assert protocol.scene.negative_constraints == ("watermark", "blurry details")
    assert "background_to_moonlit_cathedral" in protocol.instruction.explicit_changes
    assert "hair_color_to_black" in protocol.instruction.explicit_changes
    assert protocol.instruction.preserve_requests == (
        "hair_color",
        "outfit",
        "face_impression",
        "identity",
    )
    assert protocol.instruction.forbidden_changes == ("hair_color", "outfit", "face_impression", "identity")


def test_protocol_parser_keeps_preserve_requests_conservative_without_reference() -> None:
    protocol = ProtocolParser().parse(
        user_request="draw a tense neon alley, preserve the outfit, without text",
    )

    assert protocol.reference.source_image_id == ""
    assert protocol.reference.strength == 0.0
    assert protocol.reference.preserve == ("outfit",)
    assert protocol.scene.mood == "tense"
    assert protocol.mood.tension == "high"
    assert protocol.mood.lighting == "neon"
    assert protocol.scene.negative_constraints == ("text",)


def test_high_severity_reference_text_conflict_blocks_renderer(tmp_path: Path) -> None:
    renderer = CountingRenderer()
    conductor = GemmAnimaConductor(
        session_id="conflict-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
        renderer=renderer,
    )
    conductor.history.append(
        ChatTurn(role="user", content="Reference image: silver hair, blue cloak, calm face")
    )

    response = conductor.handle_user_message(
        "draw the same character in a ruined cathedral but make her black-haired"
    )

    assert response.status == JobStatus.ASK_CLARIFY
    assert response.output_path is None
    assert renderer.calls == 0
    assert "hair" in response.message.lower()
    assert "conflict:blocked" in response.progress
    data = ManifestStore(tmp_path / "manifests").read_json(response.manifest_path)
    assert data["renderer"]["conflict"]["requires_user_confirmation"] is True
    assert data["renderer"]["conflict"]["fields"] == ["hair_color"]


def test_conductor_resumes_blocked_request_from_short_clarification(tmp_path: Path) -> None:
    renderer = CountingRenderer()
    conductor = GemmAnimaConductor(
        session_id="conflict-resume-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
        renderer=renderer,
    )
    conductor.history.append(
        ChatTurn(role="user", content="Reference image: silver hair, blue cloak, calm face")
    )

    blocked = conductor.handle_user_message(
        "draw the same character in a ruined cathedral but make her black-haired"
    )
    resumed = conductor.handle_user_message("Change it to black hair.")

    assert blocked.status == JobStatus.ASK_CLARIFY
    assert resumed.status == JobStatus.COMPLETED
    assert resumed.clarification_required is False
    assert renderer.calls == 1
    assert "clarification:resume" in resumed.progress


def test_conflict_resolver_detects_outfit_style_identity_and_unsafe_risks() -> None:
    resolver = ConflictResolver()
    plan = GenerationPlan(prompt="unused")

    outfit = ContextRelevanceFilter().build(
        [ChatTurn(role="user", content="Reference image: blue cloak, calm face")],
        "draw the same character wearing a red dress",
    )
    style = ContextRelevanceFilter().build(
        [ChatTurn(role="user", content="Reference image: anime illustration style, soft line art")],
        "draw the same character as a photorealistic portrait",
    )
    identity = ContextRelevanceFilter().build(
        [ChatTurn(role="user", content="Reference image: identity Nahida, green outfit")],
        "draw Furina instead in a moonlit garden",
    )
    unsafe = ContextRelevanceFilter().build(
        [],
        "draw a sexualized minor",
    )

    assert resolver.resolve(outfit, plan).fields == ("outfit",)
    assert resolver.resolve(style, plan).fields == ("style",)
    assert resolver.resolve(identity, plan).fields == ("identity",)
    assert resolver.resolve(unsafe, plan).fields == ("unsafe_content",)
    assert resolver.resolve(unsafe, plan).blocks_generation() is True


def test_conflict_resolver_returns_multiple_conflicts_together() -> None:
    capsule = ContextRelevanceFilter().build(
        [ChatTurn(role="user", content="Reference image: silver hair, blue cloak, anime illustration style")],
        "draw the same character with black hair wearing a red dress as a photorealistic portrait",
    )

    report = ConflictResolver().resolve(capsule, GenerationPlan(prompt="unused"))

    assert report.fields == ("hair_color", "outfit", "style")
    assert report.blocks_generation() is True
    assert len(report.conflicts) == 3


def test_conflict_resolution_applies_only_to_the_confirmed_field() -> None:
    capsule = ContextRelevanceFilter().build(
        [
            ChatTurn(role="user", content="Reference image: silver hair, blue cloak, anime illustration style"),
            ChatTurn(role="user", content="change to black hair"),
        ],
        "draw the same character with black hair wearing a red dress as a photorealistic portrait",
    )

    report = ConflictResolver().resolve(capsule, GenerationPlan(prompt="unused"))

    assert report.fields == ("outfit", "style")
    assert report.blocks_generation() is True


def test_conflict_resolution_accepts_explicit_reference_preservation() -> None:
    capsule = ContextRelevanceFilter().build(
        [
            ChatTurn(role="user", content="Reference image: silver hair, blue cloak, anime illustration style"),
            ChatTurn(role="user", content="preserve the reference hair color"),
        ],
        "draw the same character with black hair wearing a red dress as a photorealistic portrait",
    )

    report = ConflictResolver().resolve(capsule, GenerationPlan(prompt="unused"))

    assert report.fields == ("outfit", "style")
    assert "hair_color_to_black" not in report.modify


def test_hiddenstage_bundle_copies_protocol_axes(tmp_path: Path) -> None:
    conductor = GemmAnimaConductor(
        session_id="protocol-axis-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
    )
    conductor.history.append(
        ChatTurn(role="user", content="Reference image: silver hair, blue cloak, calm face")
    )

    response = conductor.handle_user_message("draw the same character in a moonlit forest")
    data = ManifestStore(tmp_path / "manifests").read_json(response.manifest_path)
    bundle = data["renderer"]["hiddenstage_conditioning"]

    assert bundle["semantic_conditioning"]["scene"]["action"] == "draw the same character in a moonlit forest"
    assert bundle["reference_conditioning"]["preserve"] == ["hair_color", "outfit", "face_impression"]
    assert bundle["style_conditioning"]["medium"] == ""
    assert bundle["mood_conditioning"]["emotion"] == "calm"
    assert bundle["mood_conditioning"]["lighting"] == "moonlit"
    assert bundle["negative_conditioning"]["constraints"] == []


def test_manifest_store_rejects_invalid_protocol_version(tmp_path: Path) -> None:
    manifest = Manifest.new(
        session_id="schema-test",
        mode=Mode.GENERATE_IMAGE,
        status=JobStatus.COMPLETED,
        user_request="draw a forest",
    )
    manifest.protocol_version = "9.9"

    with pytest.raises(ManifestSchemaError, match="protocol_version"):
        ManifestStore(tmp_path / "manifests").write(manifest)


def test_schema_validation_accepts_protocol_bundle_and_conflict_payloads() -> None:
    bundle = ConditioningBundle(
        source="protocol_v0_1",
        semantic_conditioning={"scene": {}},
        strength_weights={"semantic_weight": 1.0},
        conflict_state={"has_conflict": False},
        renderer_profile="anima_fp16_final",
    )

    validate_protocol_payload(GemmanimaProtocol().to_json_dict())
    validate_conditioning_bundle_payload(bundle.to_json_dict())
    validate_conflict_report_payload(ConflictReport().to_json_dict())


def test_schema_validation_rejects_bad_protocol_and_bundle_payloads() -> None:
    protocol_payload = GemmanimaProtocol().to_json_dict()
    protocol_payload["version"] = "9.9"
    with pytest.raises(SchemaValidationError, match="version"):
        validate_protocol_payload(protocol_payload)

    bundle_payload = ConditioningBundle(source="x").to_json_dict()
    bundle_payload["shape"] = [1, 64, 1024]
    with pytest.raises(SchemaValidationError, match="shape"):
        validate_conditioning_bundle_payload(bundle_payload)
