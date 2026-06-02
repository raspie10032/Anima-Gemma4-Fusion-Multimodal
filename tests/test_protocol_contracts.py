import json
from pathlib import Path

from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.manifest import Manifest
from gemmanima.core.protocol import (
    PROTOCOL_VERSION,
    CharacterCapsule,
    ConflictItem,
    ConflictReport,
    GemmanimaProtocol,
    InstructionCapsule,
    ReferenceCapsule,
    SceneCapsule,
    StyleCapsule,
)
from gemmanima.core.schemas import ConditioningBundle, JobStatus, Mode


def test_protocol_payload_keeps_modal_axes_separate() -> None:
    protocol = GemmanimaProtocol(
        scene=SceneCapsule(
            location="ruined cathedral",
            subjects=("silver-haired girl",),
            action="battle stance",
            mood="tense",
            visual_anchors=("moonlight",),
            negative_constraints=("low detail",),
        ),
        character=CharacterCapsule(
            identity="same_character_identity",
            appearance=("silver hair", "blue cloak"),
            outfit="blue cloak",
            accessories=("staff",),
            facial_impression="calm",
            consistency_priority="high",
        ),
        style=StyleCapsule(
            medium="anime illustration",
            palette=("green", "silver"),
        ),
        reference=ReferenceCapsule(
            source_image_id="ref-001",
            preserve=("hair_color", "face_impression"),
            allow_modify=("background",),
            strength=0.8,
        ),
        instruction=InstructionCapsule(
            user_intent="change the background but preserve the character",
            explicit_changes=("background_to_ruined_cathedral",),
            preserve_requests=("same_character_identity",),
            forbidden_changes=("hair_color",),
        ),
    )

    payload = protocol.to_json_dict()

    assert payload["version"] == PROTOCOL_VERSION == "0.1"
    assert payload["scene"]["location"] == "ruined cathedral"
    assert payload["character"]["appearance"] == ["silver hair", "blue cloak"]
    assert payload["reference"]["strength"] == 0.8
    assert payload["conditioning"]["semantic_weight"] > payload["conditioning"]["mood_weight"]
    assert payload["conflict"]["has_conflict"] is False


def test_conflict_report_blocks_high_identity_conflict() -> None:
    report = ConflictReport(
        preserve=("same_character_identity", "face_impression"),
        modify=("hair_color_to_black",),
        conflicts=(
            ConflictItem(
                field="hair_color",
                image_value="silver",
                text_value="black",
                severity="high",
            ),
        ),
    )

    payload = report.to_json_dict()

    assert report.requires_user_confirmation is True
    assert report.blocks_generation() is True
    assert payload["has_conflict"] is True
    assert payload["requires_user_confirmation"] is True
    assert payload["fields"] == ["hair_color"]


def test_manifest_defaults_include_next_version_reproducibility_fields() -> None:
    manifest = Manifest.new(
        session_id="protocol-test",
        mode=Mode.GENERATE_IMAGE,
        status=JobStatus.COMPLETED,
        user_request="draw a forest",
    )

    data = manifest.to_json_dict()

    assert data["protocol_version"] == PROTOCOL_VERSION
    assert data["translator_version"] == ""
    assert data["precision"]["anima_dtype"] == "fp16"
    assert data["precision"]["gemma_quantization"] == "int4"
    assert data["memory_policy"]["cache_gemma_states"] is True
    assert data["conditioning_metrics"]["measurement_policy"] == "observed_only"
    assert data["conditioning_metrics"]["run_conditioning_mse"] is None
    assert data["conditioning_metrics"]["measured"] is False
    assert data["lineage"]["lineage"] == "internal_experimental"
    assert data["lineage"]["public_release_allowed"] is False
    assert data["modules"]["frozen"] == []
    assert data["modules"]["trainable"] == []


def test_conditioning_bundle_can_record_split_conditioning_axes() -> None:
    bundle = ConditioningBundle(
        source="protocol_v0_1",
        semantic_conditioning={"shape": [1, 512, 1024]},
        reference_conditioning={"source_image_id": "ref-001"},
        style_conditioning={"medium": "anime illustration"},
        mood_conditioning={"emotion": "tense"},
        negative_conditioning={"constraints": ["low detail"]},
        strength_weights={"semantic_weight": 1.0, "reference_weight": 0.6},
        conflict_state={"has_conflict": False},
        renderer_profile="anima_fp16_final",
    )

    bundle.validate()
    payload = bundle.to_json_dict()

    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["semantic_conditioning"]["shape"] == [1, 512, 1024]
    assert payload["reference_conditioning"]["source_image_id"] == "ref-001"
    assert payload["strength_weights"]["semantic_weight"] == 1.0
    assert payload["conflict_state"]["has_conflict"] is False
    assert payload["renderer_profile"] == "anima_fp16_final"


def test_conductor_manifest_records_full_conditioning_bundle_contract(tmp_path: Path) -> None:
    conductor = GemmAnimaConductor(
        session_id="protocol-run-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
    )

    response = conductor.handle_user_message("draw a quiet moonlit cathedral")
    data = conductor.manifest_store.read_json(response.manifest_path)
    hiddenstage = data["renderer"]["hiddenstage_conditioning"]

    assert hiddenstage["protocol_version"] == PROTOCOL_VERSION
    assert hiddenstage["shape"] == [1, 512, 1024]
    assert "semantic_conditioning" in hiddenstage
    assert hiddenstage["conflict_state"]["has_conflict"] is False
    assert hiddenstage["renderer_profile"] == "anima_fp16_final"
    assert data["conditioning_metrics"]["measurement_policy"] == "observed_only"
    assert data["conditioning_metrics"]["run_conditioning_mse"] is None
    assert data["conditioning_metrics"]["bridge_val_mse"] is not None


def test_next_version_json_schemas_are_present_and_parseable() -> None:
    schema_paths = (
        Path("schemas/gemmanima_protocol.schema.json"),
        Path("schemas/conditioning_bundle.schema.json"),
        Path("schemas/conflict_report.schema.json"),
        Path("schemas/run_manifest.schema.json"),
    )

    for schema_path in schema_paths:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
