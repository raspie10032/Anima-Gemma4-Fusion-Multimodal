from pathlib import Path

import pytest

from gemmanima.core.validation import SchemaValidationError, validate_cache_build_manifest_payload
from gemmanima.training.cache_manifest import CacheBuildManifest, build_cache_manifest_write_command, write_cache_build_manifest


def test_cache_build_manifest_records_poc1_contract(tmp_path: Path) -> None:
    manifest = CacheBuildManifest(
        stage="poc1_1k_smoke",
        cache_kind="anima_te_conditioning",
        sample_count=1000,
        source_manifest=tmp_path / "source.jsonl",
        output_dir=tmp_path / "cache",
        success_count=990,
        failure_count=10,
        shape=(1, 512, 1024),
        dtype="float16",
        device="cuda:0",
    )

    payload = manifest.to_json_dict()
    validate_cache_build_manifest_payload(payload)

    assert payload["protocol_version"] == "0.1"
    assert payload["lineage"]["lineage"] == "internal_experimental"
    assert payload["lineage"]["public_release_allowed"] is False
    assert payload["success_rate"] == 0.99
    assert payload["shape"] == [1, 512, 1024]


def test_cache_build_manifest_rejects_invalid_counts(tmp_path: Path) -> None:
    manifest = CacheBuildManifest(
        stage="poc1_1k_smoke",
        cache_kind="gemma_text_state",
        sample_count=10,
        source_manifest=tmp_path / "source.jsonl",
        output_dir=tmp_path / "cache",
        success_count=11,
        failure_count=0,
    )

    with pytest.raises(SchemaValidationError, match="success_count"):
        manifest.validate()


def test_write_cache_build_manifest_validates_and_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "CACHE_BUILD_MANIFEST.json"
    manifest = CacheBuildManifest(
        stage="poc1_1k_smoke",
        cache_kind="gemma_text_state",
        sample_count=2,
        source_manifest=tmp_path / "source.jsonl",
        output_dir=tmp_path / "cache",
        success_count=2,
        failure_count=0,
        shape=(1, 16, 1536),
        dtype="float32",
        device="cuda:0",
    )

    written = write_cache_build_manifest(manifest, out)

    assert written == out
    assert '"cache_kind": "gemma_text_state"' in out.read_text(encoding="utf-8")


def test_cache_manifest_write_command_points_at_cli(tmp_path: Path) -> None:
    command = build_cache_manifest_write_command(
        cache_kind="anima_te_conditioning",
        sample_count=1000,
        source_manifest=tmp_path / "subset.jsonl",
        output_dir=tmp_path / "targets",
        manifest_out=tmp_path / "targets" / "CACHE_BUILD_MANIFEST.json",
        success_count=1000,
        shape=(1, 512, 1024),
        dtype="float16",
        device="cuda:0",
    )

    assert "python -m gemmanima.cli write-cache-manifest" in command
    assert "--cache-kind anima_te_conditioning" in command
    assert "--shape 1,512,1024" in command
