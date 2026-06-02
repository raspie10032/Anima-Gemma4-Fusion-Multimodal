import json

from scripts.build_v37_10k_curriculum import (
    BUCKET_COUNTS,
    build_v37_10k_curriculum,
    write_v37_10k_curriculum,
)


def test_v37_10k_curriculum_builds_protected_large_dataset() -> None:
    payload = build_v37_10k_curriculum(count=10000, seed=37000)

    records = payload["records"]
    assert len(records) == 10000
    assert payload["manifest"]["record_count"] == 10000
    assert payload["manifest"]["record_buckets"] == BUCKET_COUNTS
    assert payload["audit"]["status"] == "pass"
    assert payload["audit"]["unique_idx_count"] == 10000
    assert payload["audit"]["unique_id_count"] == 10000
    assert payload["manifest"]["protected_baseline"] == "v5"
    assert payload["manifest"]["v5_default_changed"] is False
    assert payload["manifest"]["promotion_allowed"] is False
    assert payload["manifest"]["required_gpu_policy"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert payload["manifest"]["required_gpu_policy"]["disallowed_gpu"] == "RTX 5060"

    bucket_counts = {
        bucket: sum(record["cache_source_bucket"] == bucket for record in records)
        for bucket in BUCKET_COUNTS
    }
    assert bucket_counts == BUCKET_COUNTS

    fixed6_cases = {
        record["source_case_id"]
        for record in records
        if record["cache_source_bucket"] == "00_fixed6_hard"
    }
    assert fixed6_cases >= {
        "text_eval_001_sign_luna_gate",
        "text_eval_002_book_cover_star_atlas",
        "text_eval_003_magic_circle_aether",
        "text_eval_004_ui_panel_hp_42",
        "text_eval_005_handwritten_note_meet_at_dawn",
        "text_eval_006_label_tea",
    }

    target_texts = {record.get("target_text") for record in records}
    assert {"AETHER", "MEET AT DAWN", "TEA", "HP 42"} <= target_texts
    assert len(payload["surface_curriculum"]["records"]) == 10000


def test_v37_10k_curriculum_writer_emits_manifest_audit_and_splits(tmp_path) -> None:
    payload = write_v37_10k_curriculum(output_dir=tmp_path, count=10000, seed=37000)

    manifest = json.loads((tmp_path / "v37_10k_curriculum_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "v37_10k_manifest_audit.json").read_text(encoding="utf-8"))
    assert manifest["record_count"] == 10000
    assert audit["status"] == "pass"

    all_records = (tmp_path / "v37_10k_prompt_records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(all_records) == 10000
    for bucket, expected_count in BUCKET_COUNTS.items():
        split_path = tmp_path / f"v37_{bucket}_prompts.jsonl"
        assert split_path.exists()
        assert len(split_path.read_text(encoding="utf-8").splitlines()) == expected_count
    assert payload["audit"]["split_prompt_files"]["40_synthetic_text"].endswith(
        "v37_40_synthetic_text_prompts.jsonl"
    )
