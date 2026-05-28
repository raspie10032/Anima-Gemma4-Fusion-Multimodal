import json
from pathlib import Path

from gemmanima.training.readiness import audit_manifest, build_training_readiness_report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def test_audit_manifest_detects_missing_crossattn_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    embed = tmp_path / "embed.pt"
    embed.write_bytes(b"fake")
    write_jsonl(
        manifest,
        [
            {
                "id": 1,
                "action": "generate",
                "teacher_prompt": "masterpiece",
                "target_crossattn_emb": None,
                "image_embed_pre": str(embed),
                "rating": "g",
            },
            {
                "id": 2,
                "action": "generate",
                "teacher_prompt": "best quality",
                "target_crossattn_emb": str(tmp_path / "target.pt"),
                "image_embed_pre": str(embed),
                "rating": "e",
            },
        ],
    )

    audit = audit_manifest(manifest, check_image_embed_exists=True)

    assert audit.rows == 2
    assert audit.generate_rows == 2
    assert audit.rows_with_target_crossattn == 1
    assert audit.needs_teacher_target_extraction is True
    assert audit.rating_counts == {"e": 1, "g": 1}


def test_training_readiness_selects_teacher_extraction_when_targets_missing(tmp_path: Path) -> None:
    planner = tmp_path / "planner"
    planner.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "embed_vision.pt", "EVAL_PASS.json"):
        (planner / name).write_bytes(b"x")
    (planner / "eval_checkpoints.jsonl").write_bytes(b"x")
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    write_jsonl(train_manifest, [{"id": 1, "action": "generate", "teacher_prompt": "x"}])
    write_jsonl(eval_manifest, [{"id": 2, "action": "generate", "teacher_prompt": "y"}])

    report = build_training_readiness_report(
        planner_out=planner,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
    )

    assert report["planner_ready"] is True
    assert report["hiddenstage_bridge_dataset_ready"] is False
    assert report["next_required_action"] == "extract_teacher_crossattn_targets"
