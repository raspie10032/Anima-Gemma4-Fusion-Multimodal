import json
from pathlib import Path

from gemmanima.training.teacher_targets import audit_target_cache, export_teacher_subset


def test_export_teacher_subset_uses_teacher_prompt(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    subset = tmp_path / "subset.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"id": 10, "action": "generate", "teacher_prompt": "masterpiece, cat"}),
                json.dumps({"id": 11, "action": "clarify", "teacher_prompt": "skip"}),
                json.dumps({"id": 12, "action": "generate", "visible_prompt": "fallback prompt"}),
            ]
        ),
        encoding="utf-8",
    )

    result = export_teacher_subset(manifest, subset)

    rows = [json.loads(line) for line in subset.read_text(encoding="utf-8").splitlines()]
    assert result.rows_written == 2
    assert rows[0]["idx"] == 10
    assert rows[0]["text"] == "masterpiece, cat"
    assert rows[1]["idx"] == 12
    assert rows[1]["text"] == "fallback prompt"
    assert "06_cache_targets.py" in result.command
    assert "python_embeded" in result.command


def test_audit_target_cache_reports_shards(tmp_path: Path) -> None:
    (tmp_path / "shard_0000.pt").write_bytes(b"x")
    (tmp_path / "shard_0001.pt").write_bytes(b"x")

    audit = audit_target_cache(tmp_path)

    assert audit["exists"] is True
    assert audit["shard_count"] == 2
    assert audit["first_shard"].endswith("shard_0000.pt")
