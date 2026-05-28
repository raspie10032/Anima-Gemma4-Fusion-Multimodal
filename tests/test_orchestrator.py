from pathlib import Path

from gemmanima.training.orchestrator import pipeline_status
from gemmanima.training.teacher_targets import audit_split_target_completion, expected_split_shards


def test_expected_split_shards_matches_current_dataset() -> None:
    expected = expected_split_shards()

    assert expected["4070_ti_super_rows"] == 135280
    assert expected["5060_rows"] == 57978
    assert expected["4070_ti_super_shards"] == 68
    assert expected["5060_shards"] == 29
    assert expected["total_shards"] == 97


def test_audit_split_target_completion(tmp_path: Path) -> None:
    for idx in range(68):
        (tmp_path / f"shard_{idx:04d}.pt").write_bytes(b"x")
    for idx in range(29):
        (tmp_path / f"shard_5060_{idx:04d}.pt").write_bytes(b"x")

    audit = audit_split_target_completion(tmp_path)

    assert audit["complete"] is True
    assert audit["complete_4070_ti_super"] is True
    assert audit["complete_5060"] is True


def test_audit_split_target_completion_accepts_rebalanced_total(tmp_path: Path) -> None:
    for idx in range(55):
        (tmp_path / f"shard_{idx:04d}.pt").write_bytes(b"x")
    for idx in range(42):
        (tmp_path / f"shard_5060_{idx:04d}.pt").write_bytes(b"x")

    audit = audit_split_target_completion(tmp_path)

    assert audit["complete_total"] is True
    assert audit["complete"] is True


def test_pipeline_status_waits_for_teacher_targets(tmp_path: Path) -> None:
    status = pipeline_status(target_dir=tmp_path / "targets", gemma_dir=tmp_path / "gemma")

    assert status["next_action"] == "wait_for_teacher_targets"
