from pathlib import Path

from gemmanima.training.gemma_cache import audit_cache_pairing, default_split_gemma_cache_plans


def test_default_gemma_cache_plans_keep_gpu_roles() -> None:
    plans = default_split_gemma_cache_plans()

    assert plans[0].name == "gemma_4070_ti_super"
    assert plans[0].gpu_index == 0
    assert plans[0].embed_on_gpu is True
    assert "shard_[0-9][0-9][0-9][0-9].pt" in plans[0].command()
    assert "shard_re4070_*.pt" in plans[0].command()
    assert "shard_4070w2*.pt" in plans[0].command()
    assert "shard_missing4070_*.pt" in plans[0].command()
    assert plans[1].name == "gemma_5060"
    assert plans[1].gpu_index == 1
    assert plans[1].embed_on_gpu is False
    assert "shard_5060_*.pt" in plans[1].command()


def test_audit_cache_pairing_reports_missing_gemma(tmp_path: Path) -> None:
    targets = tmp_path / "targets"
    gemma = tmp_path / "gemma"
    targets.mkdir()
    gemma.mkdir()
    (targets / "shard_0000.pt").write_bytes(b"x")
    (targets / "shard_0001.pt").write_bytes(b"x")
    (gemma / "shard_0000.pt").write_bytes(b"x")

    audit = audit_cache_pairing(target_dir=targets, gemma_dir=gemma)

    assert audit["target_shards"] == 2
    assert audit["gemma_shards"] == 1
    assert audit["paired_shards"] == 1
    assert audit["missing_gemma_shards"] == 1
    assert audit["ready_for_bridge_training"] is False
