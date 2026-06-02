import json
from pathlib import Path

from gemmanima.cli import main
from gemmanima.training.poc1_cache import build_poc1_bridge_plan, build_poc1_cache_plan


def test_build_poc1_cache_plan_uses_1k_subset_and_manifest_commands(tmp_path: Path) -> None:
    plan = build_poc1_cache_plan(
        manifest=tmp_path / "source.jsonl",
        subset=tmp_path / "poc1_subset.jsonl",
        target_dir=tmp_path / "targets",
        gemma_dir=tmp_path / "gemma",
    )

    assert plan["stage"] == "poc1_1k_smoke"
    assert plan["limit"] == 1000
    assert "--limit 1000" in plan["prepare_subset_command"]
    assert "06_cache_targets.py" in plan["teacher_target_command"]
    assert "write-cache-manifest" in plan["teacher_target_manifest_command"]
    assert len(plan["gemma_cache_plans"]) == 2
    assert "07_cache_gemma_batched.py" in plan["gemma_cache_plans"][0]["command"]
    assert "write-cache-manifest" in plan["gemma_cache_plans"][0]["cache_manifest_command"]


def test_build_poc1_cache_plan_can_describe_10k_pilot(tmp_path: Path) -> None:
    plan = build_poc1_cache_plan(
        manifest=tmp_path / "source.jsonl",
        subset=tmp_path / "poc1_10k_subset.jsonl",
        target_dir=tmp_path / "poc1_10k" / "targets",
        gemma_dir=tmp_path / "poc1_10k" / "gemma",
        limit=10000,
        gpu_profile="4070-only",
    )

    assert plan["stage"] == "poc1_10k_pilot"
    assert plan["limit"] == 10000
    assert plan["gpu_profile"] == "4070-only"
    assert "$env:CUDA_VISIBLE_DEVICES='0'" in plan["teacher_target_command"]
    assert "5060" not in json.dumps(plan, ensure_ascii=False)
    assert "--limit 10000" in plan["prepare_subset_command"]
    assert "poc1_10k_pilot_CACHE_BUILD_MANIFEST.json" in plan["teacher_target_manifest_path"]
    assert "--stage poc1_10k_pilot" in plan["teacher_target_manifest_command"]
    assert "poc1_10k_" in plan["gemma_cache_plans"][0]["cache_manifest_path"]
    assert "--stage poc1_10k_pilot" in plan["gemma_cache_plans"][0]["cache_manifest_command"]
    assert "--sample-count 10000" in plan["gemma_cache_plans"][0]["cache_manifest_command"]


def test_cli_poc1_cache_plan_outputs_json(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "poc1-cache-plan",
            "--manifest",
            str(tmp_path / "source.jsonl"),
            "--subset",
            str(tmp_path / "poc1_subset.jsonl"),
            "--target-dir",
            str(tmp_path / "targets"),
            "--gemma-dir",
            str(tmp_path / "gemma"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "poc1_1k_smoke"
    assert payload["limit"] == 1000
    assert "prepare-teacher-targets" in payload["prepare_subset_command"]


def test_cli_poc1_cache_plan_accepts_10k_limit(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "poc1-cache-plan",
            "--manifest",
            str(tmp_path / "source.jsonl"),
            "--subset",
            str(tmp_path / "poc1_10k_subset.jsonl"),
            "--target-dir",
            str(tmp_path / "poc1_10k" / "targets"),
            "--gemma-dir",
            str(tmp_path / "poc1_10k" / "gemma"),
            "--limit",
            "10000",
            "--gpu-profile",
            "4070-only",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "poc1_10k_pilot"
    assert payload["limit"] == 10000
    assert payload["gpu_profile"] == "4070-only"
    assert len(payload["gemma_cache_plans"]) == 1


def test_build_poc1_bridge_plan_uses_local_cache_dirs(tmp_path: Path) -> None:
    plan = build_poc1_bridge_plan(
        target_dir=tmp_path / "targets",
        gemma_dir=tmp_path / "gemma",
        output=tmp_path / "bridge" / "poc1_bridge.pt",
    )

    assert plan["stage"] == "poc1_bridge_smoke"
    assert plan["ready"] is False
    assert "--limit-shards 1" in plan["train_command"]
    assert "bridge-eval-status" in plan["eval_command"]
    assert "smoke_hiddenstage_bridge_forward.py" in plan["forward_smoke_command"]


def test_build_poc1_bridge_plan_can_describe_full_pilot_training(tmp_path: Path) -> None:
    plan = build_poc1_bridge_plan(
        target_dir=tmp_path / "targets",
        gemma_dir=tmp_path / "gemma",
        output=tmp_path / "bridge" / "poc1_10k_bridge.pt",
        limit_shards=None,
    )

    assert plan["stage"] == "poc1_bridge_pilot"
    assert "--limit-shards" not in plan["train_command"]
    assert "$env:CUDA_VISIBLE_DEVICES='0'" in plan["train_command"]
    assert "poc1_10k_bridge.pt" in plan["eval_command"]


def test_cli_poc1_bridge_plan_outputs_json(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "poc1-bridge-plan",
            "--target-dir",
            str(tmp_path / "targets"),
            "--gemma-dir",
            str(tmp_path / "gemma"),
            "--output",
            str(tmp_path / "bridge" / "poc1_bridge.pt"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "poc1_bridge_smoke"
    assert "poc1_bridge.pt" in payload["train_command"]


def test_cli_poc1_bridge_plan_accepts_unlimited_shards(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "poc1-bridge-plan",
            "--target-dir",
            str(tmp_path / "targets"),
            "--gemma-dir",
            str(tmp_path / "gemma"),
            "--output",
            str(tmp_path / "bridge" / "poc1_10k_bridge.pt"),
            "--limit-shards",
            "0",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "poc1_bridge_pilot"
    assert "--limit-shards" not in payload["train_command"]
