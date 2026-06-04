import json
from pathlib import Path

from PIL import Image

from gemmanima.cli import main
from gemmanima.training.text_preservation import (
    build_text_preservation_heldout_eval_plan,
    build_text_preservation_heldout_prompt_records,
    build_text_preservation_blended_plan,
    build_text_preservation_bridge_plan,
    build_text_preservation_bridge_status,
    build_text_preservation_compact_promotion_status,
    build_text_preservation_v9_artifact_feedback_dataset,
    build_text_preservation_artifact_feedback_alignment_audit,
    build_text_preservation_v9_candidate_plan,
    build_text_preservation_v10_candidate_plan,
    build_text_preservation_v11_candidate_plan,
    build_text_preservation_kv_delta_audit,
    build_text_preservation_v12_surface_plan,
    build_text_preservation_render_readability_label_manifest,
    build_text_preservation_surface_curriculum_manifest,
    build_text_preservation_qwen_target_refresh_manifest,
    build_text_preservation_v12_trainer_surface_contract_audit,
    build_text_preservation_v13_recovery_plan,
    build_text_preservation_v13_guard_weighted_manifest,
    build_text_preservation_v14_focus_fixed_gate_manifest,
    build_text_preservation_v17_targeted_teacher_refresh_manifest,
    build_text_preservation_v18_tea_micro_refresh_manifest,
    build_text_preservation_v19_dual_guard_refresh_manifest,
    build_text_preservation_v23_hard_heldout_refresh_manifest,
    build_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest,
    build_text_preservation_v21_text_roi_gate_report,
    build_text_preservation_prompt_records,
    build_text_preservation_promotion_status,
    build_text_preservation_release_gate_status,
    build_text_preservation_v9_artifact_gate_objective,
    build_text_preservation_v9_objective_plan,
    build_text_preservation_v9_trainer_support_audit,
    build_text_preservation_v5_plan,
    build_text_preservation_v8_fixed_gate_plan,
    build_text_preservation_v7_balanced_plan,
    build_text_preservation_v6_hard_negative_plan,
    build_text_preservation_v6_prompt_records,
    write_text_preservation_prompt_file,
)


def test_text_preservation_bridge_plan_reuses_qwen_prompt_subset_and_4070_only() -> None:
    plan = build_text_preservation_bridge_plan()

    assert plan["stage"] == "text_preservation_micro_overfit"
    assert plan["prompt_file"] == "reports/text_rendering_qwen_baseline/prompts.jsonl"
    assert plan["sample_count"] == 6
    assert plan["gpu_policy"] == {
        "cuda_visible_devices": "0",
        "gpu_name": "RTX 4070 Ti SUPER",
        "reserved_gpu": "RTX 5060 / CUDA device 1",
    }
    encoded = json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES='0'" in encoded
    assert "CUDA_VISIBLE_DEVICES='1'" not in encoded
    assert "RTX 5060" in encoded


def test_text_preservation_bridge_plan_uses_existing_te_distillation_scripts() -> None:
    plan = build_text_preservation_bridge_plan()

    assert "06_cache_targets.py" in plan["target_cache_command"]
    assert "--subset \"reports\\text_rendering_qwen_baseline\\prompts.jsonl\"" in plan["target_cache_command"]
    assert "--outdir \"runs\\cache\\text_preservation_qwen\\targets\"" in plan["target_cache_command"]
    assert "--shard 1000 --resume" in plan["target_cache_command"]

    assert "07_cache_gemma_batched.py" in plan["gemma_cache_command"]
    assert "GEMMA_EMBED_ON_GPU='1'" in plan["gemma_cache_command"]
    assert "--patterns \"*.pt\"" in plan["gemma_cache_command"]

    assert "08_train_stream_batched.py" in plan["train_command"]
    assert "--targets \"runs\\cache\\text_preservation_qwen\\targets\"" in plan["train_command"]
    assert "--gemma \"runs\\cache\\text_preservation_qwen\\gemma\"" in plan["train_command"]
    assert "--out \"runs\\cache\\text_preservation_qwen\\bridge\\text_preservation_bridge.pt\"" in plan["train_command"]
    assert "--resume-kv \"runs\\cache\\poc1_10k\\bridge\\poc1_10k_bridge.pt\"" in plan["train_command"]


def test_text_preservation_bridge_plan_emits_post_train_eval_with_distinct_student_name() -> None:
    plan = build_text_preservation_bridge_plan()

    assert plan["output"] == "runs/cache/text_preservation_qwen/bridge/text_preservation_bridge.pt"
    assert plan["post_train_qwen_eval_plan_command"] == (
        "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
        "--student-checkpoint \"runs\\cache\\text_preservation_qwen\\bridge\\text_preservation_bridge.pt\" "
        "--student-name gemma_text_preservation --json"
    )


def test_text_preservation_bridge_status_observes_artifacts_without_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs/cache/text_preservation_qwen/targets").mkdir(parents=True)
    (tmp_path / "runs/cache/text_preservation_qwen/gemma").mkdir(parents=True)
    (tmp_path / "runs/cache/text_preservation_qwen/targets/shard_0000.pt").write_bytes(b"placeholder")
    (tmp_path / "runs/cache/text_preservation_qwen/gemma/shard_0000.pt").write_bytes(b"placeholder")

    status = build_text_preservation_bridge_status()

    assert status["stage"] == "text_preservation_micro_overfit"
    assert status["ready_for_training"] is True
    assert status["cache_pairing"]["paired_shards"] == 1
    assert status["bridge"]["exists"] is False


def test_cli_text_preservation_bridge_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-bridge-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_micro_overfit"
    assert payload["training_strategy"]["resume_from"] == "runs/cache/poc1_10k/bridge/poc1_10k_bridge.pt"


def test_cli_text_preservation_bridge_status_outputs_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["text-preservation-bridge-status", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_micro_overfit"
    assert payload["ready_for_training"] is False


def test_text_preservation_prompt_records_expand_beyond_eval_six() -> None:
    records = build_text_preservation_prompt_records(count=48, start_idx=900000)

    assert len(records) == 48
    assert records[0]["idx"] == 900000
    assert records[-1]["idx"] == 900047
    assert records[0]["eval_idx"] == 0
    assert all("Object-only text preservation blend" in record["text"] for record in records)
    assert len({record["text"] for record in records}) == 48


def test_text_preservation_prompt_records_stay_unique_at_larger_scale() -> None:
    records = build_text_preservation_prompt_records(count=512, start_idx=910000)

    assert len(records) == 512
    assert len({record["text"] for record in records}) == 512
    assert len({record["target_text"] for record in records}) >= 64


def test_text_preservation_prompt_records_can_omit_sample_marker() -> None:
    records = build_text_preservation_prompt_records(
        count=12,
        prompt_index_offset=20000,
        src_prefix="text_preserve_v5",
        include_sample_marker=False,
    )

    assert records[0]["src"] == "text_preserve_v5_000"
    assert all("sample " not in record["text"] for record in records)
    assert len({record["text"] for record in records}) == 12


def test_text_preservation_heldout_prompt_records_do_not_overlap_training_prompts() -> None:
    training = build_text_preservation_prompt_records(count=512)
    heldout = build_text_preservation_heldout_prompt_records(count=64)

    assert len(heldout) == 64
    assert heldout[0]["idx"] == 950000
    assert heldout[0]["src"] == "text_preserve_heldout_000"
    assert not ({record["text"] for record in training} & {record["text"] for record in heldout})


def test_text_preservation_heldout_prompt_records_can_omit_sample_marker() -> None:
    heldout = build_text_preservation_heldout_prompt_records(
        count=8,
        prompt_index_offset=30000,
        src_prefix="text_preserve_heldout_clean",
        include_sample_marker=False,
    )

    assert heldout[0]["src"] == "text_preserve_heldout_clean_000"
    assert all("sample " not in record["text"] for record in heldout)


def test_text_preservation_prompt_records_can_include_eval_cases_first() -> None:
    records = build_text_preservation_prompt_records(count=2, include_eval_cases=True)

    assert len(records) == 8
    assert records[0]["src"] == "text_eval_001_sign_luna_gate"
    assert records[5]["src"] == "text_eval_006_label_tea"
    assert records[6]["src"] == "text_preserve_blend_000"


def test_write_text_preservation_prompt_file_writes_jsonl(tmp_path: Path) -> None:
    path = write_text_preservation_prompt_file(tmp_path / "prompts.jsonl", count=3, start_idx=42)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["idx"] for row in rows] == [42, 43, 44]
    assert rows[0]["src"] == "text_preserve_blend_000"


def test_text_preservation_blended_plan_oversamples_text_and_hardlinks_general_shard() -> None:
    plan = build_text_preservation_blended_plan(text_repeat=4, general_shards=1)

    assert plan["stage"] == "text_preservation_blended_candidate"
    assert plan["prompt_file"] == "reports/text_preservation_blended/prompts.jsonl"
    assert plan["sample_count"] == 54
    assert plan["blend"]["text_repeat"] == 4
    assert plan["blend"]["general_shards"] == 1
    assert plan["gpu_policy"]["cuda_visible_devices"] == "0"
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)
    assert "06_cache_targets.py" in plan["text_target_cache_command"]
    assert "--include-eval-cases" in plan["prompt_write_command"]
    assert "07_cache_gemma_batched.py" in plan["text_gemma_cache_command"]
    assert any("New-Item -ItemType HardLink" in command for command in plan["blend_link_commands"])
    assert any("00_text_0000.pt" in command for command in plan["blend_link_commands"])
    assert any("10_general_0000.pt" in command for command in plan["blend_link_commands"])
    assert "--targets \"runs\\cache\\text_preservation_blended\\blend_targets\"" in plan["train_command"]
    assert "--gemma \"runs\\cache\\text_preservation_blended\\blend_gemma\"" in plan["train_command"]
    assert "--resume-kv \"runs\\cache\\text_preservation_qwen\\bridge\\text_preservation_bridge.pt\"" in plan["train_command"]


def test_text_preservation_blended_plan_links_all_text_shards_when_count_exceeds_one_shard() -> None:
    plan = build_text_preservation_blended_plan(sample_count=1024, text_repeat=2, general_shards=1)

    assert plan["sample_count"] == 1030
    assert plan["blend"]["text_shards"] == 2
    assert "runs/cache/text_preservation_blended/text_targets/shard_0001.pt" in plan["blend"]["text_source_shards"]
    assert any("00_text_r00_s0001.pt" in command for command in plan["blend_link_commands"])
    assert any("00_text_r01_s0000.pt" in command for command in plan["blend_link_commands"])


def test_text_preservation_v5_plan_removes_sample_marker_and_resumes_from_v4() -> None:
    plan = build_text_preservation_v5_plan()

    assert plan["stage"] == "text_preservation_blended_v5_candidate"
    assert plan["sample_count"] == 1030
    assert plan["blend"]["text_repeat"] == 6
    assert plan["blend"]["general_shards"] == 4
    assert plan["blend"]["text_shards"] == 2
    assert "--no-sample-marker" in plan["prompt_write_command"]
    assert "--src-prefix text_preserve_v5" in plan["prompt_write_command"]
    assert plan["training_strategy"]["resume_from"] == (
        "runs/cache/text_preservation_blended_v4/bridge/text_preservation_blended_v4_bridge.pt"
    )
    assert "CUDA_VISIBLE_DEVICES='0'" in json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_text_preservation_v6_prompt_records_focus_hard_negative_targets() -> None:
    records = build_text_preservation_v6_prompt_records(count=80)
    targets = {record["target_text"] for record in records}

    assert len(records) == 80
    assert "sample " not in records[0]["text"]
    assert {"BRIGHT", "NOON BELL", "GREEN TEA", "QUIET HILL"} <= targets
    assert all(record["src"].startswith("text_preserve_v6_hard_") for record in records)
    assert len({record["text"] for record in records}) == 80


def test_text_preservation_v6_plan_resumes_from_v5_and_uses_hard_negative_writer() -> None:
    plan = build_text_preservation_v6_hard_negative_plan()

    assert plan["stage"] == "text_preservation_blended_v6_hard_negative_candidate"
    assert plan["sample_count"] == 326
    assert plan["training_strategy"]["resume_from"] == (
        "runs/cache/text_preservation_blended_v5/bridge/text_preservation_blended_v5_bridge.pt"
    )
    assert plan["training_strategy"]["method"] == "hard_negative_no_sample_marker_blend"
    assert "text-preservation-v6-prompts" in plan["prompt_write_command"]
    assert "--no-sample-marker" not in plan["prompt_write_command"]
    assert plan["blend"]["general_shards"] == 4
    assert plan["blend"]["text_repeat"] == 10
    assert "CUDA_VISIBLE_DEVICES='0'" in json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_cli_text_preservation_v6_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-v6-plan", "--count", "64", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_v6_hard_negative_candidate"
    assert payload["sample_count"] == 70
    assert "text-preservation-v6-prompts" in payload["prompt_write_command"]


def test_text_preservation_v7_balanced_plan_uses_v5_replay_hard_negative_and_all_general_shards() -> None:
    plan = build_text_preservation_v7_balanced_plan()

    assert plan["stage"] == "text_preservation_blended_v7_balanced_candidate"
    assert plan["training_strategy"]["method"] == "balanced_v5_replay_hard_negative_general_replay"
    assert plan["training_strategy"]["resume_from"] == (
        "runs/cache/text_preservation_blended_v5/bridge/text_preservation_blended_v5_bridge.pt"
    )
    assert plan["blend"]["v5_text_repeats"] == 4
    assert plan["blend"]["hard_negative_repeats"] == 1
    assert plan["blend"]["general_shards"] == 10
    assert any("00_v5_text_r00_s0000.pt" in command for command in plan["blend_link_commands"])
    assert any("05_hard_negative_r00_s0000.pt" in command for command in plan["blend_link_commands"])
    assert any("10_general_0009.pt" in command for command in plan["blend_link_commands"])
    assert "--resume-kv \"runs\\cache\\text_preservation_blended_v5\\bridge\\text_preservation_blended_v5_bridge.pt\"" in plan["train_command"]
    assert "--lr 1e-05" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='0'" in json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_cli_text_preservation_v7_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-v7-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_v7_balanced_candidate"
    assert payload["blend"]["general_shards"] == 10


def test_text_preservation_v8_plan_prioritizes_fixed_gate_replay() -> None:
    plan = build_text_preservation_v8_fixed_gate_plan()

    assert plan["stage"] == "text_preservation_blended_v8_fixed_gate_candidate"
    assert plan["training_strategy"]["method"] == "fixed_gate_preserving_conservative_replay"
    assert plan["training_strategy"]["resume_from"] == (
        "runs/cache/text_preservation_blended_v5/bridge/text_preservation_blended_v5_bridge.pt"
    )
    assert plan["blend"]["fixed_gate_repeats"] == 8
    assert plan["blend"]["v5_text_repeats"] == 2
    assert plan["blend"]["hard_negative_repeats"] == 1
    assert plan["blend"]["general_shards"] == 4
    assert any("00_fixed_gate_r00_s0000.pt" in command for command in plan["blend_link_commands"])
    assert any("10_v5_text_r00_s0000.pt" in command for command in plan["blend_link_commands"])
    assert any("20_hard_negative_r00_s0000.pt" in command for command in plan["blend_link_commands"])
    assert any("30_general_0003.pt" in command for command in plan["blend_link_commands"])
    assert "--epochs 1" in plan["train_command"]
    assert "--lr 5e-06" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='0'" in json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_cli_text_preservation_v8_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-v8-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_v8_fixed_gate_candidate"
    assert payload["blend"]["fixed_gate_repeats"] == 8


def test_text_preservation_promotion_status_rejects_fixed_gate_regression(tmp_path: Path) -> None:
    report_root = tmp_path / "fixed"
    report_root.mkdir()

    def write_report(student: str, index: int, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = report_root / f"text_eval_{index:03d}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    for index, mse in enumerate([0.01, 0.02], start=1):
        write_report("gemma_text_preservation_blended_v5", index, mse)
    for index, mse in enumerate([0.03, 0.04], start=1):
        write_report("gemma_text_preservation_blended_v7", index, mse)

    status = build_text_preservation_promotion_status(
        fixed_report_root=report_root,
        candidates={
            "v5": {"student_name": "gemma_text_preservation_blended_v5"},
            "v7": {"student_name": "gemma_text_preservation_blended_v7"},
        },
        baseline="v5",
    )

    assert status["baseline"] == "v5"
    assert status["candidates"]["v5"]["decision"]["status"] == "current_baseline"
    assert status["candidates"]["v7"]["fixed_gate"]["mean_mse"] == 0.035
    assert status["candidates"]["v7"]["decision"]["status"] == "reject"
    assert "fixed gate regressed" in status["candidates"]["v7"]["decision"]["reason"]


def test_text_preservation_promotion_status_recommends_protected_baseline(tmp_path: Path) -> None:
    report_root = tmp_path / "fixed"
    report_root.mkdir()

    def write_report(student: str, index: int, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = report_root / f"text_eval_{index:03d}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    for index, mse in enumerate([0.01, 0.02], start=1):
        write_report("gemma_text_preservation_blended_v5", index, mse)
    for index, mse in enumerate([0.03, 0.04], start=1):
        write_report("gemma_text_preservation_blended_v8", index, mse)

    status = build_text_preservation_promotion_status(
        fixed_report_root=report_root,
        candidates={
            "v5": {"student_name": "gemma_text_preservation_blended_v5"},
            "v8": {"student_name": "gemma_text_preservation_blended_v8"},
        },
        baseline="v5",
    )

    assert status["recommendation"] == {
        "status": "protect_baseline",
        "protected_baseline": "v5",
        "promote_candidate": None,
        "rejected_candidates": ["v8"],
        "pending_candidates": [],
        "reason": "no candidate beat the protected baseline fixed gate",
    }
    assert status["candidates"]["v8"]["failure_reasons"] == [
        "fixed_gate_mean_regression",
        "fixed_gate_max_regression",
        "fixed_gate_case_regression:text_eval_001",
        "fixed_gate_case_regression:text_eval_002",
    ]
    assert status["candidates"]["v8"]["required_artifacts"][0] == {
        "name": "fixed_gate_reports",
        "path": str(report_root / "*_qwen_vs_gemma_text_preservation_blended_v8_compare.json"),
        "expected_count": 2,
        "actual_count": 2,
        "exists": True,
    }


def test_text_preservation_promotion_status_rejects_per_case_regression_even_when_mean_improves(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "fixed"
    report_root.mkdir()

    def write_report(student: str, case_id: str, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = report_root / f"{case_id}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    write_report("gemma_text_preservation_blended_v5", "text_eval_001", 0.02)
    write_report("gemma_text_preservation_blended_v5", "text_eval_002", 0.02)
    write_report("gemma_text_preservation_blended_v9", "text_eval_001", 0.01)
    write_report("gemma_text_preservation_blended_v9", "text_eval_002", 0.021)

    status = build_text_preservation_promotion_status(
        fixed_report_root=report_root,
        candidates={
            "v5": {"student_name": "gemma_text_preservation_blended_v5"},
            "v9": {"student_name": "gemma_text_preservation_blended_v9"},
        },
        baseline="v5",
    )

    assert status["candidates"]["v9"]["fixed_gate"]["mean_mse"] == 0.0155
    assert status["candidates"]["v9"]["decision"]["status"] == "reject"
    assert status["candidates"]["v9"]["failure_reasons"] == [
        "fixed_gate_max_regression",
        "fixed_gate_case_regression:text_eval_002",
    ]


def test_text_preservation_promotion_status_tracks_v22_alpha28_by_default(tmp_path: Path) -> None:
    report_root = tmp_path / "fixed"
    report_root.mkdir()

    def write_report(student: str, case_id: str, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = report_root / f"{case_id}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    write_report("gemma_text_preservation_blended_v5", "text_eval_001", 0.02)
    write_report("gemma_text_preservation_blended_v22_alpha28", "text_eval_001", 0.01)

    status = build_text_preservation_promotion_status(
        fixed_report_root=report_root,
        baseline="v5",
    )

    assert status["candidates"]["v22_alpha28"]["student_name"] == "gemma_text_preservation_blended_v22_alpha28"
    assert status["candidates"]["v22_alpha28"]["decision"]["status"] == "eligible_for_next_gate"
    assert status["recommendation"]["promote_candidate"] == "v22_alpha28"
    assert status["recommendation"]["status"] == "candidate_needs_next_gate"


def test_cli_text_preservation_promotion_status_outputs_json(capsys) -> None:
    code = main(["text-preservation-promotion-status", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_promotion_status"
    assert payload["baseline"] == "v5"
    assert payload["recommendation"]["protected_baseline"] == "v5"


def test_cli_text_preservation_promotion_status_can_write_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "promotion_status.json"

    code = main(["text-preservation-promotion-status", "--output", str(output), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["output"] == str(output)
    assert written["stage"] == "text_preservation_promotion_status"
    assert written["recommendation"]["protected_baseline"] == "v5"


def test_cli_text_preservation_promotion_status_can_write_compact_report(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "promotion_status.json"
    compact_output = tmp_path / "promotion_status_compact.json"

    code = main(
        [
            "text-preservation-promotion-status",
            "--output",
            str(output),
            "--compact-output",
            str(compact_output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    compact = json.loads(compact_output.read_text(encoding="utf-8"))
    assert payload["compact_output"] == str(compact_output)
    assert compact["stage"] == "text_preservation_promotion_status_compact"
    assert compact["recommendation"]["protected_baseline"] == "v5"


def test_text_preservation_compact_promotion_status_keeps_review_sized_table(tmp_path: Path) -> None:
    report_root = tmp_path / "fixed"
    report_root.mkdir()

    def write_report(student: str, case_id: str, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = report_root / f"{case_id}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    write_report("gemma_text_preservation_blended_v5", "text_eval_001", 0.01)
    write_report("gemma_text_preservation_blended_v8", "text_eval_001", 0.03)
    status = build_text_preservation_promotion_status(
        fixed_report_root=report_root,
        candidates={
            "v5": {"student_name": "gemma_text_preservation_blended_v5"},
            "v8": {"student_name": "gemma_text_preservation_blended_v8"},
        },
        baseline="v5",
    )

    compact = build_text_preservation_compact_promotion_status(status)

    assert compact == {
        "stage": "text_preservation_promotion_status_compact",
        "recommendation": status["recommendation"],
        "candidates": [
            {
                "version": "v5",
                "student_name": "gemma_text_preservation_blended_v5",
                "decision": "current_baseline",
                "fixed_gate_mean_mse": 0.01,
                "fixed_gate_max_mse": 0.01,
                "failure_reasons": [],
            },
            {
                "version": "v8",
                "student_name": "gemma_text_preservation_blended_v8",
                "decision": "reject",
                "fixed_gate_mean_mse": 0.03,
                "fixed_gate_max_mse": 0.03,
                "failure_reasons": [
                    "fixed_gate_mean_regression",
                    "fixed_gate_max_regression",
                    "fixed_gate_case_regression:text_eval_001",
                ],
            },
        ],
    }


def test_text_preservation_release_gate_passes_protected_v5_with_required_artifacts(tmp_path: Path) -> None:
    fixed_root = tmp_path / "fixed"
    heldout_dir = tmp_path / "heldout"
    general_dir = tmp_path / "general"
    fixed_root.mkdir()
    heldout_dir.mkdir()
    general_dir.mkdir()

    def write_report(student: str, index: int, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = fixed_root / f"text_eval_{index:03d}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    for index, mse in enumerate([0.01, 0.02, 0.01, 0.01, 0.02, 0.01], start=1):
        write_report("gemma_text_preservation_blended_v5", index, mse)
    (heldout_dir / "metrics_summary.json").write_text(
        json.dumps({"case_count": 64, "mse": {"mean": 0.04}}),
        encoding="utf-8",
    )
    (heldout_dir / "visual_review.json").write_text(
        json.dumps({"counts": {"readable": 47, "partial": 13, "failed": 4}}),
        encoding="utf-8",
    )
    (general_dir / "metrics_summary.json").write_text(
        json.dumps({"case_count": 50}),
        encoding="utf-8",
    )
    (general_dir / "visual_review.json").write_text(
        json.dumps(
            {
                "decision": "pass_general_scene_smoke_expanded_50",
                "notes": ["No regression pattern indicating blank outputs or text-preservation overfit collapse."],
            }
        ),
        encoding="utf-8",
    )

    status = build_text_preservation_release_gate_status(
        fixed_report_root=fixed_root,
        candidate_specs={
            "v5": {
                "student_name": "gemma_text_preservation_blended_v5",
                "heldout_metrics": str(heldout_dir / "metrics_summary.json"),
                "heldout_review": str(heldout_dir / "visual_review.json"),
                "general_metrics": str(general_dir / "metrics_summary.json"),
                "general_review": str(general_dir / "visual_review.json"),
            }
        },
    )

    assert status["stage"] == "text_preservation_release_gate"
    assert status["executes_gpu_commands"] is False
    assert status["protected_baseline"] == "v5"
    assert status["release_gate"]["status"] == "pass"
    assert status["v9_training_gate"] == {
        "status": "blocked_until_objective_redesign",
        "reason": "release gate protects v5 and no candidate beat fixed image/text gates",
    }


def test_text_preservation_release_gate_fails_when_heldout_review_regresses(tmp_path: Path) -> None:
    fixed_root = tmp_path / "fixed"
    heldout_dir = tmp_path / "heldout"
    general_dir = tmp_path / "general"
    fixed_root.mkdir()
    heldout_dir.mkdir()
    general_dir.mkdir()
    (heldout_dir / "metrics_summary.json").write_text(json.dumps({"case_count": 64}), encoding="utf-8")
    (heldout_dir / "visual_review.json").write_text(
        json.dumps({"counts": {"readable": 44, "partial": 14, "failed": 6}}),
        encoding="utf-8",
    )
    (general_dir / "metrics_summary.json").write_text(json.dumps({"case_count": 50}), encoding="utf-8")
    (general_dir / "visual_review.json").write_text(
        json.dumps({"decision": "pass_general_scene_smoke_expanded_50"}),
        encoding="utf-8",
    )

    status = build_text_preservation_release_gate_status(
        fixed_report_root=fixed_root,
        candidate_specs={
            "v5": {
                "student_name": "gemma_text_preservation_blended_v5",
                "heldout_metrics": str(heldout_dir / "metrics_summary.json"),
                "heldout_review": str(heldout_dir / "visual_review.json"),
                "general_metrics": str(general_dir / "metrics_summary.json"),
                "general_review": str(general_dir / "visual_review.json"),
            }
        },
    )

    assert status["release_gate"]["status"] == "fail"
    assert "heldout_readable_below_minimum" in status["release_gate"]["failure_reasons"]
    assert "heldout_failed_above_maximum" in status["release_gate"]["failure_reasons"]


def test_text_preservation_release_gate_reports_promoted_candidate_next_gate_failures(tmp_path: Path) -> None:
    fixed_root = tmp_path / "fixed"
    baseline_heldout = tmp_path / "baseline_heldout"
    baseline_general = tmp_path / "baseline_general"
    candidate_heldout = tmp_path / "candidate_heldout"
    candidate_general = tmp_path / "candidate_general"
    for path in (fixed_root, baseline_heldout, baseline_general, candidate_heldout, candidate_general):
        path.mkdir()

    def write_report(student: str, index: int, mse: float) -> None:
        payload = {
            "images": {"teacher": "teacher.png", "student": "student.png"},
            "image_metrics": {"mse": mse, "psnr_db": 20.0},
        }
        path = fixed_root / f"text_eval_{index:03d}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    for index in range(1, 7):
        write_report("gemma_text_preservation_blended_v5", index, 0.02)
        write_report("gemma_text_preservation_blended_v22_alpha28", index, 0.01)
    (baseline_heldout / "metrics_summary.json").write_text(json.dumps({"case_count": 64}), encoding="utf-8")
    (baseline_heldout / "visual_review.json").write_text(
        json.dumps({"counts": {"readable": 47, "partial": 13, "failed": 4}}),
        encoding="utf-8",
    )
    (baseline_general / "metrics_summary.json").write_text(json.dumps({"case_count": 50}), encoding="utf-8")
    (baseline_general / "visual_review.json").write_text(
        json.dumps({"decision": "pass_general_scene_smoke_expanded_50"}),
        encoding="utf-8",
    )
    (candidate_heldout / "metrics_summary.json").write_text(json.dumps({"case_count": 64}), encoding="utf-8")
    (candidate_heldout / "visual_review.json").write_text(
        json.dumps({"counts": {"readable": 42, "partial": 17, "failed": 5}}),
        encoding="utf-8",
    )
    (candidate_general / "metrics_summary.json").write_text(json.dumps({"case_count": 50}), encoding="utf-8")
    (candidate_general / "visual_review.json").write_text(
        json.dumps({"decision": "pass_general_scene_smoke_expanded_50"}),
        encoding="utf-8",
    )

    status = build_text_preservation_release_gate_status(
        fixed_report_root=fixed_root,
        candidate_specs={
            "v5": {
                "student_name": "gemma_text_preservation_blended_v5",
                "heldout_metrics": str(baseline_heldout / "metrics_summary.json"),
                "heldout_review": str(baseline_heldout / "visual_review.json"),
                "general_metrics": str(baseline_general / "metrics_summary.json"),
                "general_review": str(baseline_general / "visual_review.json"),
            },
            "v22_alpha28": {
                "student_name": "gemma_text_preservation_blended_v22_alpha28",
                "heldout_metrics": str(candidate_heldout / "metrics_summary.json"),
                "heldout_review": str(candidate_heldout / "visual_review.json"),
                "general_metrics": str(candidate_general / "metrics_summary.json"),
                "general_review": str(candidate_general / "visual_review.json"),
            },
        },
    )

    assert status["promotion"]["recommendation"]["promote_candidate"] == "v22_alpha28"
    assert status["release_gate"]["status"] == "fail"
    assert "candidate_heldout_readable_below_minimum:v22_alpha28" in status["release_gate"]["failure_reasons"]
    assert "candidate_heldout_failed_above_maximum:v22_alpha28" in status["release_gate"]["failure_reasons"]


def test_cli_text_preservation_release_gate_outputs_json(capsys) -> None:
    code = main(["text-preservation-release-gate", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_release_gate"
    assert payload["protected_baseline"] == "v5"


def test_text_preservation_v9_objective_plan_keeps_training_blocked_until_redesign() -> None:
    plan = build_text_preservation_v9_objective_plan()

    assert plan["stage"] == "text_preservation_v9_objective_plan"
    assert plan["mode"] == "design_contract"
    assert plan["executes_gpu_commands"] is False
    assert plan["protected_baseline"] == "v5"
    assert plan["training_plan"] == {
        "status": "blocked",
        "train_command": None,
        "reason": "objective redesign required before GPU training",
    }
    assert plan["objective_redesign"]["recommended_approach"] == "artifact_gate_first"
    assert "fixed6_per_case_protection" in plan["objective_redesign"]["hard_preconditions"]
    encoded = json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES" not in encoded
    assert "train_bridge.py" not in encoded


def test_cli_text_preservation_v9_objective_plan_can_write_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "v9_objective_plan.json"

    code = main(["text-preservation-v9-objective-plan", "--output", str(output), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_v9_objective_plan"
    assert written["stage"] == "text_preservation_v9_objective_plan"
    assert written["training_plan"]["train_command"] is None
    assert written["next_safe_actions"][0]["action"] == "implement_artifact_gate_first_objective"


def test_text_preservation_v9_artifact_gate_objective_defines_non_gpu_training_contract() -> None:
    objective = build_text_preservation_v9_artifact_gate_objective()

    assert objective["stage"] == "text_preservation_v9_artifact_gate_objective"
    assert objective["mode"] == "objective_contract"
    assert objective["executes_gpu_commands"] is False
    assert objective["objective"]["method"] == "artifact_gate_first"
    assert objective["candidate_planning_permission"]["status"] == "allowed"
    assert objective["gpu_training_permission"] == {
        "status": "blocked_until_trainer_supports_artifact_feedback",
        "train_command": None,
    }
    assert objective["objective"]["artifact_gates"][0]["id"] == "fixed6_per_case_image_mse"
    encoded = json.dumps(objective, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES" not in encoded
    assert "train_bridge.py" not in encoded


def test_cli_text_preservation_v9_artifact_gate_objective_can_write_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "v9_artifact_gate_objective.json"

    code = main(["text-preservation-v9-artifact-gate-objective", "--output", str(output), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_v9_artifact_gate_objective"
    assert written["candidate_plan_contract"]["required_before_train_command"][0] == "trainer_artifact_feedback_support"


def test_text_preservation_v9_trainer_support_audit_blocks_without_artifact_feedback(tmp_path: Path) -> None:
    train_script = tmp_path / "train_bridge.py"
    train_script.write_text("def main():\n    pass\n", encoding="utf-8")

    audit = build_text_preservation_v9_trainer_support_audit(train_script=train_script)

    assert audit["stage"] == "text_preservation_v9_trainer_support_audit"
    assert audit["executes_gpu_commands"] is False
    assert audit["trainer_support"]["status"] == "missing_artifact_feedback_support"
    assert audit["gpu_training_permission"] == {
        "status": "blocked_until_trainer_supports_artifact_feedback",
        "train_command": None,
    }
    assert set(audit["trainer_support"]["missing_features"]) == {
        "artifact_feedback_dataset",
        "artifact_gate_loss_config",
        "post_render_metric_ingest",
        "kv_anchor_regularization",
        "source_bucket_feedback_filter",
    }


def test_cli_text_preservation_v9_trainer_support_audit_can_write_report(
    tmp_path: Path,
    capsys,
) -> None:
    train_script = tmp_path / "train_bridge.py"
    train_script.write_text("# --artifact-feedback\n# artifact_gate_loss_config\n", encoding="utf-8")
    output = tmp_path / "v9_trainer_support_audit.json"

    code = main(
        [
            "text-preservation-v9-trainer-support-audit",
            "--train-script",
            str(train_script),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_v9_trainer_support_audit"
    assert written["trainer_support"]["status"] == "missing_artifact_feedback_support"
    assert written["trainer_support"]["missing_features"] == [
        "post_render_metric_ingest",
        "kv_anchor_regularization",
        "source_bucket_feedback_filter",
    ]


def test_text_preservation_v9_artifact_feedback_dataset_weights_fixed_gate_cases(tmp_path: Path) -> None:
    report_root = tmp_path / "fixed"
    report_root.mkdir()
    student = "gemma_text_preservation_blended_v5"
    for index, mse in enumerate([0.01, 0.03], start=1):
        payload = {"image_metrics": {"mse": mse, "psnr_db": 20.0}}
        path = report_root / f"text_eval_{index:03d}_qwen_vs_{student}_compare.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_text_preservation_v9_artifact_feedback_dataset(fixed_report_root=report_root)

    assert dataset["stage"] == "text_preservation_v9_artifact_feedback_dataset"
    assert dataset["executes_gpu_commands"] is False
    assert [record["idx"] for record in dataset["records"]] == [0, 1]
    assert dataset["records"][1]["weight"] > dataset["records"][0]["weight"]
    assert dataset["records"][1]["readability_feedback"] == "partial"


def test_text_preservation_v9_candidate_plan_uses_artifact_feedback_and_4070_only() -> None:
    plan = build_text_preservation_v9_candidate_plan(
        trainer_audit={
            "trainer_support": {"status": "supported"},
            "gpu_training_permission": {"status": "allowed_to_build_candidate_train_command"},
        }
    )

    assert plan["stage"] == "text_preservation_blended_v9_artifact_gate_candidate"
    assert plan["executes_gpu_commands"] is True
    assert plan["training_strategy"]["method"] == "artifact_gate_first_feedback_weighted_replay"
    assert "--artifact-feedback" in plan["train_command"]
    assert "--artifact-gate-loss-config" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='0'" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_cli_text_preservation_v9_candidate_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-v9-candidate-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_v9_artifact_gate_candidate"
    assert payload["artifact_feedback"] == "reports/text_rendering_qwen_baseline/v9_artifact_feedback.jsonl"


def test_text_preservation_v10_candidate_plan_adds_v5_kv_anchor_and_4070_only() -> None:
    plan = build_text_preservation_v10_candidate_plan(
        trainer_audit={
            "trainer_support": {
                "status": "supported",
                "present_features": [
                    "artifact_feedback_dataset",
                    "artifact_gate_loss_config",
                    "post_render_metric_ingest",
                    "kv_anchor_regularization",
                ],
            },
            "gpu_training_permission": {"status": "allowed_to_build_candidate_train_command"},
        }
    )

    assert plan["stage"] == "text_preservation_blended_v10_protected_anchor_candidate"
    assert plan["executes_gpu_commands"] is True
    assert plan["training_strategy"]["method"] == "artifact_feedback_with_v5_kv_anchor"
    assert plan["anchor"]["checkpoint"] == "runs/cache/text_preservation_blended_v5/bridge/text_preservation_blended_v5_bridge.pt"
    assert plan["anchor"]["lambda"] == 0.1
    assert "--kv-anchor" in plan["train_command"]
    assert "--kv-anchor-lambda 0.1" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='0'" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_text_preservation_v10_candidate_plan_blocks_without_kv_anchor_support() -> None:
    plan = build_text_preservation_v10_candidate_plan(
        trainer_audit={
            "trainer_support": {
                "status": "supported",
                "present_features": [
                    "artifact_feedback_dataset",
                    "artifact_gate_loss_config",
                    "post_render_metric_ingest",
                ],
            },
            "gpu_training_permission": {"status": "allowed_to_build_candidate_train_command"},
        }
    )

    assert plan["mode"] == "blocked"
    assert plan["train_command"] is None
    assert plan["reason"] == "trainer kv anchor regularization support is required before v10 training"


def test_cli_text_preservation_v10_candidate_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-v10-candidate-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_v10_protected_anchor_candidate"
    assert payload["artifact_feedback"] == "reports/text_rendering_qwen_baseline/v10_artifact_feedback.jsonl"


def test_artifact_feedback_alignment_audit_detects_replay_source_spillover(tmp_path: Path) -> None:
    import torch

    target_dir = tmp_path / "blend_targets"
    target_dir.mkdir()
    torch.save([{"idx": 0}, {"idx": 1}], target_dir / "00_fixed_gate_r00_s0000.pt")
    torch.save([{"idx": 0}, {"idx": 1}, {"idx": 900000}], target_dir / "10_v5_text_r00_s0000.pt")
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(
        "\n".join(
            [
                json.dumps({"idx": 0, "case_id": "case0", "weight": 2.0}),
                json.dumps({"idx": 1, "case_id": "case1", "weight": 3.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_text_preservation_artifact_feedback_alignment_audit(
        blend_target_dir=target_dir,
        artifact_feedback=feedback,
    )

    assert audit["stage"] == "text_preservation_artifact_feedback_alignment_audit"
    assert audit["executes_gpu_commands"] is False
    assert audit["decision"]["status"] == "warn_feedback_spills_into_replay_sources"
    assert audit["weighted_occurrence_count"] == 4
    assert audit["source_buckets"]["00_fixed_gate"]["weighted_occurrences"] == 2
    assert audit["source_buckets"]["10_v5_text"]["weighted_occurrences"] == 2


def test_cli_artifact_feedback_alignment_audit_outputs_json(capsys) -> None:
    code = main(["text-preservation-artifact-feedback-alignment-audit", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_artifact_feedback_alignment_audit"


def test_text_preservation_v11_candidate_plan_filters_feedback_to_fixed_gate_source() -> None:
    plan = build_text_preservation_v11_candidate_plan(
        trainer_audit={
            "trainer_support": {
                "status": "supported",
                "present_features": [
                    "artifact_feedback_dataset",
                    "artifact_gate_loss_config",
                    "post_render_metric_ingest",
                    "kv_anchor_regularization",
                    "source_bucket_feedback_filter",
                ],
            },
            "gpu_training_permission": {"status": "allowed_to_build_candidate_train_command"},
        }
    )

    assert plan["stage"] == "text_preservation_blended_v11_source_filtered_candidate"
    assert plan["executes_gpu_commands"] is True
    assert plan["training_strategy"]["method"] == "fixed_gate_source_filtered_artifact_feedback"
    assert plan["artifact_feedback_source_buckets"] == ["00_fixed_gate"]
    assert plan["artifact_gate_loss_config"] == "reports/text_rendering_qwen_baseline/v11_artifact_gate_loss_config.json"
    assert "--kv-anchor" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='0'" in plan["train_command"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_text_preservation_v11_candidate_plan_blocks_without_source_filter_support() -> None:
    plan = build_text_preservation_v11_candidate_plan(
        trainer_audit={
            "trainer_support": {
                "status": "supported",
                "present_features": [
                    "artifact_feedback_dataset",
                    "artifact_gate_loss_config",
                    "post_render_metric_ingest",
                    "kv_anchor_regularization",
                ],
            },
            "gpu_training_permission": {"status": "allowed_to_build_candidate_train_command"},
        }
    )

    assert plan["mode"] == "blocked"
    assert plan["train_command"] is None
    assert plan["reason"] == "trainer source-bucket feedback filter support is required before v11 training"


def test_text_preservation_kv_delta_audit_measures_candidate_drift(tmp_path: Path) -> None:
    import torch

    baseline = tmp_path / "v5.pt"
    candidate = tmp_path / "v10.pt"
    torch.save({"kv": {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([2.0])}}, baseline)
    torch.save({"kv": {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([4.0])}}, candidate)

    audit = build_text_preservation_kv_delta_audit(
        baseline_checkpoint=baseline,
        candidate_checkpoints={"v10": candidate},
    )

    assert audit["stage"] == "text_preservation_kv_delta_audit"
    assert audit["executes_gpu_commands"] is False
    assert audit["baseline"]["kv_key_count"] == 2
    candidate_audit = audit["candidates"]["v10"]
    assert candidate_audit["status"] == "compared"
    assert candidate_audit["shared_kv_key_count"] == 2
    assert candidate_audit["mean_tensor_mse"] == 2.25
    assert round(candidate_audit["element_weighted_mse"], 6) == round(5.0 / 3.0, 6)
    assert candidate_audit["max_tensor_mse"] == 4.0
    assert candidate_audit["tensors"]["a"]["mse"] == 0.5


def test_cli_text_preservation_kv_delta_audit_writes_report(tmp_path: Path, capsys) -> None:
    import torch

    baseline = tmp_path / "v5.pt"
    candidate = tmp_path / "v11.pt"
    output = tmp_path / "kv_delta.json"
    torch.save({"kv": {"a": torch.tensor([0.0, 1.0])}}, baseline)
    torch.save({"kv": {"a": torch.tensor([0.0, 2.0])}}, candidate)

    code = main(
        [
            "text-preservation-kv-delta-audit",
            "--baseline-checkpoint",
            str(baseline),
            "--candidate-checkpoint",
            f"v11={candidate}",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_kv_delta_audit"
    assert written["candidates"]["v11"]["element_weighted_mse"] == 0.5


def test_text_preservation_v12_surface_plan_blocks_replay_only_training() -> None:
    kv_audit = {
        "summary": {
            "status": "compared",
            "anchor_effect": {
                "v10": {"status": "not_materially_changed_vs_v9"},
                "v11": {"status": "not_materially_changed_vs_v9"},
            },
        }
    }

    plan = build_text_preservation_v12_surface_plan(kv_delta_audit=kv_audit)

    assert plan["stage"] == "text_preservation_v12_training_surface_plan"
    assert plan["executes_gpu_commands"] is False
    assert plan["protected_baseline"] == "v5"
    assert plan["gpu_training_permission"]["status"] == "blocked_until_surface_redesign_artifacts_exist"
    assert plan["gpu_training_permission"]["train_command"] is None
    assert "replay_weight_only_training" in plan["forbidden_strategies"]
    assert "source_bucket_filter_only_training" in plan["forbidden_strategies"]
    assert plan["recommended_surface"]["id"] == "render_readability_conditioned_target_refresh"
    assert "render_readability_label_manifest" in plan["required_artifacts_before_training"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_cli_text_preservation_v12_surface_plan_writes_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "v12_plan.json"

    code = main(["text-preservation-v12-surface-plan", "--output", str(output), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_v12_training_surface_plan"
    assert written["workflow_position"]["current_step"] == "v12_surface_redesign"
    assert written["gpu_training_permission"]["train_command"] is None


def test_render_readability_label_manifest_merges_reviews_and_metrics(tmp_path: Path) -> None:
    fixed_compare = tmp_path / "fixed_case_qwen_vs_student_compare.json"
    fixed_compare.write_text(
        json.dumps(
            {
                "prompt": "draw a sign that clearly reads MOON",
                "images": {"teacher": "qwen/fixed.png", "student": "student/fixed.png"},
                "image_metrics": {"mse": 0.1, "psnr_db": 10.0},
            }
        ),
        encoding="utf-8",
    )
    fixed_review = tmp_path / "fixed_review.json"
    fixed_review.write_text(
        json.dumps(
            {
                "suite": "fixed",
                "decision": "protect_baseline",
                "fixed_gate": {"reports": [{"report": str(fixed_compare), "mse": 0.1, "psnr_db": 10.0}]},
            }
        ),
        encoding="utf-8",
    )
    heldout_compare = tmp_path / "text_preserve_heldout_clean_001_qwen_vs_student_compare.json"
    heldout_compare.write_text(
        json.dumps(
            {
                "prompt": "draw a label that clearly reads LIGHT",
                "images": {"teacher": "qwen/heldout.png", "student": "student/heldout.png"},
                "image_metrics": {"mse": 0.2, "psnr_db": 8.0},
            }
        ),
        encoding="utf-8",
    )
    heldout_review = tmp_path / "heldout_review.json"
    heldout_review.write_text(
        json.dumps({"counts": {"partial": 1}, "partial_indices": [1], "readable_indices": [], "failed_indices": []}),
        encoding="utf-8",
    )
    general_review = tmp_path / "general_review.json"
    general_review.write_text(json.dumps({"decision": "pass_general_scene_smoke_expanded_50"}), encoding="utf-8")

    manifest = build_text_preservation_render_readability_label_manifest(
        fixed_review=fixed_review,
        heldout_review=heldout_review,
        heldout_report_root=tmp_path,
        general_review=general_review,
    )

    assert manifest["stage"] == "text_preservation_render_readability_label_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["record_count"] == 2
    assert manifest["label_counts"]["accepted_baseline"] == 1
    assert manifest["label_counts"]["partial"] == 1
    partial = [record for record in manifest["records"] if record["readability_label"] == "partial"][0]
    assert partial["curriculum_role"] == "v12_priority_refresh"
    assert partial["target_text"] == "LIGHT"
    assert manifest["general_scene_guard"]["decision"] == "pass_general_scene_smoke_expanded_50"
    assert manifest["gpu_training_permission"]["train_command"] is None


def test_cli_render_readability_label_manifest_writes_report(tmp_path: Path, capsys) -> None:
    fixed_review = tmp_path / "fixed_review.json"
    fixed_review.write_text(json.dumps({"fixed_gate": {"reports": []}}), encoding="utf-8")
    heldout_review = tmp_path / "heldout_review.json"
    heldout_review.write_text(json.dumps({"readable_indices": [], "partial_indices": [], "failed_indices": []}), encoding="utf-8")
    output = tmp_path / "manifest.json"

    code = main(
        [
            "text-preservation-render-readability-label-manifest",
            "--fixed-review",
            str(fixed_review),
            "--heldout-review",
            str(heldout_review),
            "--heldout-report-root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_render_readability_label_manifest"
    assert written["workflow_position"]["current_step"] == "build_render_readability_label_manifest"


def test_surface_curriculum_manifest_prioritizes_partial_and_failed_records() -> None:
    label_manifest = {
        "records": [
            {
                "case_id": "fixed",
                "readability_label": "accepted_baseline",
                "curriculum_role": "fixed_gate_protection",
                "target_text": "MOON",
                "image_mse": 0.1,
            },
            {
                "case_id": "good",
                "readability_label": "readable",
                "curriculum_role": "readable_replay_guard",
                "target_text": "MAPLE",
                "image_mse": 0.02,
            },
            {
                "case_id": "partial",
                "readability_label": "partial",
                "curriculum_role": "v12_priority_refresh",
                "target_text": "LIGHT",
                "image_mse": 0.05,
            },
            {
                "case_id": "failed",
                "readability_label": "failed",
                "curriculum_role": "v12_priority_refresh",
                "target_text": "BRIGHT",
                "image_mse": 0.18,
            },
        ]
    }

    curriculum = build_text_preservation_surface_curriculum_manifest(label_manifest=label_manifest)

    assert curriculum["stage"] == "text_preservation_surface_curriculum_manifest"
    assert curriculum["executes_gpu_commands"] is False
    assert curriculum["priority_refresh_count"] == 2
    assert curriculum["curriculum_counts"]["failed_refresh"] == 1
    assert curriculum["curriculum_counts"]["partial_refresh"] == 1
    assert curriculum["curriculum"][0]["case_id"] == "failed"
    assert curriculum["curriculum"][0]["sample_weight"] > curriculum["curriculum"][1]["sample_weight"]
    assert curriculum["fixed_gate_guard_count"] == 1
    assert curriculum["gpu_training_permission"]["train_command"] is None
    assert curriculum["workflow_position"]["next_step"] == "build_qwen_target_refresh_manifest"


def test_cli_surface_curriculum_manifest_writes_report(tmp_path: Path, capsys) -> None:
    label_manifest = tmp_path / "labels.json"
    output = tmp_path / "curriculum.json"
    label_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_id": "failed",
                        "readability_label": "failed",
                        "curriculum_role": "v12_priority_refresh",
                        "target_text": "BRIGHT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-surface-curriculum-manifest",
            "--label-manifest",
            str(label_manifest),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_surface_curriculum_manifest"
    assert written["workflow_position"]["current_step"] == "build_surface_curriculum_manifest"


def test_qwen_target_refresh_manifest_emits_prompt_records_and_4070_command() -> None:
    curriculum_manifest = {
        "curriculum": [
            {
                "case_id": "failed",
                "target_text": "BRIGHT",
                "prompt": "draw a label that clearly reads BRIGHT",
                "curriculum_bucket": "failed_refresh",
                "sample_weight": 4.0,
            },
            {
                "case_id": "guard",
                "target_text": "MOON",
                "prompt": "draw a sign that clearly reads MOON",
                "curriculum_bucket": "fixed_gate_guard",
                "sample_weight": 1.0,
            },
        ]
    }

    manifest = build_text_preservation_qwen_target_refresh_manifest(curriculum_manifest=curriculum_manifest)

    assert manifest["stage"] == "text_preservation_qwen_target_refresh_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["prompt_record_count"] == 2
    assert manifest["prompt_records"][0]["text"] == "draw a label that clearly reads BRIGHT"
    assert manifest["prompt_records"][0]["target_text"] == "BRIGHT"
    assert manifest["prompt_records"][0]["curriculum_bucket"] == "failed_refresh"
    assert manifest["target_cache_command"].startswith("$env:CUDA_VISIBLE_DEVICES='0';")
    assert "06_cache_targets.py" in manifest["target_cache_command"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)
    assert manifest["gpu_training_permission"]["train_command"] is None
    assert manifest["workflow_position"]["next_step"] == "audit_trainer_surface_contract"


def test_cli_qwen_target_refresh_manifest_writes_report_and_prompt_jsonl(tmp_path: Path, capsys) -> None:
    curriculum = tmp_path / "curriculum.json"
    output = tmp_path / "target_refresh.json"
    prompt_file = tmp_path / "prompts.jsonl"
    curriculum.write_text(
        json.dumps(
            {
                "curriculum": [
                    {
                        "case_id": "failed",
                        "target_text": "BRIGHT",
                        "prompt": "draw a label that clearly reads BRIGHT",
                        "curriculum_bucket": "failed_refresh",
                        "sample_weight": 4.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-qwen-target-refresh-manifest",
            "--curriculum-manifest",
            str(curriculum),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_qwen_target_refresh_manifest"
    assert written["workflow_position"]["current_step"] == "build_qwen_target_refresh_manifest"
    assert prompt_rows[0]["target_text"] == "BRIGHT"


def test_v12_trainer_surface_contract_audit_blocks_when_surface_features_missing(tmp_path: Path) -> None:
    train_script = tmp_path / "train.py"
    train_script.write_text("# --artifact-feedback\n# readability_feedback\n", encoding="utf-8")

    audit = build_text_preservation_v12_trainer_surface_contract_audit(train_script=train_script)

    assert audit["stage"] == "text_preservation_v12_trainer_surface_contract_audit"
    assert audit["executes_gpu_commands"] is False
    assert audit["trainer_surface_contract"]["status"] == "missing_surface_contract_support"
    assert "surface_curriculum_manifest" in audit["trainer_surface_contract"]["missing_features"]
    assert "per_case_gate_loss_budget" in audit["trainer_surface_contract"]["missing_features"]
    assert audit["gpu_training_permission"]["train_command"] is None
    assert audit["workflow_position"]["current_step"] == "audit_trainer_surface_contract"


def test_cli_v12_trainer_surface_contract_audit_writes_report(tmp_path: Path, capsys) -> None:
    train_script = tmp_path / "train.py"
    output = tmp_path / "audit.json"
    train_script.write_text("# --surface-curriculum\n# per_case_gate_loss_budget\n# pre_train_promotion_gate\n# readability_label\n", encoding="utf-8")

    code = main(
        [
            "text-preservation-v12-trainer-surface-contract-audit",
            "--train-script",
            str(train_script),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_v12_trainer_surface_contract_audit"
    assert written["trainer_surface_contract"]["status"] == "supported"


def test_text_preservation_v13_recovery_plan_uses_v12_rejection(tmp_path: Path) -> None:
    status_path = tmp_path / "promotion_status.json"
    status_path.write_text(
        json.dumps(
            {
                "candidates": {
                    "v5": {
                        "fixed_gate": {
                            "mean_mse": 0.01,
                            "reports": [
                                {"case_id": "text_eval_001", "mse": 0.01},
                                {"case_id": "text_eval_002", "mse": 0.02},
                            ],
                        }
                    },
                    "v12": {
                        "fixed_gate": {
                            "mean_mse": 0.05,
                            "reports": [
                                {"case_id": "text_eval_001", "mse": 0.08, "report": "case1.json"},
                                {"case_id": "text_eval_002", "mse": 0.03, "report": "case2.json"},
                            ],
                        },
                        "failure_reasons": [
                            "fixed_gate_mean_regression",
                            "fixed_gate_case_regression:text_eval_001",
                        ],
                        "decision": {"status": "reject", "reason": "regressed"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    plan = build_text_preservation_v13_recovery_plan(promotion_status=status_path)

    assert plan["stage"] == "text_preservation_v13_recovery_plan"
    assert plan["executes_gpu_commands"] is False
    assert plan["diagnosis"]["status"] == "v12_rejected"
    assert plan["gpu_permission"]["status"] == "blocked_until_guard_manifest_exists"
    assert plan["v13_strategy"]["resume_from"].endswith("text_preservation_blended_v5_bridge.pt")
    assert plan["v13_strategy"]["do_not_resume_from"].endswith("text_preservation_blended_v12_bridge.pt")
    assert plan["gate_evidence"]["worst_regressions"][0]["case_id"] == "text_eval_001"
    assert plan["gate_evidence"]["worst_regressions"][0]["delta_mse"] == 0.07


def test_cli_text_preservation_v13_recovery_plan_writes_report(tmp_path: Path, capsys) -> None:
    status_path = tmp_path / "promotion_status.json"
    output = tmp_path / "v13_plan.json"
    status_path.write_text(
        json.dumps(
            {
                "candidates": {
                    "v5": {"fixed_gate": {"mean_mse": 0.01, "reports": [{"case_id": "a", "mse": 0.01}]}},
                    "v12": {
                        "fixed_gate": {"mean_mse": 0.04, "reports": [{"case_id": "a", "mse": 0.04}]},
                        "failure_reasons": ["fixed_gate_mean_regression"],
                        "decision": {"status": "reject"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-v13-recovery-plan",
            "--promotion-status",
            str(status_path),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "text_preservation_v13_recovery_plan"
    assert written["workflow_position"]["current_step"] == "v13_recovery_planning"


def test_v13_guard_weighted_manifest_caps_refresh_and_weights_fixed_guards() -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
                "sample_weight": 1.0,
            },
            {
                "text": "draw a book that reads STAR ATLAS",
                "source_case_id": "text_eval_002",
                "target_text": "STAR ATLAS",
                "curriculum_bucket": "fixed_gate_guard",
                "sample_weight": 1.0,
            },
            {
                "text": "draw a label that reads BRIGHT",
                "source_case_id": "failed",
                "target_text": "BRIGHT",
                "curriculum_bucket": "failed_refresh",
                "readability_label": "failed",
                "eval_idx": 0,
            },
            {
                "text": "draw a note that reads ROW 7",
                "source_case_id": "partial",
                "target_text": "ROW 7",
                "curriculum_bucket": "partial_refresh",
                "readability_label": "partial",
                "eval_idx": 1,
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "readable",
                "target_text": "TEA",
                "curriculum_bucket": "readable_replay_guard",
                "readability_label": "readable",
            },
        ]
    }

    manifest = build_text_preservation_v13_guard_weighted_manifest(source_manifest=source, max_refresh_records=1)

    assert manifest["stage"] == "text_preservation_v13_guard_weighted_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["selected_counts"]["fixed_gate_guard"] == 2
    assert manifest["selected_counts"]["capped_refresh_ablation"] == 1
    assert manifest["selected_weight_totals"]["fixed_gate_guard"] == 16.0
    assert manifest["selected_weight_totals"]["capped_refresh_ablation"] == 0.75
    assert manifest["target_cache_command"].startswith("$env:CUDA_VISIBLE_DEVICES='0';")
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)
    assert manifest["training_contract"]["resume_from"].endswith("text_preservation_blended_v5_bridge.pt")
    assert manifest["training_contract"]["do_not_resume_from"].endswith("text_preservation_blended_v12_bridge.pt")


def test_cli_v13_guard_weighted_manifest_writes_report_and_prompt_jsonl(tmp_path: Path, capsys) -> None:
    source_manifest = tmp_path / "source.json"
    output = tmp_path / "guard.json"
    prompt_file = tmp_path / "guard.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_records": [
                    {
                        "text": "draw a sign that reads LUNA GATE",
                        "source_case_id": "text_eval_001",
                        "target_text": "LUNA GATE",
                        "curriculum_bucket": "fixed_gate_guard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-v13-guard-weighted-manifest",
            "--source-manifest",
            str(source_manifest),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_v13_guard_weighted_manifest"
    assert written["workflow_position"]["current_step"] == "build_guard_weighted_v13_manifest"
    assert prompt_rows[0]["curriculum_bucket"] == "fixed_gate_guard"


def test_v14_focus_fixed_gate_manifest_uses_only_fixed_gate_and_focus_weights() -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001_sign_luna_gate",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "text_eval_006_label_tea",
                "target_text": "TEA",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a book that reads STAR ATLAS",
                "source_case_id": "text_eval_002_book_cover_star_atlas",
                "target_text": "STAR ATLAS",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a failed refresh",
                "source_case_id": "failed",
                "curriculum_bucket": "failed_refresh",
            },
        ]
    }

    manifest = build_text_preservation_v14_focus_fixed_gate_manifest(source_manifest=source)

    assert manifest["stage"] == "text_preservation_v14_focus_fixed_gate_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["prompt_record_count"] == 3
    assert manifest["selected_counts"]["v14_fixed_gate_focus"] == 2
    assert manifest["selected_counts"]["v14_fixed_gate_guard"] == 1
    assert manifest["selected_weight_totals"]["v14_fixed_gate_focus"] == 32.0
    assert manifest["selected_weight_totals"]["v14_fixed_gate_guard"] == 6.0
    assert manifest["training_contract"]["refresh_records"] == 0
    assert manifest["training_contract"]["max_lr"] == 5e-6
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)


def test_cli_v14_focus_fixed_gate_manifest_writes_report_and_prompt_jsonl(tmp_path: Path, capsys) -> None:
    source_manifest = tmp_path / "source.json"
    output = tmp_path / "v14.json"
    prompt_file = tmp_path / "v14.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_records": [
                    {
                        "text": "draw a sign that reads LUNA GATE",
                        "source_case_id": "text_eval_001_sign_luna_gate",
                        "target_text": "LUNA GATE",
                        "curriculum_bucket": "fixed_gate_guard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-v14-focus-fixed-gate-manifest",
            "--source-manifest",
            str(source_manifest),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--focus-case-id",
            "text_eval_001_sign_luna_gate",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_v14_focus_fixed_gate_manifest"
    assert written["workflow_position"]["current_step"] == "build_v14_focus_fixed_gate_manifest"
    assert prompt_rows[0]["curriculum_bucket"] == "v14_fixed_gate_focus"


def test_v17_targeted_teacher_refresh_manifest_uses_unique_stage_and_prompt_variants() -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001_sign_luna_gate",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a note that reads MEET AT DAWN",
                "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                "target_text": "MEET AT DAWN",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "text_eval_006_label_tea",
                "target_text": "TEA",
                "curriculum_bucket": "fixed_gate_guard",
            },
        ]
    }

    manifest = build_text_preservation_v17_targeted_teacher_refresh_manifest(
        source_manifest=source,
        focus_variant_count=2,
    )

    assert manifest["stage"] == "text_preservation_v17_targeted_teacher_refresh_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["workflow_position"]["current_step"] == "build_v17_targeted_teacher_refresh_manifest"
    assert manifest["workflow_position"]["next_step"] == "cache_v17_targeted_teacher_refresh_qwen_targets"
    assert manifest["selected_counts"] == {
        "v17_fixed_gate_guard": 3,
        "v17_targeted_teacher_refresh": 4,
    }
    assert manifest["training_contract"]["refresh_records"] == 4
    assert manifest["gpu_training_permission"]["train_command"] is None
    prompt_json = json.dumps(manifest["prompt_records"], ensure_ascii=False)
    assert "v13_guard" not in prompt_json
    assert "v14_fixed_gate" not in prompt_json
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)
    focus_rows = [
        row
        for row in manifest["prompt_records"]
        if row["curriculum_bucket"] == "v17_targeted_teacher_refresh"
    ]
    assert {row["target_text"] for row in focus_rows} == {"MEET AT DAWN", "TEA"}
    assert all(row["src"].startswith("v17_teacher_refresh_") for row in focus_rows)


def test_cli_v17_targeted_teacher_refresh_manifest_writes_report_and_prompt_jsonl(
    tmp_path: Path,
    capsys,
) -> None:
    source_manifest = tmp_path / "source.json"
    output = tmp_path / "v17.json"
    prompt_file = tmp_path / "v17.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_records": [
                    {
                        "text": "draw a note that reads MEET AT DAWN",
                        "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                        "target_text": "MEET AT DAWN",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                    {
                        "text": "draw a jar that reads TEA",
                        "source_case_id": "text_eval_006_label_tea",
                        "target_text": "TEA",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-v17-targeted-teacher-refresh-manifest",
            "--source-manifest",
            str(source_manifest),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--focus-variant-count",
            "1",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_v17_targeted_teacher_refresh_manifest"
    assert written["workflow_position"]["current_step"] == "build_v17_targeted_teacher_refresh_manifest"
    assert {row["target_text"] for row in prompt_rows} == {"MEET AT DAWN", "TEA"}
    assert any(row["curriculum_bucket"] == "v17_targeted_teacher_refresh" for row in prompt_rows)


def test_v18_tea_micro_refresh_manifest_focuses_tea_and_guards_v17_gain() -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001_sign_luna_gate",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a note that reads MEET AT DAWN",
                "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                "target_text": "MEET AT DAWN",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "text_eval_006_label_tea",
                "target_text": "TEA",
                "curriculum_bucket": "fixed_gate_guard",
            },
        ]
    }

    manifest = build_text_preservation_v18_tea_micro_refresh_manifest(
        source_manifest=source,
        focus_variant_count=3,
    )

    assert manifest["stage"] == "text_preservation_v18_tea_micro_refresh_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["workflow_position"]["previous_step"] == "v17_targeted_teacher_refresh_fixed6_rejected"
    assert manifest["workflow_position"]["current_step"] == "build_v18_tea_micro_refresh_manifest"
    assert manifest["selected_counts"] == {
        "v18_fixed_gate_guard": 3,
        "v18_v17_gain_guard": 1,
        "v18_tea_micro_refresh": 3,
    }
    assert manifest["training_contract"]["protected_baseline"] == "v5"
    assert manifest["training_contract"]["resume_from"].endswith("text_preservation_blended_v17_bridge.pt")
    assert manifest["training_contract"]["baseline_resume_from"].endswith("text_preservation_blended_v5_bridge.pt")
    assert manifest["training_contract"]["refresh_records"] == 3
    assert manifest["training_contract"]["v17_gain_guard_records"] == 1
    prompt_json = json.dumps(manifest["prompt_records"], ensure_ascii=False)
    assert "v13_guard" not in prompt_json
    assert "v14_fixed_gate" not in prompt_json
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)
    focus_rows = [
        row
        for row in manifest["prompt_records"]
        if row["curriculum_bucket"] == "v18_tea_micro_refresh"
    ]
    assert {row["target_text"] for row in focus_rows} == {"TEA"}
    assert all(row["src"].startswith("v18_tea_micro_refresh_") for row in focus_rows)
    assert all(row["teacher_refresh_strategy"] == "tea_only_micro_refresh_after_v17" for row in focus_rows)


def test_cli_v18_tea_micro_refresh_manifest_writes_report_and_prompt_jsonl(
    tmp_path: Path,
    capsys,
) -> None:
    source_manifest = tmp_path / "source.json"
    output = tmp_path / "v18.json"
    prompt_file = tmp_path / "v18.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_records": [
                    {
                        "text": "draw a note that reads MEET AT DAWN",
                        "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                        "target_text": "MEET AT DAWN",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                    {
                        "text": "draw a jar that reads TEA",
                        "source_case_id": "text_eval_006_label_tea",
                        "target_text": "TEA",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-v18-tea-micro-refresh-manifest",
            "--source-manifest",
            str(source_manifest),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--focus-variant-count",
            "2",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_v18_tea_micro_refresh_manifest"
    assert written["workflow_position"]["current_step"] == "build_v18_tea_micro_refresh_manifest"
    assert any(row["curriculum_bucket"] == "v18_tea_micro_refresh" for row in prompt_rows)
    assert {row["target_text"] for row in prompt_rows if row["curriculum_bucket"] == "v18_tea_micro_refresh"} == {
        "TEA"
    }


def test_v19_dual_guard_refresh_manifest_restarts_from_v5_with_meet_tea_focus() -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001_sign_luna_gate",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a book that reads STAR ATLAS",
                "source_case_id": "text_eval_002_book_cover_star_atlas",
                "target_text": "STAR ATLAS",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a note that reads MEET AT DAWN",
                "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                "target_text": "MEET AT DAWN",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "text_eval_006_label_tea",
                "target_text": "TEA",
                "curriculum_bucket": "fixed_gate_guard",
            },
        ]
    }

    manifest = build_text_preservation_v19_dual_guard_refresh_manifest(
        source_manifest=source,
        focus_variant_count=2,
    )

    assert manifest["stage"] == "text_preservation_v19_dual_guard_refresh_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["workflow_position"]["previous_step"] == "v18_tea_micro_refresh_fixed6_rejected"
    assert manifest["workflow_position"]["current_step"] == "build_v19_dual_guard_refresh_manifest"
    assert manifest["selected_counts"] == {
        "v19_fixed_gate_guard": 4,
        "v19_stability_guard": 3,
        "v19_meet_tea_refresh": 4,
    }
    assert manifest["training_contract"]["protected_baseline"] == "v5"
    assert manifest["training_contract"]["resume_from"].endswith("text_preservation_blended_v5_bridge.pt")
    assert manifest["training_contract"]["do_not_resume_from"][0].endswith("text_preservation_blended_v17_bridge.pt")
    assert manifest["training_contract"]["do_not_resume_from"][1].endswith("text_preservation_blended_v18_bridge.pt")
    assert manifest["training_contract"]["refresh_records"] == 4
    assert manifest["training_contract"]["stability_guard_records"] == 3
    focus_rows = [
        row
        for row in manifest["prompt_records"]
        if row["curriculum_bucket"] == "v19_meet_tea_refresh"
    ]
    assert {row["target_text"] for row in focus_rows} == {"MEET AT DAWN", "TEA"}
    assert all(row["src"].startswith("v19_meet_tea_refresh_") for row in focus_rows)
    assert all(row["teacher_refresh_strategy"] == "dual_guard_refresh_from_v5_after_v18" for row in focus_rows)
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)


def test_cli_v19_dual_guard_refresh_manifest_writes_report_and_prompt_jsonl(
    tmp_path: Path,
    capsys,
) -> None:
    source_manifest = tmp_path / "source.json"
    output = tmp_path / "v19.json"
    prompt_file = tmp_path / "v19.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_records": [
                    {
                        "text": "draw a book that reads STAR ATLAS",
                        "source_case_id": "text_eval_002_book_cover_star_atlas",
                        "target_text": "STAR ATLAS",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                    {
                        "text": "draw a note that reads MEET AT DAWN",
                        "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                        "target_text": "MEET AT DAWN",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                    {
                        "text": "draw a jar that reads TEA",
                        "source_case_id": "text_eval_006_label_tea",
                        "target_text": "TEA",
                        "curriculum_bucket": "fixed_gate_guard",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "text-preservation-v19-dual-guard-refresh-manifest",
            "--source-manifest",
            str(source_manifest),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--focus-variant-count",
            "1",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_v19_dual_guard_refresh_manifest"
    assert written["workflow_position"]["current_step"] == "build_v19_dual_guard_refresh_manifest"
    assert any(row["curriculum_bucket"] == "v19_meet_tea_refresh" for row in prompt_rows)
    assert {row["target_text"] for row in prompt_rows if row["curriculum_bucket"] == "v19_meet_tea_refresh"} == {
        "MEET AT DAWN",
        "TEA",
    }


def test_v23_hard_heldout_refresh_manifest_targets_v22_failed_and_partial_cases(tmp_path: Path) -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001_sign_luna_gate",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "text_eval_006_label_tea",
                "target_text": "TEA",
                "curriculum_bucket": "fixed_gate_guard",
            },
        ]
    }
    heldout_review = tmp_path / "visual_review.json"
    heldout_review.write_text(
        json.dumps(
            {
                "failed_indices": [25, 59],
                "partial_indices": [1, 5, 9],
                "readable_indices": [0, 2],
            }
        ),
        encoding="utf-8",
    )
    heldout_prompts = tmp_path / "prompts.jsonl"
    prompt_rows = [
        {"text": f"draw heldout {index}", "src": f"text_preserve_heldout_clean_{index:03d}", "target_text": f"T{index}"}
        for index in range(60)
    ]
    heldout_prompts.write_text(
        "\n".join(json.dumps(row) for row in prompt_rows) + "\n",
        encoding="utf-8",
    )

    manifest = build_text_preservation_v23_hard_heldout_refresh_manifest(
        source_manifest=source,
        heldout_review=heldout_review,
        heldout_prompts=heldout_prompts,
        max_partial_records=2,
    )

    assert manifest["stage"] == "text_preservation_v23_hard_heldout_refresh_manifest"
    assert manifest["executes_gpu_commands"] is False
    assert manifest["workflow_position"]["previous_step"] == "v22_alpha28_failed_heldout_promotion_gate"
    assert manifest["selected_counts"] == {
        "v23_fixed_gate_guard": 2,
        "v23_failed_heldout_refresh": 2,
        "v23_partial_heldout_refresh": 2,
    }
    assert manifest["training_contract"]["protected_baseline"] == "v5"
    assert manifest["training_contract"]["resume_from"].endswith("text_preservation_blended_v5_bridge.pt")
    assert manifest["training_contract"]["do_not_resume_from"].endswith(
        "text_preservation_blended_v22_alpha28_bridge.pt"
    )
    assert manifest["target_cache_command"].startswith("$env:CUDA_VISIBLE_DEVICES='0';")
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)
    focus_rows = [
        row
        for row in manifest["prompt_records"]
        if row["curriculum_bucket"] in {"v23_failed_heldout_refresh", "v23_partial_heldout_refresh"}
    ]
    assert {row["source_case_id"] for row in focus_rows} == {
        "text_preserve_heldout_clean_001",
        "text_preserve_heldout_clean_005",
        "text_preserve_heldout_clean_025",
        "text_preserve_heldout_clean_059",
    }
    assert all(row["teacher_refresh_strategy"] == "hard_heldout_refresh_after_v22_alpha28" for row in focus_rows)


def test_cli_v23_hard_heldout_refresh_manifest_writes_report_and_prompt_jsonl(
    tmp_path: Path, capsys
) -> None:
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_records": [
                    {
                        "text": "draw a sign that reads LUNA GATE",
                        "source_case_id": "text_eval_001_sign_luna_gate",
                        "target_text": "LUNA GATE",
                        "curriculum_bucket": "fixed_gate_guard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    heldout_review = tmp_path / "visual_review.json"
    heldout_review.write_text(
        json.dumps({"failed_indices": [25], "partial_indices": [1, 5]}),
        encoding="utf-8",
    )
    heldout_prompts = tmp_path / "prompts.jsonl"
    heldout_prompts.write_text(
        "\n".join(
            json.dumps(
                {
                    "text": f"draw heldout {index}",
                    "src": f"text_preserve_heldout_clean_{index:03d}",
                    "eval_idx": index,
                    "target_text": f"T{index}",
                }
            )
            for index in range(26)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "v23.json"
    prompt_file = tmp_path / "v23.jsonl"

    code = main(
        [
            "text-preservation-v23-hard-heldout-refresh-manifest",
            "--source-manifest",
            str(source_manifest),
            "--heldout-review",
            str(heldout_review),
            "--heldout-prompts",
            str(heldout_prompts),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--max-partial-records",
            "1",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    prompt_rows = [json.loads(line) for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    assert payload["stage"] == "text_preservation_v23_hard_heldout_refresh_manifest"
    assert written["workflow_position"]["current_step"] == "build_v23_hard_heldout_refresh_manifest"
    assert written["selected_counts"] == {
        "v23_fixed_gate_guard": 1,
        "v23_failed_heldout_refresh": 1,
        "v23_partial_heldout_refresh": 1,
    }
    assert {row["curriculum_bucket"] for row in prompt_rows} == {
        "v23_fixed_gate_guard",
        "v23_failed_heldout_refresh",
        "v23_partial_heldout_refresh",
    }


def test_v24_fixed_gate_protected_heldout_refresh_reweights_meet_tea_regressions(
    tmp_path: Path,
) -> None:
    source = {
        "prompt_records": [
            {
                "text": "draw a note that reads MEET AT DAWN",
                "source_case_id": "text_eval_005_handwritten_note_meet_at_dawn",
                "target_text": "MEET AT DAWN",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a jar that reads TEA",
                "source_case_id": "text_eval_006_label_tea",
                "target_text": "TEA",
                "curriculum_bucket": "fixed_gate_guard",
            },
            {
                "text": "draw a sign that reads LUNA GATE",
                "source_case_id": "text_eval_001_sign_luna_gate",
                "target_text": "LUNA GATE",
                "curriculum_bucket": "fixed_gate_guard",
            },
        ]
    }
    heldout_review = tmp_path / "visual_review.json"
    heldout_review.write_text(
        json.dumps({"failed_indices": [44, 59], "partial_indices": [5, 14, 17]}),
        encoding="utf-8",
    )
    heldout_prompts = tmp_path / "prompts.jsonl"
    heldout_prompts.write_text(
        "\n".join(
            json.dumps(
                {
                    "text": f"draw heldout {index}",
                    "src": f"text_preserve_heldout_clean_{index:03d}",
                    "eval_idx": index,
                    "target_text": f"T{index}",
                }
            )
            for index in range(60)
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest(
        source_manifest=source,
        heldout_review=heldout_review,
        heldout_prompts=heldout_prompts,
        max_partial_records=2,
        focus_variant_count=1,
    )

    assert manifest["stage"] == "text_preservation_v24_fixed_gate_protected_heldout_refresh_manifest"
    assert manifest["workflow_position"]["previous_step"] == "v23_rejected_fixed_gate_regression"
    assert manifest["selected_counts"] == {
        "v24_fixed_gate_guard": 3,
        "v24_fixed_gate_regression_focus": 2,
        "v24_fixed_gate_regression_variant": 2,
        "v24_failed_heldout_refresh": 2,
        "v24_partial_heldout_refresh": 2,
    }
    assert manifest["training_contract"]["protected_baseline"] == "v5"
    assert manifest["training_contract"]["resume_from"].endswith("text_preservation_blended_v5_bridge.pt")
    assert any(
        item.endswith("text_preservation_blended_v23_bridge.pt")
        for item in manifest["training_contract"]["do_not_resume_from"]
    )
    assert manifest["training_contract"]["artifact_gate_loss_config"]["max_weight"] == 28.0
    focus_rows = [
        row
        for row in manifest["prompt_records"]
        if row["curriculum_bucket"] == "v24_fixed_gate_regression_focus"
    ]
    assert {row["target_text"] for row in focus_rows} == {"MEET AT DAWN", "TEA"}
    assert all(float(row["sample_weight"]) == 28.0 for row in focus_rows)
    assert all(
        row["teacher_refresh_strategy"] == "fixed_gate_protection_after_v23"
        for row in manifest["prompt_records"]
        if row["curriculum_bucket"] == "v24_fixed_gate_regression_variant"
    )
    assert manifest["target_cache_command"].startswith("$env:CUDA_VISIBLE_DEVICES='0';")
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(manifest, ensure_ascii=False)


def test_v21_text_roi_gate_report_is_observer_only(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.png"
    baseline_student = tmp_path / "baseline.png"
    candidate_student = tmp_path / "candidate.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(teacher)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(baseline_student)
    edited = Image.open(candidate_student if candidate_student.exists() else baseline_student)
    edited.putpixel((2, 2), (255, 255, 255))
    edited.save(candidate_student)
    baseline_report = tmp_path / "case_qwen_vs_baseline_compare.json"
    candidate_report = tmp_path / "case_qwen_vs_candidate_compare.json"
    baseline_report.write_text(
        json.dumps({"images": {"teacher": str(teacher), "student": str(baseline_student)}}),
        encoding="utf-8",
    )
    candidate_report.write_text(
        json.dumps({"images": {"teacher": str(teacher), "student": str(candidate_student)}}),
        encoding="utf-8",
    )

    report = build_text_preservation_v21_text_roi_gate_report(
        baseline_reports=[baseline_report],
        candidate_specs={
            "candidate": {
                "student_name": "candidate",
                "reports": [candidate_report],
            }
        },
        roi_map={"case": {"name": "center_2x2", "box": [1, 1, 3, 3]}},
    )

    assert report["stage"] == "text_preservation_v21_text_roi_gate_report"
    assert report["executes_gpu_commands"] is False
    assert report["promotion_effect"] == "none_observer_only"
    assert report["baseline"]["case"]["text_roi_metrics"]["mse"] == 0.0
    assert report["candidates"]["candidate"]["cases"]["case"]["text_roi_metrics"]["mse"] == 0.25
    assert report["candidates"]["candidate"]["failure_reasons"] == ["text_roi_case_regression:case"]
    assert report["recommendation"]["protected_baseline"] == "v5"


def test_cli_text_preservation_blended_plan_outputs_json(capsys) -> None:
    code = main(
        [
            "text-preservation-blended-plan",
            "--root",
            r"runs\cache\custom_blend",
            "--output",
            r"runs\cache\custom_blend\bridge\custom.pt",
            "--text-repeat",
            "8",
            "--general-shards",
            "2",
            "--count",
            "512",
            "--resume-kv",
            r"runs\cache\text_preservation_blended_v3\bridge\text_preservation_blended_v3_bridge.pt",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_candidate"
    assert payload["root"] == "runs/cache/custom_blend"
    assert payload["output"] == "runs/cache/custom_blend/bridge/custom.pt"
    assert payload["sample_count"] == 518
    assert payload["blend"]["text_repeat"] == 8
    assert payload["blend"]["general_shards"] == 2
    assert payload["training_strategy"]["resume_from"] == "runs/cache/text_preservation_blended_v3/bridge/text_preservation_blended_v3_bridge.pt"


def test_cli_text_preservation_v5_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-v5-plan", "--count", "128", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_v5_candidate"
    assert payload["sample_count"] == 134
    assert "--no-sample-marker" in payload["prompt_write_command"]


def test_cli_text_preservation_blended_status_accepts_custom_paths(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "targets"
    gemma = tmp_path / "gemma"
    bridge = tmp_path / "bridge.pt"
    target.mkdir()
    gemma.mkdir()
    (target / "same.pt").write_bytes(b"placeholder")
    (gemma / "same.pt").write_bytes(b"placeholder")

    code = main(
        [
            "text-preservation-blended-status",
            "--target-dir",
            str(target),
            "--gemma-dir",
            str(gemma),
            "--bridge-checkpoint",
            str(bridge),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_blended_candidate"
    assert payload["ready_for_training"] is True
    assert payload["bridge"]["exists"] is False


def test_text_preservation_heldout_eval_plan_uses_v4_checkpoint_and_4070_only() -> None:
    plan = build_text_preservation_heldout_eval_plan(count=2)

    assert plan["stage"] == "text_preservation_heldout_eval"
    assert plan["prompt_file"] == "reports/text_preservation_heldout_v4/prompts.jsonl"
    assert plan["student_checkpoint"] == "runs/cache/text_preservation_blended_v4/bridge/text_preservation_blended_v4_bridge.pt"
    assert plan["total_cases"] == 2
    assert plan["gpu_policy"]["cuda_visible_devices"] == "0"
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)
    assert "--mode qwen" in plan["qwen_command"]
    assert "--mode gemma" in plan["gemma_command"]
    assert "--adapter \"runs\\cache\\text_preservation_blended_v4\\bridge\\text_preservation_blended_v4_bridge.pt\"" in plan["gemma_command"]
    assert plan["cases"][0]["compare_report"] == (
        "reports/text_preservation_heldout_v4/text_preserve_heldout_000_qwen_vs_gemma_text_preservation_blended_v4_compare.json"
    )


def test_text_preservation_heldout_eval_plan_accepts_clean_v5_paths() -> None:
    plan = build_text_preservation_heldout_eval_plan(
        count=2,
        prompt_file=r"reports\text_preservation_heldout_v5_clean\prompts.jsonl",
        out_root=r"runs\images\text_preservation_heldout_v5_clean",
        report_root=r"reports\text_preservation_heldout_v5_clean",
        student_checkpoint=r"runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt",
        student_name="gemma_text_preservation_blended_v5",
        prompt_index_offset=30000,
        src_prefix="text_preserve_heldout_clean",
        include_sample_marker=False,
    )

    assert plan["prompt_file"] == "reports/text_preservation_heldout_v5_clean/prompts.jsonl"
    assert plan["student_name"] == "gemma_text_preservation_blended_v5"
    assert "--no-sample-marker" in plan["prompt_write_command"]
    assert "--src-prefix text_preserve_heldout_clean" in plan["prompt_write_command"]
    assert "sample " not in plan["cases"][0]["prompt"]


def test_cli_text_preservation_heldout_eval_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-heldout-eval-plan", "--count", "2", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_preservation_heldout_eval"
    assert payload["total_cases"] == 2
