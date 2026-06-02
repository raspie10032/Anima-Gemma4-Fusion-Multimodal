import json
import pytest

from gemmanima.cli import main
from gemmanima.training.text_rendering_eval import (
    build_text_rendering_eval_execution_plan,
    build_text_rendering_eval_pack,
    build_text_rendering_qwen_baseline_plan,
    build_text_rendering_qwen_prompt_records,
    build_text_rendering_eval_run_plan,
    build_text_rendering_eval_status,
)


def test_text_rendering_eval_pack_covers_required_categories() -> None:
    pack = build_text_rendering_eval_pack()
    categories = {item["category"] for item in pack["prompts"]}

    assert pack["stage"] == "text_rendering_preservation"
    assert {"sign", "book_cover", "magic_circle", "UI_panel", "handwritten_note", "label"} <= categories
    assert all(item["target_text"] for item in pack["prompts"])


def test_text_rendering_eval_prompts_are_object_only_text_tests() -> None:
    pack = build_text_rendering_eval_pack()

    for item in pack["prompts"]:
        prompt = item["prompt"].lower()
        assert "object-only text rendering test" in prompt
        assert "no people" in prompt
        assert "no characters" in prompt


def test_text_rendering_eval_pack_includes_deterministic_reporting_contract() -> None:
    pack = build_text_rendering_eval_pack()
    first = pack["prompts"][0]

    assert first["id"] == "text_eval_001_sign_luna_gate"
    assert first["seed"] == 440001
    assert first["artifacts"] == {
        "teacher_image": "runs/images/text_rendering_eval/text_eval_001_sign_luna_gate_teacher.png",
        "student_image": "runs/images/text_rendering_eval/text_eval_001_sign_luna_gate_student.png",
        "compare_report": "reports/text_rendering_eval/text_eval_001_sign_luna_gate_compare.json",
    }
    assert pack["reporting"] == {
        "version": "0.1",
        "image_dir": "runs/images/text_rendering_eval",
        "report_dir": "reports/text_rendering_eval",
        "comparison_metrics": ["mse", "mae", "psnr_db", "ocr_exact_text_match"],
        "cases": [
            {
                "id": item["id"],
                "target_text": item["target_text"],
                "seed": item["seed"],
                "teacher_image": item["artifacts"]["teacher_image"],
                "student_image": item["artifacts"]["student_image"],
                "compare_report": item["artifacts"]["compare_report"],
            }
            for item in pack["prompts"]
        ],
    }


def test_cli_text_rendering_eval_pack_outputs_json(capsys) -> None:
    code = main(["text-rendering-eval-pack", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_rendering_preservation"
    assert len(payload["prompts"]) >= 6


def test_text_rendering_eval_status_marks_missing_artifacts_pending(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    status = build_text_rendering_eval_status()

    assert status["ready"] is False
    assert status["summary"] == {"total_cases": 6, "ready_cases": 0, "pending_cases": 6}
    assert status["metrics_policy"] == "observed_artifacts_only"
    assert status["cases"][0]["status"] == "pending"
    assert status["cases"][0]["artifacts"]["teacher_image"]["exists"] is False
    assert status["cases"][0]["metrics"] == {}


def test_text_rendering_eval_status_reads_existing_compare_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pack = build_text_rendering_eval_pack()
    first = pack["prompts"][0]
    artifacts = first["artifacts"]

    for key in ("teacher_image", "student_image"):
        path = tmp_path / artifacts[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    report_path = tmp_path / artifacts["compare_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "image_metrics": {"mse": 0.12, "mae": 0.24},
                "ocr_metrics": {"exact_text_match": True},
            }
        ),
        encoding="utf-8",
    )

    status = build_text_rendering_eval_status()

    assert status["ready"] is False
    assert status["summary"] == {"total_cases": 6, "ready_cases": 1, "pending_cases": 5}
    assert status["cases"][0]["status"] == "ready"
    assert status["cases"][0]["metrics"] == {
        "image_metrics": {"mse": 0.12, "mae": 0.24},
        "ocr_metrics": {"exact_text_match": True},
    }


def test_text_rendering_eval_status_ignores_null_conditioning_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pack = build_text_rendering_eval_pack()
    first = pack["prompts"][0]
    artifacts = first["artifacts"]

    for key in ("teacher_image", "student_image"):
        path = tmp_path / artifacts[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    report_path = tmp_path / artifacts["compare_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "conditioning": {"mse": None},
                "image_metrics": {"mse": 0.12},
            }
        ),
        encoding="utf-8",
    )

    status = build_text_rendering_eval_status()

    assert status["cases"][0]["metrics"] == {
        "image_metrics": {"mse": 0.12},
    }


def test_cli_text_rendering_eval_status_outputs_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["text-rendering-eval-status", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_rendering_preservation"
    assert payload["summary"]["total_cases"] == 6


def test_cli_text_rendering_eval_plan_outputs_json(capsys) -> None:
    code = main(["text-rendering-eval-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_rendering_preservation"
    assert payload["mode"] == "dry_run"
    assert payload["executes_gpu_commands"] is False
    assert payload["total_cases"] == 6


def test_text_rendering_eval_execution_plan_is_deterministic_and_json_serializable() -> None:
    plan = build_text_rendering_eval_execution_plan()
    encoded = json.dumps(plan, sort_keys=True)
    expected_prompt = (
        "Object-only text rendering test: draw a close-up moonlit street sign that clearly reads LUNA GATE. "
        "No people, no characters, no faces, no bodies; keep the sign centered and large enough to inspect the letters."
    )

    assert json.loads(encoded) == plan
    assert plan["stage"] == "text_rendering_preservation"
    assert plan["mode"] == "dry_run"
    assert plan["executes_gpu_commands"] is False
    assert plan["total_cases"] == 6
    assert len(plan["cases"]) == 6
    assert plan["cases"][0] == {
        "id": "text_eval_001_sign_luna_gate",
        "category": "sign",
        "target_text": "LUNA GATE",
        "seed": 440001,
        "teacher": {
            "prompt": expected_prompt,
            "seed": 440001,
            "output_path": "runs/images/text_rendering_eval/text_eval_001_sign_luna_gate_teacher.png",
        },
        "student": {
            "prompt": expected_prompt,
            "seed": 440001,
            "output_path": "runs/images/text_rendering_eval/text_eval_001_sign_luna_gate_student.png",
        },
        "comparison": {
            "teacher_image": "runs/images/text_rendering_eval/text_eval_001_sign_luna_gate_teacher.png",
            "student_image": "runs/images/text_rendering_eval/text_eval_001_sign_luna_gate_student.png",
            "compare_report": "reports/text_rendering_eval/text_eval_001_sign_luna_gate_compare.json",
            "target_text": "LUNA GATE",
            "metrics": ["mse", "mae", "psnr_db", "ocr_exact_text_match"],
        },
    }


def test_text_rendering_eval_execution_plan_reuses_pack_contract() -> None:
    pack = build_text_rendering_eval_pack()
    plan = build_text_rendering_eval_execution_plan(pack)

    planned_ids = [case["id"] for case in plan["cases"]]
    pack_ids = [case["id"] for case in pack["prompts"]]

    assert planned_ids == pack_ids
    assert plan["artifact_policy"] == "declare_paths_only"
    for source, planned in zip(pack["prompts"], plan["cases"]):
        assert planned["teacher"]["output_path"] == source["artifacts"]["teacher_image"]
        assert planned["student"]["output_path"] == source["artifacts"]["student_image"]
        assert planned["comparison"]["compare_report"] == source["artifacts"]["compare_report"]


def test_text_rendering_eval_run_plan_emits_4070_only_executable_commands() -> None:
    plan = build_text_rendering_eval_run_plan(max_cases=1)

    assert plan["mode"] == "executable"
    assert plan["executes_gpu_commands"] is True
    assert plan["gpu_policy"] == {
        "cuda_visible_devices": "0",
        "gpu_name": "RTX 4070 Ti SUPER",
        "reserved_gpu": "RTX 5060 / CUDA device 1",
    }
    assert plan["total_cases"] == 1
    encoded = json.dumps(plan, ensure_ascii=False)
    assert "5060" in encoded
    assert "CUDA_VISIBLE_DEVICES='1'" not in encoded
    case = plan["cases"][0]
    assert case["teacher"]["command"].startswith("$env:CUDA_VISIBLE_DEVICES='0'; & ")
    assert case["student"]["command"].startswith("$env:CUDA_VISIBLE_DEVICES='0'; & ")
    assert "--adapter \"E:\\anima_gemma_swap\\kv_proj_hiddenstage_planner_v2.pt\"" in case["teacher"]["command"]
    assert "--adapter \"runs\\cache\\poc1_10k\\bridge\\poc1_10k_bridge.pt\"" in case["student"]["command"]
    assert "write-compare-report" in case["comparison"]["command"]


def test_text_rendering_eval_run_plan_rejects_non_4070_gpu() -> None:
    with pytest.raises(ValueError, match="4070"):
        build_text_rendering_eval_run_plan(gpu_index=1)


def test_text_rendering_eval_run_plan_escapes_powershell_expansion() -> None:
    pack = build_text_rendering_eval_pack()
    pack["prompts"] = [
        {
            **pack["prompts"][0],
            "prompt": 'Render $HOME and $(Get-Process) with a "quoted" label.',
        }
    ]

    plan = build_text_rendering_eval_run_plan(pack)
    command = plan["cases"][0]["teacher"]["command"]

    assert "`$HOME" in command
    assert "`$(" in command
    assert '`"quoted`"' in command


def test_cli_text_rendering_eval_run_plan_outputs_json(capsys) -> None:
    code = main(["text-rendering-eval-run-plan", "--max-cases", "1", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_rendering_preservation"
    assert payload["mode"] == "executable"
    assert payload["total_cases"] == 1


def test_qwen_baseline_prompt_records_match_eval_generate_format() -> None:
    records = build_text_rendering_qwen_prompt_records()

    assert records[0] == {
        "text": build_text_rendering_eval_pack()["prompts"][0]["prompt"],
        "src": "text_eval_001_sign_luna_gate",
        "id": "text_eval_001_sign_luna_gate",
        "idx": 0,
        "eval_idx": 0,
    }
    assert [record["eval_idx"] for record in records] == list(range(6))


def test_qwen_baseline_plan_uses_real_qwen_teacher_and_4070_only_gemma_student() -> None:
    plan = build_text_rendering_qwen_baseline_plan(max_cases=1)

    assert plan["stage"] == "text_rendering_qwen_baseline"
    assert plan["teacher_mode"] == "qwen"
    assert plan["student_mode"] == "gemma"
    assert plan["seed_base"] == 440001
    assert plan["prompt_file"] == "reports/text_rendering_qwen_baseline/prompts.jsonl"
    assert plan["qwen_command"].startswith("$env:CUDA_VISIBLE_DEVICES='0'; & ")
    assert "--mode qwen" in plan["qwen_command"]
    assert "--mode gemma" in plan["gemma_command"]
    assert "--adapter \"runs\\cache\\poc1_10k\\bridge\\poc1_10k_bridge.pt\"" in plan["gemma_command"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)
    assert plan["cases"][0]["qwen_image"] == "runs/images/text_rendering_qwen_baseline/qwen/text_eval_001_sign_luna_gate.png"
    assert plan["cases"][0]["gemma_image"] == "runs/images/text_rendering_qwen_baseline/gemma_poc1_10k/text_eval_001_sign_luna_gate.png"
    assert "write-compare-report" in plan["cases"][0]["compare_command"]


def test_qwen_baseline_plan_accepts_distinct_student_name_and_checkpoint() -> None:
    plan = build_text_rendering_qwen_baseline_plan(
        max_cases=1,
        student_checkpoint=r"runs\cache\text_preservation_qwen\bridge\text_preservation_bridge.pt",
        student_name="gemma_text_preservation",
    )

    assert "--name gemma_text_preservation" in plan["gemma_command"]
    assert "--adapter \"runs\\cache\\text_preservation_qwen\\bridge\\text_preservation_bridge.pt\"" in plan["gemma_command"]
    assert plan["cases"][0]["gemma_image"] == (
        "runs/images/text_rendering_qwen_baseline/gemma_text_preservation/text_eval_001_sign_luna_gate.png"
    )
    assert plan["cases"][0]["compare_report"] == (
        "reports/text_rendering_qwen_baseline/text_eval_001_sign_luna_gate_qwen_vs_gemma_text_preservation_compare.json"
    )


def test_cli_text_rendering_qwen_baseline_plan_outputs_json(capsys) -> None:
    code = main(["text-rendering-qwen-baseline-plan", "--max-cases", "1", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "text_rendering_qwen_baseline"
    assert payload["total_cases"] == 1
