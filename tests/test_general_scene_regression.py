import json

from gemmanima.cli import main
from gemmanima.training.text_preservation import (
    build_general_scene_regression_eval_plan,
    build_general_scene_regression_prompt_records,
)


def test_general_scene_regression_prompt_records_are_not_text_specific() -> None:
    records = build_general_scene_regression_prompt_records(count=6)

    assert len(records) == 6
    assert records[0]["src"] == "general_scene_regression_000"
    assert len({record["text"] for record in records}) == 6
    assert all("clearly reads" not in record["text"] for record in records)
    assert all("target_text" not in record for record in records)
    assert {record["category"] for record in records} >= {"cafe_interior", "forest_character"}


def test_general_scene_regression_prompt_records_scale_to_50_unique_cases() -> None:
    records = build_general_scene_regression_prompt_records(count=50)

    assert len(records) == 50
    assert len({record["text"] for record in records}) == 50
    assert records[-1]["src"] == "general_scene_regression_049"
    assert all("clearly reads" not in record["text"] for record in records)


def test_general_scene_regression_eval_plan_is_4070_only_and_customizable() -> None:
    plan = build_general_scene_regression_eval_plan(
        count=2,
        prompt_file=r"reports\general_scene_regression_v5\prompts.jsonl",
        out_root=r"runs\images\general_scene_regression_v5",
        report_root=r"reports\general_scene_regression_v5",
        student_checkpoint=r"runs\cache\text_preservation_blended_v5\bridge\text_preservation_blended_v5_bridge.pt",
        student_name="gemma_text_preservation_blended_v5",
    )

    assert plan["stage"] == "general_scene_regression_eval"
    assert plan["prompt_file"] == "reports/general_scene_regression_v5/prompts.jsonl"
    assert plan["student_name"] == "gemma_text_preservation_blended_v5"
    assert plan["total_cases"] == 2
    assert "CUDA_VISIBLE_DEVICES='0'" in json.dumps(plan, ensure_ascii=False)
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)
    assert "--mode qwen" in plan["qwen_command"]
    assert "--mode gemma" in plan["gemma_command"]
    assert plan["cases"][0]["compare_report"] == (
        "reports/general_scene_regression_v5/general_scene_regression_000_qwen_vs_gemma_text_preservation_blended_v5_compare.json"
    )


def test_cli_general_scene_regression_eval_plan_outputs_json(capsys) -> None:
    code = main(["text-preservation-general-scene-eval-plan", "--count", "2", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "general_scene_regression_eval"
    assert payload["total_cases"] == 2
