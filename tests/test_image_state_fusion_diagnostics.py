import json
from pathlib import Path

import torch
from PIL import Image

from gemmanima.training.image_state_fusion_diagnostics import (
    build_conditioning_fusion_preflight_manifest,
    build_image_state_replay_training_objective,
    build_conditioning_fusion_guard_manifest,
    write_conditioning_fusion_preflight_manifest,
    write_image_state_replay_training_objective,
    write_conditioning_fusion_guard_manifest,
)


def _write_subset(path: Path, embed_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "idx": 0,
                "source_id": 7000,
                "text": "masterpiece, best quality, safe, test",
                "visible_prompt": "masterpiece, best quality, safe, test",
                "image": "D:/images/7000.jpg",
                "image_embed_pre": str(embed_path),
                "rating": "g",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, output_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "image_text_fusion_poc",
                "subset": "subset.jsonl",
                "outputs": [
                    {
                        "idx": 0,
                        "mode": "conditioning_fusion",
                        "output": str(output_path),
                        "seed": 950001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_conditioning_fusion_guard_manifest_records_failed_tensor_and_image_stats(tmp_path: Path) -> None:
    embed = tmp_path / "embed.pt"
    subset = tmp_path / "subset.jsonl"
    image = tmp_path / "out.png"
    report = tmp_path / "report.json"
    torch.save(torch.tensor([[1.0, -2.0], [0.5, 0.25]]), embed)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image)
    _write_subset(subset, embed)
    _write_report(report, image)

    manifest = build_conditioning_fusion_guard_manifest(
        sweep_report=report,
        subset=subset,
        failed_indices=[0],
    )

    assert manifest["failed_count"] == 1
    assert manifest["passed_count"] == 0
    case = manifest["failed_cases"][0]
    assert case["idx"] == 0
    assert case["image_state"]["shape"] == [2, 2]
    assert case["image_state"]["absmax"] == 2.0
    assert case["output_image"]["size"] == [4, 4]
    assert manifest["replay_rows"][0]["image_embed_pre"] == str(embed)


def test_write_conditioning_fusion_guard_manifest_writes_json_and_replay_jsonl(tmp_path: Path) -> None:
    embed = tmp_path / "embed.pt"
    subset = tmp_path / "subset.jsonl"
    image = tmp_path / "out.png"
    report = tmp_path / "report.json"
    output = tmp_path / "guard.json"
    replay = tmp_path / "replay.jsonl"
    torch.save(torch.zeros(2, 2), embed)
    Image.new("RGB", (4, 4), (100, 120, 140)).save(image)
    _write_subset(subset, embed)
    _write_report(report, image)

    payload = write_conditioning_fusion_guard_manifest(
        output=output,
        replay_output=replay,
        sweep_report=report,
        subset=subset,
        failed_indices=[0],
    )

    assert output.exists()
    assert replay.exists()
    assert payload["guard_manifest"] == str(output)
    replay_rows = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    assert replay_rows[0]["idx"] == 0
    assert replay_rows[0]["guard_reason"] == "conditioning_fusion_noise_or_instability"


def test_build_image_state_replay_training_objective_preserves_guard_gate(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    base_subset = tmp_path / "subset.jsonl"
    checkpoint = tmp_path / "image_translator.pt"
    base_subset.write_text('{"idx": 0, "image_embed_pre": "x.pt", "text": "safe"}\n', encoding="utf-8")
    replay.write_text('{"idx": 0, "guard_reason": "conditioning_fusion_noise_or_instability"}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    objective = build_image_state_replay_training_objective(
        base_subset=base_subset,
        guard_replay=replay,
        current_checkpoint=checkpoint,
        stage="image_state_conditioning_v3_guarded",
        replay_weight=12,
    )

    assert objective["stage"] == "image_state_conditioning_v3_guarded"
    assert objective["training_policy"]["replay_weight"] == 12
    assert objective["training_policy"]["guard_replay_count"] == 1
    assert objective["promotion_gates"]["conditioning_fusion_guard"]["required"] is True
    assert objective["promotion_gates"]["conditioning_fusion_guard"]["fail_on_noise_collapse"] is True
    assert "CUDA_VISIBLE_DEVICES='0'" in objective["commands"]["train_candidate"]
    assert "--init-checkpoint" in objective["commands"]["train_candidate"]
    assert "--guard-replay-weight 12" in objective["commands"]["train_candidate"]
    assert "RTX 5060" in objective["gpu_policy"]["forbidden_gpu"]


def test_write_image_state_replay_training_objective_writes_manifest(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    base_subset = tmp_path / "subset.jsonl"
    output = tmp_path / "objective.json"
    base_subset.write_text('{"idx": 0, "image_embed_pre": "x.pt", "text": "safe"}\n', encoding="utf-8")
    replay.write_text('{"idx": 0, "guard_reason": "conditioning_fusion_noise_or_instability"}\n', encoding="utf-8")

    payload = write_image_state_replay_training_objective(
        output=output,
        base_subset=base_subset,
        guard_replay=replay,
        stage="image_state_conditioning_v3_guarded",
    )

    assert output.exists()
    assert payload["objective_manifest"] == str(output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["training_policy"]["guard_replay_count"] == 1


def test_build_conditioning_fusion_preflight_manifest_splits_text_and_image_failures(tmp_path: Path) -> None:
    subset = tmp_path / "subset.jsonl"
    fusion = tmp_path / "fusion.json"
    text_only = tmp_path / "text_only.json"
    subset.write_text(
        "\n".join(
            [
                json.dumps({"idx": 0, "text": "bad text path", "image_embed_pre": "0.pt"}),
                json.dumps({"idx": 1, "text": "image-specific failure", "image_embed_pre": "1.pt"}),
                json.dumps({"idx": 2, "text": "passed", "image_embed_pre": "2.pt"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fusion.write_text(
        json.dumps(
            {
                "outputs": [
                    {"idx": 0, "seed": 950001, "mode": "conditioning_fusion", "output": "fusion0.png"},
                    {"idx": 1, "seed": 950001, "mode": "conditioning_fusion", "output": "fusion1.png"},
                    {"idx": 2, "seed": 950001, "mode": "conditioning_fusion", "output": "fusion2.png"},
                ]
            }
        ),
        encoding="utf-8",
    )
    text_only.write_text(
        json.dumps(
            {
                "outputs": [
                    {"idx": 0, "seed": 950001, "mode": "conditioning_fusion", "output": "text0.png"},
                    {"idx": 1, "seed": 950001, "mode": "conditioning_fusion", "output": "text1.png"},
                    {"idx": 2, "seed": 950001, "mode": "conditioning_fusion", "output": "text2.png"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_conditioning_fusion_preflight_manifest(
        fusion_report=fusion,
        text_only_report=text_only,
        subset=subset,
        fusion_failed_indices=[0, 1],
        text_only_failed_indices=[0],
    )

    assert manifest["text_conditioning_failed_count"] == 1
    assert manifest["image_conditioning_failed_count"] == 1
    assert manifest["passed_count"] == 1
    assert manifest["text_conditioning_failed"][0]["idx"] == 0
    assert manifest["image_conditioning_failed"][0]["idx"] == 1
    assert manifest["image_replay_rows"][0]["idx"] == 1
    assert manifest["image_replay_rows"][0]["guard_reason"] == "image_conditioning_failed_text_only_passed"


def test_write_conditioning_fusion_preflight_manifest_writes_only_image_replay_rows(tmp_path: Path) -> None:
    subset = tmp_path / "subset.jsonl"
    fusion = tmp_path / "fusion.json"
    text_only = tmp_path / "text_only.json"
    output = tmp_path / "preflight.json"
    replay = tmp_path / "image_replay.jsonl"
    subset.write_text(
        '{"idx": 0, "text": "bad text", "image_embed_pre": "0.pt"}\n'
        '{"idx": 1, "text": "image fail", "image_embed_pre": "1.pt"}\n',
        encoding="utf-8",
    )
    fusion.write_text(
        json.dumps({"outputs": [{"idx": 0, "seed": 1, "output": "f0.png"}, {"idx": 1, "seed": 1, "output": "f1.png"}]}),
        encoding="utf-8",
    )
    text_only.write_text(
        json.dumps({"outputs": [{"idx": 0, "seed": 1, "output": "t0.png"}, {"idx": 1, "seed": 1, "output": "t1.png"}]}),
        encoding="utf-8",
    )

    payload = write_conditioning_fusion_preflight_manifest(
        output=output,
        image_replay_output=replay,
        fusion_report=fusion,
        text_only_report=text_only,
        subset=subset,
        fusion_failed_indices=[0, 1],
        text_only_failed_indices=[0],
    )

    assert payload["preflight_manifest"] == str(output)
    assert payload["image_replay_manifest"] == str(replay)
    replay_rows = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    assert [row["idx"] for row in replay_rows] == [1]
