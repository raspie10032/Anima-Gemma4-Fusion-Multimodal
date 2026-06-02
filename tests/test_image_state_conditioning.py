import json
from pathlib import Path

from gemmanima.cli import main
from gemmanima.training.image_state_conditioning import (
    build_image_state_conditioning_plan,
    build_image_state_subset,
)


def _write_source_manifest(path: Path, image_embed: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "hiddenstage_multimodal_planner_anima_v2",
                "id": 7000000,
                "visible_prompt": "masterpiece, best quality, 1girl, green hair",
                "teacher_prompt": "masterpiece, best quality, 1girl, green hair",
                "image": "D:/data/images/7000000.jpg",
                "image_embed_pre": str(image_embed),
                "rating": "g",
                "target_tags": "1girl, green hair",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_image_state_subset_writes_text_and_image_embed_records(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    subset = tmp_path / "subset.jsonl"
    image_embed = tmp_path / "7000000.pt"
    image_embed.write_bytes(b"fake")
    _write_source_manifest(source, image_embed)

    stats = build_image_state_subset(source_manifest=source, output=subset, limit=10)

    rows = [json.loads(line) for line in subset.read_text(encoding="utf-8").splitlines()]
    assert stats["ready"] is True
    assert stats["written"] == 1
    assert rows[0]["idx"] == 0
    assert rows[0]["source_id"] == 7000000
    assert rows[0]["text"] == "masterpiece, best quality, 1girl, green hair"
    assert rows[0]["image_embed_pre"] == str(image_embed)
    assert rows[0]["input_modalities"] == ["image", "text"]


def test_build_image_state_subset_skips_missing_image_embed_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    subset = tmp_path / "subset.jsonl"
    _write_source_manifest(source, tmp_path / "missing.pt")

    stats = build_image_state_subset(source_manifest=source, output=subset, limit=10)

    assert stats["written"] == 0
    assert stats["skipped_missing_image_embed"] == 1


def test_image_state_conditioning_plan_uses_text_translator_anchor_and_4070_only(tmp_path: Path) -> None:
    translator = tmp_path / "kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
    translator.write_bytes(b"checkpoint")
    plan = build_image_state_conditioning_plan(
        source_manifest=tmp_path / "source.jsonl",
        output_root=tmp_path / "image_state",
        text_translator=translator,
        sample_count=10000,
    )

    assert plan["architecture"]["text_path_status"] == "treated_as_functional_anchor"
    assert plan["architecture"]["text_translator_anchor_exists"] is True
    assert plan["gpu_policy"]["cuda_visible_devices"] == "0"
    assert plan["gpu_policy"]["forbidden_gpu"] == "RTX 5060"
    assert "CUDA_VISIBLE_DEVICES='0'" in plan["commands"]["cache_targets"]
    assert "CUDA_VISIBLE_DEVICES='0'" in plan["commands"]["train_image_translator"]
    assert "train_image_state_translator.py" in plan["commands"]["train_image_translator"]
    assert "CUDA_VISIBLE_DEVICES='1'" not in json.dumps(plan, ensure_ascii=False)


def test_cli_image_state_conditioning_plan_writes_output(tmp_path: Path, capsys) -> None:
    output = tmp_path / "plan.json"

    code = main(
        [
            "image-state-conditioning-plan",
            "--source-manifest",
            str(tmp_path / "source.jsonl"),
            "--output-root",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan_path"] == str(output)
    assert output.exists()


def test_cli_image_state_conditioning_subset_outputs_json(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    subset = tmp_path / "subset.jsonl"
    image_embed = tmp_path / "7000000.pt"
    image_embed.write_bytes(b"fake")
    _write_source_manifest(source, image_embed)

    code = main(
        [
            "image-state-conditioning-subset",
            "--source-manifest",
            str(source),
            "--output",
            str(subset),
            "--limit",
            "1",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] == 1
    assert subset.exists()


def test_cli_image_state_engine_status_reports_fusion_modes(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "translator.pt"
    subset = tmp_path / "subset.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    subset.write_text("{}\n", encoding="utf-8")

    code = main(
        [
            "image-state-engine-status",
            "--checkpoint",
            str(checkpoint),
            "--subset",
            str(subset),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["supported_modes"] == ["image_only", "hidden_fusion", "conditioning_fusion"]


def test_cli_image_state_fusion_guard_writes_replay_manifest(tmp_path: Path, capsys) -> None:
    from PIL import Image
    import torch

    embed = tmp_path / "embed.pt"
    render = tmp_path / "render.png"
    subset = tmp_path / "subset.jsonl"
    sweep = tmp_path / "sweep.json"
    output = tmp_path / "guard.json"
    replay = tmp_path / "replay.jsonl"
    torch.save(torch.zeros(2, 2), embed)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(render)
    subset.write_text(
        json.dumps({"idx": 0, "text": "safe test", "image_embed_pre": str(embed), "rating": "g"}) + "\n",
        encoding="utf-8",
    )
    sweep.write_text(
        json.dumps({"outputs": [{"idx": 0, "mode": "conditioning_fusion", "output": str(render), "seed": 1}]}),
        encoding="utf-8",
    )

    code = main(
        [
            "image-state-fusion-guard",
            "--sweep-report",
            str(sweep),
            "--subset",
            str(subset),
            "--failed-idx",
            "0",
            "--output",
            str(output),
            "--replay-output",
            str(replay),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed_count"] == 1
    assert output.exists()
    assert replay.exists()


def test_cli_image_state_replay_objective_writes_manifest(tmp_path: Path, capsys) -> None:
    subset = tmp_path / "subset.jsonl"
    replay = tmp_path / "replay.jsonl"
    output = tmp_path / "objective.json"
    subset.write_text('{"idx": 0, "text": "safe", "image_embed_pre": "x.pt"}\n', encoding="utf-8")
    replay.write_text('{"idx": 0, "guard_reason": "conditioning_fusion_noise_or_instability"}\n', encoding="utf-8")

    code = main(
        [
            "image-state-replay-objective",
            "--base-subset",
            str(subset),
            "--guard-replay",
            str(replay),
            "--output",
            str(output),
            "--stage",
            "image_state_conditioning_v3_guarded",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["objective_manifest"] == str(output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["promotion_gates"]["conditioning_fusion_guard"]["required"] is True


def test_cli_image_state_fusion_preflight_writes_filtered_image_replay(tmp_path: Path, capsys) -> None:
    subset = tmp_path / "subset.jsonl"
    fusion = tmp_path / "fusion.json"
    text_only = tmp_path / "text_only.json"
    output = tmp_path / "preflight.json"
    replay = tmp_path / "image_replay.jsonl"
    subset.write_text(
        '{"idx": 0, "text": "text fail", "image_embed_pre": "0.pt"}\n'
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

    code = main(
        [
            "image-state-fusion-preflight",
            "--fusion-report",
            str(fusion),
            "--text-only-report",
            str(text_only),
            "--subset",
            str(subset),
            "--fusion-failed-idx",
            "0",
            "1",
            "--text-only-failed-idx",
            "0",
            "--output",
            str(output),
            "--image-replay-output",
            str(replay),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["image_conditioning_failed_count"] == 1
    assert output.exists()
    replay_rows = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    assert [row["idx"] for row in replay_rows] == [1]
