import json
from pathlib import Path

from gemmanima.cli import main
from gemmanima.training.poc1_status import build_poc1_runtime_status, build_poc1_status


def test_build_poc1_status_reads_runtime_and_compare_reports(tmp_path) -> None:
    runtime = tmp_path / "runtime.json"
    compare = tmp_path / "compare.json"
    runtime.write_text(
        json.dumps(
            {
                "bridge_training": {"final_val_mse": 0.0031, "passed_mse_gate": True},
                "gemma_cache": {"pairing_ready": True},
                "real_render_smoke": {"output": "runs/images/student.png"},
            }
        ),
        encoding="utf-8",
    )
    compare.write_text(
        json.dumps(
            {
                "conditioning": {"mse": 0.0031},
                "image_metrics": {"mse": 0.11, "mae": 0.25},
            }
        ),
        encoding="utf-8",
    )

    status = build_poc1_status(runtime_report=runtime, compare_report=compare)

    assert status["ready"] is True
    assert status["bridge"]["passed_mse_gate"] is True
    assert status["comparison"]["image_mse"] == 0.11


def test_cli_poc1_status_outputs_json(tmp_path, capsys) -> None:
    runtime = tmp_path / "runtime.json"
    compare = tmp_path / "compare.json"
    runtime.write_text(
        json.dumps({"bridge_training": {"passed_mse_gate": True}, "gemma_cache": {"pairing_ready": True}}),
        encoding="utf-8",
    )
    compare.write_text(json.dumps({"image_metrics": {"mse": 0.11}}), encoding="utf-8")

    code = main(
        [
            "poc1-status",
            "--runtime-report",
            str(runtime),
            "--compare-report",
            str(compare),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True


def test_build_poc1_runtime_status_summarizes_10k_runtime_dirs(tmp_path: Path) -> None:
    targets = tmp_path / "targets"
    gemma = tmp_path / "gemma"
    bridge = tmp_path / "bridge" / "poc1_10k_bridge.pt"
    targets.mkdir()
    gemma.mkdir()
    bridge.parent.mkdir()
    (targets / "shard_0000.pt").write_bytes(b"target")
    (targets / "shard_0001.pt").write_bytes(b"target")
    (gemma / "shard_0000.pt").write_bytes(b"gemma")
    bridge.write_bytes(b"not a real checkpoint")

    status = build_poc1_runtime_status(target_dir=targets, gemma_dir=gemma, bridge_checkpoint=bridge)

    assert status["stage"] == "poc1_10k_pilot"
    assert status["executes_gpu_commands"] is False
    assert status["cache"]["target_shards"] == 2
    assert status["cache"]["gemma_shards"] == 1
    assert status["cache"]["paired_shards"] == 1
    assert status["cache"]["missing_gemma_shards"] == 1
    assert status["ready_for_bridge_training"] is False
    assert status["bridge"]["exists"] is True
    assert status["bridge"]["path"] == str(bridge)


def test_cli_poc1_runtime_status_outputs_json(tmp_path: Path, capsys) -> None:
    targets = tmp_path / "targets"
    gemma = tmp_path / "gemma"
    bridge = tmp_path / "bridge" / "poc1_10k_bridge.pt"
    targets.mkdir()
    gemma.mkdir()
    (targets / "shard_0000.pt").write_bytes(b"target")
    (gemma / "shard_0000.pt").write_bytes(b"gemma")

    code = main(
        [
            "poc1-runtime-status",
            "--target-dir",
            str(targets),
            "--gemma-dir",
            str(gemma),
            "--bridge-checkpoint",
            str(bridge),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_for_bridge_training"] is True
    assert payload["cache"]["missing_gemma_shards"] == 0
    assert payload["bridge"]["exists"] is False
