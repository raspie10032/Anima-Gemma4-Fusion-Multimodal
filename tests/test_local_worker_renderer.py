import json
import subprocess
import sys
from pathlib import Path

from gemmanima.core.schemas import ConditioningBundle, GenerationPlan
from gemmanima.modules.local_worker_anima_renderer import LocalWorkerAnimaRendererAdapter


def test_conditioning_bundle_round_trips_from_json_dict() -> None:
    original = ConditioningBundle(
        source="trained_hiddenstage_bridge",
        metadata={"hiddenstage_source_text": "forest"},
        semantic_conditioning={"subject": "spirit"},
        lora_hints=("anima",),
        renderer_profile="anima_fp16_final",
    )

    restored = ConditioningBundle.from_dict(original.to_json_dict())

    assert restored == original


def test_local_worker_renderer_invokes_module_with_plan_and_conditioning(tmp_path: Path) -> None:
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        payload = json.loads(kwargs["input"])
        output = Path(payload["output_root"]) / "worker.png"
        output.write_bytes(b"png")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"image_id": "worker", "output_path": str(output), "seed": 123, "warnings": []}),
            stderr="",
        )

    renderer = LocalWorkerAnimaRendererAdapter(output_root=tmp_path, runner=fake_runner)
    result = renderer.generate(
        GenerationPlan(prompt="bright forest", width=256, height=384, steps=1, cfg=1.0, seed=123),
        ConditioningBundle(source="trained_hiddenstage_bridge", metadata={"hiddenstage_source_text": "bright forest"}),
    )

    command, kwargs = calls[0]
    assert command[0].endswith("python.exe")
    assert command[1] == "-c"
    assert "gemmanima.rendering.worker_render" in command[2]
    assert "hard_exit_on_success=True" in command[2]
    payload = json.loads(kwargs["input"])
    assert payload["plan"]["width"] == 256
    assert payload["plan"]["height"] == 384
    assert payload["image_id"]
    assert payload["output_path"].endswith(f"{payload['image_id']}.png")
    assert payload["conditioning"]["metadata"]["hiddenstage_source_text"] == "bright forest"
    assert payload["tiled_vae"] is True
    assert payload["comfy_args"] == []
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["check"] is False
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert result.output_path == tmp_path / "worker.png"
    assert result.seed == 123


def test_local_worker_renderer_forwards_memory_options(tmp_path: Path) -> None:
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["input"])
        output = Path(payload["output_path"])
        output.write_bytes(b"png")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GEMMANIMA_WORKER_RESULT "
            + json.dumps({"image_id": payload["image_id"], "output_path": str(output), "seed": 123, "warnings": []}),
            stderr="",
        )

    renderer = LocalWorkerAnimaRendererAdapter(
        output_root=tmp_path,
        runner=fake_runner,
        tiled_vae=True,
        comfy_args=("--cpu-vae", "--reserve-vram", "1.5"),
    )
    renderer.generate(
        GenerationPlan(prompt="bright forest", width=256, height=256, steps=1, cfg=1.0, seed=123),
        ConditioningBundle(source="trained_hiddenstage_bridge"),
    )

    payload = json.loads(calls[0]["input"])
    assert payload["tiled_vae"] is True
    assert payload["comfy_args"] == ["--cpu-vae", "--reserve-vram", "1.5"]


def test_local_worker_renderer_reports_subprocess_crash(tmp_path: Path) -> None:
    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 3221225786, stdout="", stderr="native crash")

    renderer = LocalWorkerAnimaRendererAdapter(output_root=tmp_path, runner=fake_runner)

    try:
        renderer.generate(
            GenerationPlan(prompt="bright forest", width=256, height=256, steps=1, cfg=1.0),
            ConditioningBundle(source="trained_hiddenstage_bridge"),
        )
    except RuntimeError as exc:
        assert "local Anima worker failed" in str(exc)
        assert "3221225786" in str(exc)
        assert "native crash" in str(exc)
    else:
        raise AssertionError("worker crash should be surfaced as a recoverable RuntimeError")


def test_local_worker_renderer_recovers_image_written_before_native_crash(tmp_path: Path) -> None:
    def fake_runner(command, **kwargs):
        payload = json.loads(kwargs["input"])
        Path(payload["output_path"]).write_bytes(b"png")
        return subprocess.CompletedProcess(command, 3221225477, stdout="progress", stderr="")

    renderer = LocalWorkerAnimaRendererAdapter(output_root=tmp_path, runner=fake_runner)
    result = renderer.generate(
        GenerationPlan(prompt="bright forest", width=256, height=256, steps=1, cfg=1.0, seed=99),
        ConditioningBundle(source="trained_hiddenstage_bridge"),
    )

    assert result.output_path.exists()
    assert result.seed == 99
    assert any("worker recovered image after native exit code 3221225477" in warning for warning in result.warnings)


def test_local_worker_renderer_sends_unicode_payload_as_utf8_text(tmp_path: Path) -> None:
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["input"])
        output = Path(payload["output_path"])
        output.write_bytes(b"png")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GEMMANIMA_WORKER_RESULT "
            + json.dumps({"image_id": payload["image_id"], "output_path": str(output), "seed": 321, "warnings": []}),
            stderr="",
        )

    renderer = LocalWorkerAnimaRendererAdapter(output_root=tmp_path, runner=fake_runner)
    renderer.generate(
        GenerationPlan(prompt="작은 숲속 요정", width=256, height=256, steps=1, cfg=1.0, seed=321),
        ConditioningBundle(source="trained_hiddenstage_bridge", metadata={"hiddenstage_source_text": "작은 숲속 요정"}),
    )

    payload = json.loads(calls[0]["input"])
    assert calls[0]["encoding"] == "utf-8"
    assert payload["plan"]["prompt"] == "작은 숲속 요정"
    assert payload["conditioning"]["metadata"]["hiddenstage_source_text"] == "작은 숲속 요정"
