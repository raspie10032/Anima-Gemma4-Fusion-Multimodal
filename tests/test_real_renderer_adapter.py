from pathlib import Path

from gemmanima.core.config import EngineConfig, ModelConfig
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan
from gemmanima.modules.real_anima_renderer import ExternalAnimaRendererAdapter


def test_external_anima_renderer_invokes_legacy_script_with_bridge(tmp_path: Path, monkeypatch) -> None:
    render_python = tmp_path / "python.exe"
    render_script = tmp_path / "18_hiddenstage_chat_generate.py"
    bridge = tmp_path / "kv_proj_hiddenstage_planner_v2.pt"
    diffusion = tmp_path / "anima-base-v1.0.safetensors"
    vae = tmp_path / "qwen_image_vae.safetensors"
    for path in (render_python, render_script, bridge, diffusion, vae):
        path.write_bytes(b"x")
    monkeypatch.setattr("gemmanima.training.real_render.DEFAULT_EMBEDDED_PYTHON", render_python)
    monkeypatch.setattr("gemmanima.training.real_render.DEFAULT_CHAT_RENDER_SCRIPT", render_script)
    config = EngineConfig(
        models=ModelConfig(
            hiddenstage_bridge=bridge,
            anima_diffusion_model=diffusion,
            anima_vae=vae,
        )
    )
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--out") + 1])
        output.write_bytes(b"png")
        return type("Completed", (), {"returncode": 0, "stdout": "IMAGE: render.png", "stderr": ""})()

    renderer = ExternalAnimaRendererAdapter(output_root=tmp_path, config=config, runner=fake_runner)
    result = renderer.generate(
        GenerationPlan(prompt="bright forest", width=512, height=512, steps=12, cfg=4.5, seed=123),
        ConditioningBundle(source="trained_hiddenstage_bridge"),
    )

    command, kwargs = calls[0]
    command_text = " ".join(str(item) for item in command)
    assert "18_hiddenstage_chat_generate.py" in command_text
    assert "kv_proj_hiddenstage_planner_v2.pt" in command_text
    assert "--request" in command
    assert "bright forest" in command
    assert "--seed" in command
    assert "123" in command
    assert kwargs["check"] is False
    assert result.output_path.exists()
    assert result.seed == 123
