import json
from pathlib import Path

from gemmanima import cli
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult


class CapturingRenderer:
    dry_run = False
    plans = []
    init_kwargs = []

    def __init__(self, output_root, **kwargs):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        type(self).init_kwargs.append(kwargs)

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        type(self).plans.append(plan)
        output = self.output_root / "override.png"
        output.write_bytes(b"png")
        return RenderResult(image_id="override", output_path=output, seed=plan.seed or 1)


def test_run_applies_generation_overrides(monkeypatch, tmp_path: Path, capsys) -> None:
    CapturingRenderer.plans = []
    CapturingRenderer.init_kwargs = []
    monkeypatch.setattr(cli, "InProcessAnimaRendererAdapter", CapturingRenderer)

    assert (
        cli.main(
            [
                "run",
                "draw a bright forest",
                "--renderer",
                "real",
                "--steps",
                "8",
                "--size",
                "512",
                "--cfg",
                "4.5",
                "--seed",
                "123",
                "--manifest-root",
                str(tmp_path / "manifests"),
                "--image-root",
                str(tmp_path / "images"),
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    plan = CapturingRenderer.plans[0]
    assert plan.steps == 8
    assert plan.width == 512
    assert plan.height == 512
    assert plan.cfg == 4.5
    assert plan.seed == 123

    latest = ManifestStore(tmp_path / "manifests").latest()
    assert latest is not None


def test_run_applies_anima_model_and_dtype_overrides(monkeypatch, tmp_path: Path, capsys) -> None:
    CapturingRenderer.plans = []
    CapturingRenderer.init_kwargs = []
    monkeypatch.setattr(cli, "InProcessAnimaRendererAdapter", CapturingRenderer)
    anima_dm = tmp_path / "anima-base-v1.0-fp8_e4m3fn.safetensors"

    assert (
        cli.main(
            [
                "run",
                "draw a bright forest",
                "--renderer",
                "in-process",
                "--anima-dm",
                str(anima_dm),
                "--unet-dtype",
                "default",
                "--manifest-root",
                str(tmp_path / "manifests"),
                "--image-root",
                str(tmp_path / "images"),
                "--json",
            ]
        )
        == 0
    )

    json.loads(capsys.readouterr().out)
    kwargs = CapturingRenderer.init_kwargs[0]
    assert kwargs["unet_dtype"] == "default"
    assert kwargs["config"].models.anima_diffusion_model == anima_dm


def test_run_applies_hiddenstage_bridge_override(monkeypatch, tmp_path: Path, capsys) -> None:
    CapturingRenderer.plans = []
    CapturingRenderer.init_kwargs = []
    monkeypatch.setattr(cli, "InProcessAnimaRendererAdapter", CapturingRenderer)
    bridge = tmp_path / "kv_proj_text_delta_300k_from_epoch1_a0p35.pt"

    assert (
        cli.main(
            [
                "run",
                "draw a bright forest",
                "--renderer",
                "in-process",
                "--hiddenstage-bridge",
                str(bridge),
                "--manifest-root",
                str(tmp_path / "manifests"),
                "--image-root",
                str(tmp_path / "images"),
                "--json",
            ]
        )
        == 0
    )

    json.loads(capsys.readouterr().out)
    kwargs = CapturingRenderer.init_kwargs[0]
    assert kwargs["config"].models.hiddenstage_bridge == bridge
