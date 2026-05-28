from pathlib import Path

from gemmanima import cli
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult


class FakeRealRenderer:
    created = False
    dry_run = False

    def __init__(self, output_root, **kwargs):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        type(self).created = True

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        output = self.output_root / "real.png"
        output.write_bytes(b"png")
        return RenderResult(image_id="real", output_path=output, seed=plan.seed or 1)


class FakeInProcessRenderer(FakeRealRenderer):
    pass


def test_run_real_routes_to_in_process_renderer(monkeypatch, tmp_path: Path) -> None:
    FakeInProcessRenderer.created = False
    monkeypatch.setattr(cli, "InProcessAnimaRendererAdapter", FakeInProcessRenderer)

    exit_code = cli.main(
        [
            "run",
            "draw a bright forest",
            "--renderer",
            "real",
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--image-root",
            str(tmp_path / "images"),
            "--json",
        ]
    )

    assert exit_code == 0
    assert FakeInProcessRenderer.created is True


def test_run_can_route_to_external_script_renderer(monkeypatch, tmp_path: Path) -> None:
    FakeRealRenderer.created = False
    monkeypatch.setattr(cli, "ExternalAnimaRendererAdapter", FakeRealRenderer)

    exit_code = cli.main(
        [
            "run",
            "draw a bright forest",
            "--renderer",
            "external-script",
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--image-root",
            str(tmp_path / "images"),
            "--json",
        ]
    )

    assert exit_code == 0
    assert FakeRealRenderer.created is True
    latest = ManifestStore(tmp_path / "manifests").latest()
    assert latest is not None
    data = ManifestStore(tmp_path / "manifests").read_json(latest)
    assert data["renderer"]["dry_run"] is False


def test_run_can_route_to_in_process_renderer(monkeypatch, tmp_path: Path) -> None:
    FakeInProcessRenderer.created = False
    monkeypatch.setattr(cli, "InProcessAnimaRendererAdapter", FakeInProcessRenderer)

    exit_code = cli.main(
        [
            "run",
            "draw a bright forest",
            "--renderer",
            "in-process",
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--image-root",
            str(tmp_path / "images"),
            "--json",
        ]
    )

    assert exit_code == 0
    assert FakeInProcessRenderer.created is True
