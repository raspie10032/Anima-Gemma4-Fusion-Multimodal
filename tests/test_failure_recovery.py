from pathlib import Path

from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, JobStatus


class FailingRenderer:
    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle):
        raise RuntimeError("simulated renderer failure")


def test_renderer_failure_writes_failed_manifest(tmp_path: Path) -> None:
    conductor = GemmAnimaConductor(
        session_id="failure-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
        renderer=FailingRenderer(),
    )

    response = conductor.handle_user_message("draw a bright forest")

    assert response.status == JobStatus.FAILED
    assert response.manifest_path and response.manifest_path.exists()
    assert "error:recoverable" in response.progress
    data = ManifestStore(tmp_path / "manifests").read_json(response.manifest_path)
    assert data["status"] == "failed"
    assert "simulated renderer failure" in data["error"]
