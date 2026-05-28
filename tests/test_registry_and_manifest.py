from pathlib import Path

from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.renderer_profiles import RendererProfileManager


def test_model_registry_reports_expected_assets() -> None:
    health = ModelRegistry().health()

    assert "gemma_planner_adapter" in health
    assert "anima_diffusion_model" in health
    assert health["gemma_planner_adapter"]["role"] == "planner"


def test_manifest_store_can_read_latest(tmp_path: Path) -> None:
    conductor = GemmAnimaConductor(
        session_id="manifest-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
    )
    response = conductor.handle_user_message("draw a silver tower")

    store = ManifestStore(tmp_path / "manifests")
    latest = store.latest()
    assert latest == response.manifest_path
    data = store.read_json(latest)
    assert data["job_id"] == response.job_id
    assert data["models"]["registry_health"]["gemma_planner_adapter"]["role"] == "planner"


def test_renderer_profile_clamp_keeps_safe_bounds() -> None:
    manager = RendererProfileManager()

    clamped = manager.clamp(
        profile_name="anima_int8_draft",
        width=2048,
        height=1025,
        steps=100,
        cfg=100.0,
    )

    assert clamped["width"] == 768
    assert clamped["height"] == 768
    assert clamped["steps"] == 16
    assert clamped["cfg"] == 3.5
