from pathlib import Path

from gemmanima.core.model_paths import model_path
from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.manifest import ManifestStore
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.renderer_profiles import RendererProfileManager


def test_model_registry_reports_expected_assets() -> None:
    health = ModelRegistry().health()

    assert "gemma_core.shared_base_gguf" in health
    assert "anima_image_core.diffusion_model" in health
    assert "anima_image_core.text_encoder" not in health
    assert "vision_tagger.wd_swinv2_model" in health
    assert "hiddenstage_bridge.bridge_checkpoint" in health
    assert health["gemma_core.shared_base_gguf"]["component"] == "gemma_core"
    assert health["anima_image_core.diffusion_model"]["component_label"] == "Anima Image Core"
    assert health["hiddenstage_bridge.bridge_checkpoint"]["role"] == "fusion_bridge"
    assert health["gemma_core.shared_base_gguf"]["source"]["repo_id"] == "mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF"
    assert health["gemma_core.shared_base_gguf"]["source"]["license_id"] == "apache-2.0"
    assert health["anima_image_core.diffusion_model"]["source"]["repo_id"] == "circlestone-labs/Anima"
    assert health["anima_image_core.diffusion_model"]["source"]["license_id"] == "circlestone-labs-non-commercial-license"
    assert health["vision_tagger.wd_swinv2_model"]["source"]["repo_id"] == "SmilingWolf/wd-swinv2-tagger-v3"
    assert health["vision_tagger.wd_swinv2_model"]["source"]["license_id"] == "apache-2.0"
    assert health["hiddenstage_bridge.bridge_checkpoint"]["source"]["origin"] == "gemmanima_adapter_bundle"
    assert health["hiddenstage_bridge.bridge_checkpoint"]["source"]["license_id"] == "other"


def test_model_registry_groups_assets_by_named_runtime_part() -> None:
    grouped = ModelRegistry().grouped_health()

    assert list(grouped) == ["gemma_core", "anima_image_core", "vision_tagger", "hiddenstage_bridge"]
    assert grouped["gemma_core"]["label"] == "Gemma Core"
    assert grouped["anima_image_core"]["label"] == "Anima Image Core"
    assert grouped["vision_tagger"]["label"] == "Vision Tagger"
    assert grouped["hiddenstage_bridge"]["label"] == "HiddenStage Bridge"
    assert "gemma_core.vision_mmproj" in grouped["gemma_core"]["assets"]
    assert "anima_image_core.vae" in grouped["anima_image_core"]["assets"]
    assert "vision_tagger.wd_swinv2_model" in grouped["vision_tagger"]["assets"]
    assert "hiddenstage_bridge.planner_adapter" in grouped["hiddenstage_bridge"]["assets"]


def test_model_registry_download_plan_separates_original_and_adapter_sources() -> None:
    plan = ModelRegistry().download_plan()

    original = [item for item in plan["assets"] if item["source"]["origin"] == "original_model_page"]
    adapter = [item for item in plan["assets"] if item["source"]["origin"] == "gemmanima_adapter_bundle"]
    assert {item["name"] for item in original} == {
        "gemma_core.shared_base_gguf",
        "vision_tagger.wd_swinv2_model",
        "vision_tagger.wd_swinv2_tags",
        "anima_image_core.diffusion_model",
        "anima_image_core.vae",
    }
    assert "hiddenstage_bridge.bridge_checkpoint" in {item["name"] for item in adapter}


def test_model_path_uses_model_root_when_legacy_file_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMMANIMA_MODEL_ROOT", str(tmp_path / "models"))
    legacy = tmp_path / "missing.gguf"

    resolved = model_path("gemma_core", "base.gguf", legacy)

    assert resolved == tmp_path / "models" / "gemma_core" / "base.gguf"


def test_model_path_ignores_existing_legacy_path_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMMANIMA_MODEL_ROOT", raising=False)
    monkeypatch.delenv("GEMMANIMA_ALLOW_LEGACY_MODEL_PATHS", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local_app_data"))
    legacy = tmp_path / "legacy.gguf"
    legacy.write_bytes(b"legacy")

    resolved = model_path("gemma_core", "base.gguf", legacy)

    assert resolved == tmp_path / "local_app_data" / "GemmAnima" / "models" / "gemma_core" / "base.gguf"


def test_model_path_uses_existing_legacy_only_when_explicitly_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMMANIMA_MODEL_ROOT", raising=False)
    monkeypatch.setenv("GEMMANIMA_ALLOW_LEGACY_MODEL_PATHS", "1")
    legacy = tmp_path / "legacy.gguf"
    legacy.write_bytes(b"legacy")

    resolved = model_path("gemma_core", "base.gguf", legacy)

    assert resolved == legacy


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
    assert data["models"]["registry_health"]["gemma_core.shared_base_gguf"]["role"] == "language_core"
    assert data["models"]["model_parts"]["hiddenstage_bridge"]["label"] == "HiddenStage Bridge"


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
