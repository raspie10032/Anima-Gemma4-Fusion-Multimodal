import sys
from pathlib import Path

from gemmanima.core.model_paths import default_model_root
from gemmanima.rendering.comfy_bootstrap import ComfyBootstrapConfig, bootstrap_comfy


def test_comfy_bootstrap_config_uses_portable_defaults() -> None:
    config = ComfyBootstrapConfig()

    assert config.comfy_root == Path("ComfyUI")
    assert config.models_root == default_model_root() / "comfy"
    assert config.embedded_site_packages == Path(".venv/Lib/site-packages")
    assert "diffusion_models" in config.model_folders


def test_bootstrap_comfy_preserves_argv_when_import_is_disabled(monkeypatch, tmp_path) -> None:
    original_argv = ["prog", "--request", "hello"]
    monkeypatch.setattr(sys, "argv", list(original_argv))
    monkeypatch.setenv("GEMMANIMA_COMFY_ROOT", str(tmp_path / "ComfyUI"))
    monkeypatch.setenv("GEMMANIMA_COMFY_MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("GEMMANIMA_COMFY_SITE_PACKAGES", str(tmp_path / "site-packages"))
    monkeypatch.setenv("GEMMANIMA_SWAP_PROJECT_ROOT", str(tmp_path / "swap"))

    result = bootstrap_comfy(import_folder_paths=False)

    assert sys.argv == original_argv
    assert result.comfy_root == tmp_path / "ComfyUI"
    assert result.embedded_site_packages == tmp_path / "site-packages"
    assert result.model_folders["vae"] == tmp_path / "models" / "vae"
    assert result.project_core == tmp_path / "swap" / "scripts" / "core"


def test_bootstrap_comfy_blocks_incompatible_native_attention_imports() -> None:
    bootstrap_comfy(import_folder_paths=False)

    assert any(
        "flash_attn" in getattr(finder, "blocked_roots", ())
        for finder in sys.meta_path
    )
