import sys
from pathlib import Path

from gemmanima.rendering.comfy_bootstrap import ComfyBootstrapConfig, bootstrap_comfy


def test_comfy_bootstrap_config_matches_anima_environment() -> None:
    config = ComfyBootstrapConfig()

    assert config.comfy_root == Path(r"E:\ComfyUI_anima_exp")
    assert config.models_root == Path(r"E:\ComfyUI_sage\ComfyUI\models")
    assert "diffusion_models" in config.model_folders


def test_bootstrap_comfy_preserves_argv_when_import_is_disabled(monkeypatch) -> None:
    original_argv = ["prog", "--request", "hello"]
    monkeypatch.setattr(sys, "argv", list(original_argv))

    result = bootstrap_comfy(import_folder_paths=False)

    assert sys.argv == original_argv
    assert result.comfy_root == Path(r"E:\ComfyUI_anima_exp")
    assert result.model_folders["vae"] == Path(r"E:\ComfyUI_sage\ComfyUI\models\vae")
    assert result.project_core == Path(r"E:\anima_gemma_swap\scripts\core")
