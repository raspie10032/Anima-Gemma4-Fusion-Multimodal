import os
from pathlib import Path

from gemmanima.core.runtime_env import configure_local_render_runtime
from gemmanima.modules.local_worker_anima_renderer import _default_worker_python
from gemmanima.rendering.backends import renderer_backend_profile


def test_configure_local_render_runtime_detects_common_paths(tmp_path, monkeypatch) -> None:
    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    (comfy_root / "folder_paths.py").write_text("", encoding="utf-8")
    (comfy_root / "nodes.py").write_text("", encoding="utf-8")
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "comfy_aimdo").mkdir()
    worker_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    worker_python.parent.mkdir(parents=True)
    worker_python.write_text("", encoding="utf-8")
    swap_core = tmp_path / "scripts" / "core"
    swap_core.mkdir(parents=True)
    hf_dir = tmp_path / "models" / "gemma_core_hf"
    hf_dir.mkdir(parents=True)
    (hf_dir / "model.safetensors").write_text("", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("", encoding="utf-8")
    for key in (
        "GEMMANIMA_COMFY_ROOT",
        "GEMMANIMA_COMFY_SITE_PACKAGES",
        "GEMMANIMA_RENDER_PYTHON",
        "GEMMANIMA_SWAP_PROJECT_ROOT",
        "GEMMANIMA_GEMMA_HF_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    payload = configure_local_render_runtime()

    assert os.environ["GEMMANIMA_COMFY_ROOT"] == str(Path("ComfyUI"))
    assert os.environ["GEMMANIMA_COMFY_SITE_PACKAGES"] == str(Path(".venv/Lib/site-packages"))
    assert os.environ["GEMMANIMA_RENDER_PYTHON"] == str(Path(".venv/Scripts/python.exe"))
    assert os.environ["GEMMANIMA_SWAP_PROJECT_ROOT"] == "."
    assert os.environ["GEMMANIMA_GEMMA_HF_DIR"] == str(Path("models/gemma_core_hf"))
    assert payload["configured"]["GEMMANIMA_COMFY_ROOT"] == str(Path("ComfyUI"))


def test_runtime_env_does_not_override_explicit_values(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit-python.exe"
    explicit.write_text("", encoding="utf-8")
    monkeypatch.setenv("GEMMANIMA_RENDER_PYTHON", str(explicit))

    configure_local_render_runtime()

    assert _default_worker_python() == explicit
    assert renderer_backend_profile("in_process").checks["embedded_python"] is True
