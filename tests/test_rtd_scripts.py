from pathlib import Path


def test_single_windows_batch_launcher_is_available() -> None:
    path = Path("GemmAnima.bat")

    assert path.exists(), "missing single user-facing launcher"
    text = path.read_text(encoding="utf-8")
    assert text.isascii(), "Windows launcher messages must stay ASCII-only"
    assert 'cd /d "%~dp0"' in text
    assert 'set "PY=%VENV_DIR%\\Scripts\\python.exe"' in text
    assert 'set "PYTHONUNBUFFERED=1"' in text
    assert 'set "PYTHONIOENCODING=utf-8"' in text
    assert "python -m venv --system-site-packages" in text
    assert '"%PY%" -u -m gemmanima.server --host 127.0.0.1 --port 8765 --base-dir runs' in text
    assert "Starting local GUI backend on http://127.0.0.1:8765" in text
    assert "Runtime logs will stream in this terminal window." in text
    assert '"%PY%" -u -m gemmanima.cli ensure-model-assets --json' in text
    assert '"%PY%" -u -m gemmanima.cli run' in text
    assert '"%PY%" -u -m gemmanima.cli tag-image' in text


def test_rtd_scripts_have_no_extra_windows_batch_launchers() -> None:
    scripts = Path("RTD") / "scripts"

    assert list(scripts.glob("*.bat")) == []


def test_readmes_prefer_single_batch_launcher() -> None:
    root_text = Path("README.md").read_text(encoding="utf-8")
    rtd_text = Path("RTD/README.md").read_text(encoding="utf-8")

    for text in (root_text, rtd_text):
        assert "GemmAnima.bat" in text
        assert r"RTD\scripts\run_gui.bat" not in text
        assert r"RTD\scripts\health_check.bat" not in text
        assert r"RTD\scripts\smoke_dry_run.bat" not in text
        assert r"RTD\scripts\tag_image.bat" not in text
