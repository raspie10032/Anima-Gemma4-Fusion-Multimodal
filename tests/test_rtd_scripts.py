from pathlib import Path


def test_windows_batch_launchers_are_available() -> None:
    scripts = Path("RTD") / "scripts"

    for name in ("run_gui.bat", "health_check.bat", "smoke_dry_run.bat", "tag_image.bat"):
        path = scripts / name
        assert path.exists(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        assert 'cd /d "%~dp0\\..\\.."' in text
        assert "python -m gemmanima.cli" in text


def test_rtd_readme_prefers_batch_launchers() -> None:
    text = Path("RTD/README.md").read_text(encoding="utf-8")

    assert r".\RTD\scripts\run_gui.bat" in text
    assert r".\RTD\scripts\health_check.bat" in text
    assert r".\RTD\scripts\smoke_dry_run.bat" in text
