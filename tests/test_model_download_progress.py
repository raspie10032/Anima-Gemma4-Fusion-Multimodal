from __future__ import annotations

import time
from pathlib import Path

from gemmanima.core.model_sources import ModelSource, download_hf_source
from gemmanima.server import ModelDownloadManager


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._data):
            return b""
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_download_hf_source_reports_byte_progress(tmp_path: Path, monkeypatch) -> None:
    payload = b"abcdef"
    events: list[dict[str, object]] = []
    source = ModelSource(
        origin="gemmanima_adapter_bundle",
        repo_id="raspie/gemmanima-adapter-bundle",
        filename="sample.bin",
        url="https://huggingface.co/raspie/gemmanima-adapter-bundle/resolve/main/sample.bin",
    )

    def fake_urlopen(request):
        return _FakeResponse(payload)

    monkeypatch.setattr("gemmanima.core.model_sources.urllib.request.urlopen", fake_urlopen)

    result = download_hf_source(
        source,
        tmp_path / "sample.bin",
        on_progress=events.append,
        chunk_size=2,
    )

    assert result["status"] == "downloaded"
    assert (tmp_path / "sample.bin").read_bytes() == payload
    assert events[0]["status"] == "downloading"
    assert events[-1]["status"] == "downloaded"
    assert events[-1]["downloaded_bytes"] == len(payload)
    assert events[-1]["total_bytes"] == len(payload)
    assert any(event["downloaded_bytes"] == 2 for event in events)


def test_model_download_manager_tracks_started_job(monkeypatch) -> None:
    manager = ModelDownloadManager()

    def fake_run(overwrite: bool = False, names=None):
        manager._update({"status": "running", "completed_assets": 0, "total_assets": 1})
        manager._update({"status": "completed", "completed_assets": 1, "total_assets": 1})
        return {"status": "completed", "assets": []}

    monkeypatch.setattr(manager, "_run_download", fake_run)

    state = manager.start()

    assert state["status"] in {"running", "completed"}
    deadline = time.time() + 1.0
    while manager.status()["status"] != "completed" and time.time() < deadline:
        time.sleep(0.01)
    assert manager.status()["status"] == "completed"
    assert manager.status()["completed_assets"] == 1
