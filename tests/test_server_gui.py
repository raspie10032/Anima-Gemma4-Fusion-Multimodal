from pathlib import Path

from gemmanima.server import (
    GemmAnimaRequestHandler,
    initialize_server_runtime,
    resolve_image_artifact,
    save_upload_data_url,
)
from gemmanima.ui import GUI_HTML


def test_gui_html_contains_api_hooks() -> None:
    assert "GemmAnima" in GUI_HTML
    assert "/v1/health" in GUI_HTML
    assert "/v1/chat" in GUI_HTML
    assert "/v1/uploads" in GUI_HTML
    assert "/v1/models/download" in GUI_HTML
    assert "/v1/models/download/status" in GUI_HTML
    assert 'id="chat-log"' in GUI_HTML
    assert 'class="app-shell"' in GUI_HTML
    assert 'id="drop-overlay"' in GUI_HTML
    assert 'id="file-input"' in GUI_HTML
    assert 'id="attach-button"' in GUI_HTML
    assert "handleDroppedFiles" in GUI_HTML
    assert "uploadImageFile" in GUI_HTML
    assert "attachedImagePath" in GUI_HTML
    assert "addBubble" in GUI_HTML
    assert "responseSummary" in GUI_HTML
    assert 'data.mode === "generate_image" && isImagePath(data.output_path)' in GUI_HTML
    assert 'return data.message || "이미지를 만들었습니다."' in GUI_HTML
    assert 'data.mode && data.mode !== "chat"' in GUI_HTML
    assert "dry-run report" in GUI_HTML
    assert "addAssistantResponse" in GUI_HTML
    assert "isImagePath" in GUI_HTML
    assert "artifactUrl" in GUI_HTML
    assert "/artifacts/images/" in GUI_HTML
    assert "generated-image" in GUI_HTML
    assert "conversationHistory" in GUI_HTML
    assert "history:" in GUI_HTML
    assert 'id="language"' in GUI_HTML
    assert "language:" in GUI_HTML
    assert 'id="force_task"' in GUI_HTML
    assert 'id="force_chat_mode"' in GUI_HTML
    assert 'task: $("force_task").value || "auto"' in GUI_HTML
    assert "payload.chat_mode = forcedChatMode" in GUI_HTML
    assert "<summary>" in GUI_HTML
    assert "<h2>Generate</h2>" not in GUI_HTML
    assert 'id="task"' not in GUI_HTML
    assert 'id="chat_mode"' not in GUI_HTML
    assert "Chat mode" not in GUI_HTML
    assert "draw Nahida from Genshin Impact" not in GUI_HTML
    assert 'placeholder="메시지를 입력하세요."' in GUI_HTML
    assert '<option value="local-worker" selected>local-worker</option>' in GUI_HTML
    assert '<option value="in-process">in-process</option>' in GUI_HTML
    assert '<option value="dry-run">dry-run</option>' in GUI_HTML
    assert 'value="image_generation_request"' in GUI_HTML
    assert 'value="tag"' in GUI_HTML
    assert 'id="image_path"' in GUI_HTML
    assert "image_path:" in GUI_HTML
    assert 'id="generation_preset"' in GUI_HTML
    assert 'id="resolution_preset"' in GUI_HTML
    assert 'value="square_1024"' in GUI_HTML
    assert 'value="portrait_832_1216"' in GUI_HTML
    assert 'value="portrait_768_1344"' in GUI_HTML
    assert 'value="custom"' in GUI_HTML
    assert 'id="sampler"' in GUI_HTML
    assert 'id="scheduler"' in GUI_HTML
    assert 'value="euler_cfg_pp"' not in GUI_HTML
    assert 'value="heun"' not in GUI_HTML
    assert 'value="linear_quadratic"' not in GUI_HTML
    assert 'value="kl_optimal"' not in GUI_HTML
    assert "generation_preset:" in GUI_HTML
    assert "resolution_preset:" in GUI_HTML
    assert "sampler:" in GUI_HTML
    assert "scheduler:" in GUI_HTML
    assert "clarification_required" in GUI_HTML
    assert "sendClarification" in GUI_HTML
    assert "data-conflict-action" in GUI_HTML
    assert "data-conflict-field" in GUI_HTML
    assert 'id="model-download"' in GUI_HTML
    assert 'id="model-download-overall"' in GUI_HTML
    assert 'id="model-download-file"' in GUI_HTML
    assert "updateDownloadGauge" in GUI_HTML
    assert "startModelDownload" in GUI_HTML
    assert "window.GemmAnimaTest" in GUI_HTML
    assert "runBrowserAutotest" in GUI_HTML
    assert "autotest-status" in GUI_HTML
    assert "renderPendingGeneration" in GUI_HTML
    assert "updatePendingGenerationStage" in GUI_HTML
    assert "generation-spinner" in GUI_HTML
    assert "generation-stage" in GUI_HTML
    assert "replacePendingGeneration" in GUI_HTML
    assert "이미지 요청이 아닌" in GUI_HTML
    assert "어떻게" in GUI_HTML
    assert "not image" in GUI_HTML


def test_server_root_uses_gui_html() -> None:
    assert GemmAnimaRequestHandler.base_dir is not None
    assert "GemmAnima" in GUI_HTML


def test_initialize_server_runtime_loads_text_runtime(monkeypatch) -> None:
    calls = []

    def fake_initialize():
        calls.append(True)
        return {"status": "completed", "initialized": True}

    monkeypatch.setattr("gemmanima.server.initialize_tipo_text_runtime", fake_initialize)

    status = initialize_server_runtime()

    assert status["tipo_text"]["status"] == "completed"
    assert calls == [True]


def test_resolve_image_artifact_allows_nested_images(tmp_path: Path) -> None:
    target = resolve_image_artifact(tmp_path, "session/result.png")

    assert target == (tmp_path / "images" / "session" / "result.png").resolve()


def test_resolve_image_artifact_rejects_traversal(tmp_path: Path) -> None:
    assert resolve_image_artifact(tmp_path, "../secret.png") is None
    assert resolve_image_artifact(tmp_path, "session/../../secret.png") is None
    assert resolve_image_artifact(tmp_path, "C:/Windows/win.ini") is None


def test_save_upload_data_url_writes_image_under_uploads(tmp_path: Path) -> None:
    target = save_upload_data_url(tmp_path, "sample.png", "data:image/png;base64,aGVsbG8=")

    assert target.parent == (tmp_path / "uploads").resolve()
    assert target.suffix == ".png"
    assert target.read_bytes() == b"hello"


def test_save_upload_data_url_rejects_non_image(tmp_path: Path) -> None:
    try:
        save_upload_data_url(tmp_path, "sample.txt", "data:text/plain;base64,aGVsbG8=")
    except ValueError as exc:
        assert "image" in str(exc)
    else:
        raise AssertionError("non-image upload should fail")
