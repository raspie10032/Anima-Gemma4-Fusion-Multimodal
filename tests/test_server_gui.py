from pathlib import Path
import threading
import time

from gemmanima.server import (
    GemmAnimaRequestHandler,
    chat_stream_events,
    initialize_server_runtime,
    load_user_settings,
    resolve_image_artifact,
    save_upload_data_url,
    save_user_settings,
    user_settings_path,
)
from gemmanima.ui import GUI_HTML


def test_gui_html_contains_api_hooks() -> None:
    assert "GemmAnima" in GUI_HTML
    assert "/v1/health" in GUI_HTML
    assert "/v1/chat" in GUI_HTML
    assert "/v1/chat/stream" in GUI_HTML
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
    assert 'id="headroom_enabled"' in GUI_HTML
    assert "headroom_enabled:" in GUI_HTML
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
    assert "reference_image_path" in GUI_HTML
    assert "payload.reference_image_path = attachedImagePath" in GUI_HTML
    assert "setAttachment(data.path, file.name)" in GUI_HTML
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
    assert "attachImage(path" in GUI_HTML
    assert "clearAttachment()" in GUI_HTML
    assert "attachedImageName" in GUI_HTML
    assert "tagResultText" in GUI_HTML
    assert 'data.mode === "tag_image"' in GUI_HTML
    assert "shouldTagAttachedImage" not in GUI_HTML
    assert "shouldTagThenGenerateAttachedImage" not in GUI_HTML
    assert 'payload.task = "tag_then_generate"' not in GUI_HTML
    assert 'payload.task = "tag"' not in GUI_HTML
    assert "runBrowserAutotest" in GUI_HTML
    assert "autotest-status" in GUI_HTML
    assert "renderPendingGeneration" in GUI_HTML
    assert "renderThinkingBubble" in GUI_HTML
    assert "runStreamingRequest" in GUI_HTML
    assert "const pending = renderThinkingBubble(payload);" in GUI_HTML
    assert "fetch(\"/v1/chat/stream\"" in GUI_HTML
    assert "isLikelyGenerationRequest(message, payload) ? renderPendingGeneration(payload)" not in GUI_HTML
    assert "updatePendingGenerationStage" in GUI_HTML
    assert "generation-spinner" in GUI_HTML
    assert "generation-stage" in GUI_HTML
    assert "replacePendingGeneration" in GUI_HTML
    assert "generationWords" in GUI_HTML
    assert "metaQuestionWords" in GUI_HTML
    assert "이미지 요청이 아닌" in GUI_HTML
    assert "이미지 생성 요청을 어떻게 구분" in GUI_HTML
    assert "이미지를 만들어줘" in GUI_HTML
    assert "not image" in GUI_HTML


def test_gui_html_supports_first_run_bot_name_setup() -> None:
    assert 'id="name-setup"' in GUI_HTML
    assert 'id="bot-name-input"' in GUI_HTML
    assert 'id="bot-name-save"' in GUI_HTML
    assert 'id="bot-name-reset"' in GUI_HTML
    assert "/v1/settings/chatbot-name" in GUI_HTML
    assert "localStorage" not in GUI_HTML
    assert "loadBotName" in GUI_HTML
    assert "saveBotName" in GUI_HTML
    assert "applyBotName" in GUI_HTML
    assert "showNameSetupIfNeeded" in GUI_HTML
    assert 'data-bot-name-target="title"' in GUI_HTML
    assert 'data-bot-name-target="avatar"' in GUI_HTML
    assert 'data-bot-name-target="meta"' in GUI_HTML


def test_user_settings_are_saved_under_app_root_settings_dir(tmp_path: Path) -> None:
    path = user_settings_path(tmp_path)

    assert path == tmp_path / "settings" / "user_settings.json"

    saved = save_user_settings(tmp_path, {"chatbot_name": "Mina"})

    assert saved["chatbot_name"] == "Mina"
    assert path.is_file()
    assert load_user_settings(tmp_path)["chatbot_name"] == "Mina"


def test_user_settings_sanitize_blank_chatbot_name(tmp_path: Path) -> None:
    saved = save_user_settings(tmp_path, {"chatbot_name": "   "})

    assert saved["chatbot_name"] == "GemmAnima"


def test_server_root_uses_gui_html() -> None:
    assert GemmAnimaRequestHandler.base_dir is not None
    assert "GemmAnima" in GUI_HTML


def test_chat_stream_events_reports_thinking_before_complete(tmp_path: Path) -> None:
    def fake_handler(payload, *, base_dir):
        assert payload["message"] == "draw a forest girl"
        assert base_dir == tmp_path
        return {
            "mode": "generate_image",
            "status": "completed",
            "message": "이미지를 만들었습니다.",
            "output_path": "runs/images/sample.png",
            "progress": ["route:image", "renderer:complete"],
        }

    events = list(
        chat_stream_events(
            {"task": "auto", "message": "draw a forest girl"},
            base_dir=tmp_path,
            handler=fake_handler,
        )
    )

    assert [event["type"] for event in events] == ["thinking", "routing", "working", "complete"]
    assert events[0]["stage"] == "thinking"
    assert events[1]["stage"] == "routing"
    assert events[2]["stage"] == "generating"
    assert events[-1]["data"]["mode"] == "generate_image"


def test_chat_stream_events_reports_tagging_for_tag_then_generate(tmp_path: Path) -> None:
    def fake_handler(payload, *, base_dir):
        return {"mode": "generate_image", "status": "completed", "message": "done"}

    events = list(
        chat_stream_events(
            {"task": "tag_then_generate", "message": "tag then generate", "image_path": "input.png"},
            base_dir=tmp_path,
            handler=fake_handler,
        )
    )

    assert events[2]["stage"] == "tagging"


def test_chat_stream_events_yields_before_handler_completes(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_handler(payload, *, base_dir):
        started.set()
        assert release.wait(timeout=2)
        return {"mode": "chat", "status": "completed", "message": "done"}

    events = chat_stream_events({"task": "auto", "message": "hello"}, base_dir=tmp_path, handler=slow_handler)

    first = next(events)
    assert started.wait(timeout=1)
    assert first["type"] == "thinking"
    assert next(events)["type"] == "routing"
    assert next(events)["type"] == "working"
    release.set()
    deadline = time.time() + 2
    final = None
    while time.time() < deadline:
        final = next(events)
        if final["type"] == "complete":
            break
    assert final["type"] == "complete"
    assert final["data"]["message"] == "done"


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
