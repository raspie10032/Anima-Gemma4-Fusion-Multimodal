from gemmanima.api import handle_chat_payload, handle_health_payload


def test_handle_chat_payload_requires_message(tmp_path) -> None:
    result = handle_chat_payload({}, base_dir=tmp_path)

    assert result["status"] == "failed"
    assert "message is required" in result["error"]


def test_handle_chat_payload_generates_image_response(tmp_path) -> None:
    result = handle_chat_payload(
        {"message": "draw a small moonlit garden", "session_id": "api-test"},
        base_dir=tmp_path,
    )

    assert result["mode"] == "generate_image"
    assert result["status"] == "completed"
    assert result["manifest_path"]
    assert result["output_path"]


def test_handle_health_payload_reports_models() -> None:
    result = handle_health_payload()

    assert result["status"] == "ok"
    assert "gemma_planner_adapter" in result["models"]
    assert result["hiddenstage_bridge"]["passed_mse_gate"] is True
