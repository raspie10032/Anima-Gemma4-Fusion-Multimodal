import json
from pathlib import Path

from gemmanima.api import build_config, handle_chat_payload, handle_health_payload, select_bridge_profile_name
from gemmanima.modules.gemma_planner import GemmaPlannerAdapter


def test_handle_chat_payload_requires_message(tmp_path) -> None:
    result = handle_chat_payload({}, base_dir=tmp_path)

    assert result["status"] == "failed"
    assert "message is required" in result["error"]


def test_handle_chat_payload_routes_tag_task_without_generation(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")

    def fake_tag_image(*, image_path: str | Path, prompt: str | None = None, **kwargs):
        assert Path(image_path) == image_path_obj
        assert prompt == "tag this"
        return {
            "status": "completed",
            "tags": "1girl, solo, cat ears",
            "raw": "1girl, solo, cat ears",
            "stderr_tail": "",
            "seconds": 0.25,
            "command": ["llama-mtmd-cli.exe"],
            "model": "vision.gguf",
            "mmproj": "vision.mmproj",
            "device": "CUDA0",
        }

    image_path_obj = image_path
    monkeypatch.setattr("gemmanima.api.run_tipo_vision_tag", fake_tag_image)

    result = handle_chat_payload(
        {
            "task": "tag",
            "message": "tag this",
            "image_path": str(image_path),
            "session_id": "tag-test",
        },
        base_dir=tmp_path,
    )

    assert result["mode"] == "tag_image"
    assert result["status"] == "completed"
    assert result["tags"] == "1girl, solo, cat ears"
    assert result["progress"] == ["route:tag", "tipo:vision"]
    assert result["manifest_path"] is None
    assert result["output_path"] is None


def test_handle_chat_payload_cleans_tag_output_at_api_boundary(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    raw_tags = (
        "1girl, solo, <start_of_turn>user, You are a helpful assistant, "
        "Hello<end_of_turn>, <start_of_turn>model, Hi there<end_of_turn>, "
        + ", ".join(f"tag_{index}" for index in range(1, 40))
    )

    def fake_tag_image(*, image_path: str | Path, prompt: str | None = None, **kwargs):
        return {
            "status": "completed",
            "tags": raw_tags,
            "raw": raw_tags,
            "stderr_tail": "",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_vision_tag", fake_tag_image)

    result = handle_chat_payload(
        {"task": "tag", "message": "tag this", "image_path": str(image_path)},
        base_dir=tmp_path,
    )

    tags = [tag.strip() for tag in result["tags"].split(",")]
    assert len(tags) == 24
    assert tags[:4] == ["1girl", "solo", "tag_1", "tag_2"]
    assert not any("<start_of_turn>" in tag or "<end_of_turn>" in tag for tag in tags)


def test_handle_chat_payload_tag_task_requires_image_path(tmp_path) -> None:
    result = handle_chat_payload({"task": "tag", "message": "tag this"}, base_dir=tmp_path)

    assert result["status"] == "failed"
    assert "image_path is required" in result["error"]


def test_handle_chat_payload_chat_task_does_not_route_to_generation(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        assert kwargs["message"] == "draw라는 단어를 설명해줘"
        return {
            "status": "completed",
            "message": "draw는 그리다라는 뜻입니다.",
            "raw": "draw는 그리다라는 뜻입니다.",
            "stderr_tail": "",
            "seconds": 0.2,
            "model": "chat.gguf",
            "device": "CUDA0",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {"task": "chat", "message": "draw라는 단어를 설명해줘"},
        base_dir=tmp_path,
    )

    assert result["mode"] == "chat"
    assert result["status"] == "completed"
    assert result["message"] == "draw는 그리다라는 뜻입니다."
    assert result["chat_mode"] == "general_chat"
    assert result["output_contract"] == "natural_language_answer"
    assert result["manifest_path"] is None
    assert result["output_path"] is None


def test_handle_chat_payload_auto_general_chat_uses_gemma_text_runtime(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_text_chat(**kwargs):
        calls.append(kwargs)
        if kwargs["chat_mode"] == "intent_classification":
            return {
                "status": "completed",
                "message": '{"intent":"chat","confidence":0.97,"reason":"general question"}',
                "raw": '{"intent":"chat","confidence":0.97,"reason":"general question"}',
                "stderr_tail": "",
                "seconds": 0.05,
                "model": "chat.gguf",
                "device": "CUDA0",
                "language": kwargs["language"],
                "chat_mode": kwargs["chat_mode"],
                "output_contract": "route_intent_json",
            }
        return {
            "status": "completed",
            "message": "안녕하세요. GemmAnima 채팅 경로입니다.",
            "raw": "안녕하세요. GemmAnima 채팅 경로입니다.",
            "stderr_tail": "",
            "seconds": 0.2,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": kwargs["language"],
            "chat_mode": kwargs["chat_mode"],
            "output_contract": "natural_language_answer",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "auto",
            "message": "오늘은 어떤 일을 도와줄 수 있어?",
            "language": "ko",
            "history": [{"role": "user", "content": "안녕"}],
        },
        base_dir=tmp_path,
    )

    assert calls
    assert calls[0]["chat_mode"] == "intent_classification"
    assert calls[1]["message"] == "오늘은 어떤 일을 도와줄 수 있어?"
    assert calls[1]["chat_mode"] == "general_chat"
    assert calls[1]["history"] == [{"role": "user", "content": "안녕"}]
    assert result["mode"] == "chat"
    assert result["message"] == "안녕하세요. GemmAnima 채팅 경로입니다."
    assert result["progress"] == ["route:chat", "tipo:text"]
    assert result["manifest_path"] is None
    assert result["output_path"] is None


def test_handle_chat_payload_auto_does_not_treat_negated_image_meta_text_as_generation(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_text_chat(**kwargs):
        calls.append(kwargs)
        if kwargs["chat_mode"] == "intent_classification":
            return {
                "status": "completed",
                "message": '{"intent":"chat","confidence":0.99,"reason":"negated image request"}',
                "raw": '{"intent":"chat","confidence":0.99,"reason":"negated image request"}',
                "stderr_tail": "",
                "seconds": 0.05,
                "model": "chat.gguf",
                "device": "CUDA0",
                "language": kwargs["language"],
                "chat_mode": kwargs["chat_mode"],
                "output_contract": "route_intent_json",
            }
        return {
            "status": "completed",
            "message": "반갑습니다.",
            "raw": "반갑습니다.",
            "stderr_tail": "",
            "seconds": 0.2,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": kwargs["language"],
            "chat_mode": kwargs["chat_mode"],
            "output_contract": "natural_language_answer",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "auto",
            "message": "안녕. 지금 한국어로 짧게 자기소개하고, 이미지 요청이 아니면 그냥 대화로 답해줘.",
            "language": "ko",
        },
        base_dir=tmp_path,
    )

    assert calls
    assert result["mode"] == "chat"
    assert result["status"] == "completed"
    assert result["message"] == "반갑습니다."
    assert result["manifest_path"] is None
    assert result["output_path"] is None


def test_planner_does_not_route_image_meta_questions_to_generation() -> None:
    planner = GemmaPlannerAdapter()

    assert planner.is_image_request("이미지 요청이 아닌 일반 잡담으로 응답해줘.") is False
    assert planner.is_image_request("일반 채팅과 이미지 생성 요청을 어떻게 구분해야 해?") is False
    assert planner.is_image_request("이미지 품질을 올리려면 어떤 요소를 봐야 해?") is False
    assert planner.is_image_request("1024x1024 이미지 만들어줘") is True
    assert planner.is_image_request("draw a small moonlit garden") is True


def test_handle_chat_payload_generates_image_response(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        if kwargs["chat_mode"] == "intent_classification":
            return {
                "status": "completed",
                "message": '{"intent":"generate_image","confidence":0.98,"reason":"draw request"}',
                "raw": '{"intent":"generate_image","confidence":0.98,"reason":"draw request"}',
                "stderr_tail": "",
                "seconds": 0.05,
                "model": "chat.gguf",
                "device": "CUDA0",
                "language": kwargs["language"],
                "chat_mode": kwargs["chat_mode"],
                "output_contract": "route_intent_json",
            }
        return {
            "status": "completed",
            "message": '{"intent":"generate_image","prompt":"small moonlit garden","negative_prompt":"low quality","notes":"준비됨"}',
            "raw": '{"intent":"generate_image","prompt":"small moonlit garden","negative_prompt":"low quality","notes":"준비됨"}',
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": kwargs["language"],
            "chat_mode": kwargs["chat_mode"],
            "output_contract": "image_generation_json",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {"message": "draw a small moonlit garden", "session_id": "api-test"},
        base_dir=tmp_path,
    )

    assert result["mode"] == "generate_image"
    assert result["status"] == "completed"
    assert result["manifest_path"]
    assert result["output_path"]


def test_handle_chat_payload_passes_selected_chat_language(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        assert kwargs["message"] == "Explain the pipeline."
        assert kwargs["language"] == "en"
        return {
            "status": "completed",
            "message": "The pipeline is ready.",
            "raw": "The pipeline is ready.",
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": "en",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {"task": "chat", "message": "Explain the pipeline.", "language": "en"},
        base_dir=tmp_path,
    )

    assert result["mode"] == "chat"
    assert result["status"] == "completed"
    assert result["message"] == "The pipeline is ready."
    assert result["language"] == "en"


def test_handle_chat_payload_passes_selected_chat_mode(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        assert kwargs["message"] == "Where is training now?"
        assert kwargs["language"] == "ko"
        assert kwargs["chat_mode"] == "status_question"
        return {
            "status": "completed",
            "message": "현재 상태를 확인하려면 최신 로그가 필요합니다.",
            "raw": "현재 상태를 확인하려면 최신 로그가 필요합니다.",
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": "ko",
            "chat_mode": "status_question",
            "output_contract": "grounded_status_answer",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {"task": "chat", "message": "Where is training now?", "chat_mode": "status_question"},
        base_dir=tmp_path,
    )

    assert result["mode"] == "chat"
    assert result["status"] == "completed"
    assert result["chat_mode"] == "status_question"
    assert result["output_contract"] == "grounded_status_answer"


def test_handle_chat_payload_passes_headroom_toggle_to_text_runtime(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        assert kwargs["config"].headroom_enabled is True
        assert kwargs["config"].headroom_timeout_seconds == 0.25
        return {
            "status": "completed",
            "message": "Headroom is enabled locally.",
            "raw": "Headroom is enabled locally.",
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": "en",
            "chat_mode": "general_chat",
            "output_contract": "natural_language_answer",
            "headroom": {"enabled": True, "used": False, "status": "unavailable"},
            "warnings": ["Headroom context compressor is unavailable; continuing with uncompressed GemmAnima chat context."],
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "chat",
            "message": "Use compression?",
            "language": "en",
            "headroom_enabled": True,
            "headroom_timeout_seconds": 0.25,
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["headroom"]["status"] == "unavailable"
    assert result["warnings"]


def test_handle_chat_payload_propagates_tipo_preflight_failure(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        return {
            "status": "failed",
            "error": "missing model: chat.gguf",
            "error_code": "preflight_failed",
            "preflight": {
                "ready": False,
                "blocking": True,
                "issues": [
                    {
                        "code": "missing_tipo_text_model",
                        "scope": "tipo_text",
                        "asset": "model",
                        "path": "chat.gguf",
                        "severity": "error",
                        "message_ko": "채팅 모델 파일을 찾을 수 없습니다.",
                        "message_en": "Chat model is missing.",
                    }
                ],
            },
            "language": "ko",
            "chat_mode": "general_chat",
            "output_contract": "natural_language_answer",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {"task": "chat", "message": "안녕"},
        base_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "preflight_failed"
    assert result["preflight"]["issues"][0]["code"] == "missing_tipo_text_model"


def test_handle_chat_payload_chat_image_mode_generates_from_gemma_contract(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        assert kwargs["chat_mode"] == "image_generation_request"
        return {
            "status": "completed",
            "message": '{"intent":"generate_image","prompt":"1girl, solo, moonlit garden","negative_prompt":"low quality","notes":"준비됨"}',
            "raw": '{"intent":"generate_image","prompt":"1girl, solo, moonlit garden","negative_prompt":"low quality","notes":"준비됨"}',
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": "ko",
            "chat_mode": "image_generation_request",
            "output_contract": "image_generation_json",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "chat",
            "chat_mode": "image_generation_request",
            "message": "달빛 정원 캐릭터 이미지 만들어줘",
            "session_id": "chat-image-test",
        },
        base_dir=tmp_path,
    )

    assert result["mode"] == "generate_image"
    assert result["status"] == "completed"
    assert result["chat_mode"] == "image_generation_request"
    assert result["output_contract"] == "image_generation_json"
    assert result["chat_notes"] == "준비됨"
    assert result["manifest_path"]
    assert result["output_path"]


def test_handle_chat_payload_chat_image_mode_applies_generation_preset(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        return {
            "status": "completed",
            "message": (
                '{"intent":"generate_image","prompt":"1girl, solo",'
                '"negative_prompt":"low quality","width":512,"height":512,"steps":4,'
                '"cfg":2.5,"sampler":"wrong","scheduler":"wrong"}'
            ),
            "raw": (
                '{"intent":"generate_image","prompt":"1girl, solo",'
                '"negative_prompt":"low quality","width":512,"height":512,"steps":4,'
                '"cfg":2.5,"sampler":"wrong","scheduler":"wrong"}'
            ),
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": "ko",
            "chat_mode": "image_generation_request",
            "output_contract": "image_generation_json",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "chat",
            "chat_mode": "image_generation_request",
            "message": "이미지 만들어줘",
            "generation_preset": "anima_balanced",
            "resolution_preset": "portrait_832_1216",
            "orientation": "landscape",
            "session_id": "chat-image-preset-test",
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["generation_preset"] == "anima_balanced"
    assert result["resolution_preset"] == "portrait_832_1216"
    assert result["plan"]["width"] == 1216
    assert result["plan"]["height"] == 832
    assert result["plan"]["steps"] == 28
    assert result["plan"]["sampler"] == "euler_ancestral"
    assert result["plan"]["scheduler"] == "sgm_uniform"


def test_handle_chat_payload_chat_image_mode_rejects_bad_contract(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        return {
            "status": "completed",
            "message": "I will make a pretty image.",
            "raw": "I will make a pretty image.",
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": "ko",
            "chat_mode": "image_generation_request",
            "output_contract": "image_generation_json",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "chat",
            "chat_mode": "image_generation_request",
            "message": "이미지 만들어줘",
        },
        base_dir=tmp_path,
    )

    assert result["mode"] == "chat"
    assert result["status"] == "failed"
    assert result["error_code"] == "image_generation_contract_failed"
    assert result["output_contract"] == "image_generation_json"
    assert not list((tmp_path / "images").glob("*"))


def test_handle_chat_payload_auto_image_classifier_falls_back_when_contract_is_bad(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_text_chat(**kwargs):
        calls.append(kwargs)
        if kwargs["chat_mode"] == "intent_classification":
            return {
                "status": "completed",
                "message": '{"intent":"generate_image","confidence":0.98,"reason":"draw request"}',
                "raw": '{"intent":"generate_image","confidence":0.98,"reason":"draw request"}',
                "stderr_tail": "",
                "seconds": 0.05,
                "model": "chat.gguf",
                "device": "CUDA0",
                "language": kwargs["language"],
                "chat_mode": kwargs["chat_mode"],
                "output_contract": "route_intent_json",
            }
        return {
            "status": "completed",
            "message": "",
            "raw": "",
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "chat.gguf",
            "device": "CUDA0",
            "language": kwargs["language"],
            "chat_mode": kwargs["chat_mode"],
            "output_contract": "image_generation_json",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "auto",
            "message": "draw a small moonlit garden",
            "renderer": "dry-run",
        },
        base_dir=tmp_path,
    )

    assert [call["chat_mode"] for call in calls] == ["intent_classification", "image_generation_request"]
    assert result["mode"] == "generate_image"
    assert result["status"] == "completed"
    assert result["auto_route"]["intent"] == "generate_image"
    assert "contract:fallback" in result["progress"]


def test_handle_chat_payload_auto_image_classifier_falls_back_when_contract_call_fails(tmp_path, monkeypatch) -> None:
    def fake_text_chat(**kwargs):
        if kwargs["chat_mode"] == "intent_classification":
            return {
                "status": "completed",
                "message": '{"intent":"generate_image","confidence":0.98,"reason":"draw request"}',
                "raw": '{"intent":"generate_image","confidence":0.98,"reason":"draw request"}',
                "stderr_tail": "",
                "seconds": 0.05,
                "model": "chat.gguf",
                "device": "CUDA0",
                "language": kwargs["language"],
                "chat_mode": kwargs["chat_mode"],
                "output_contract": "route_intent_json",
            }
        return {
            "status": "failed",
            "error": "image generation contract is empty",
            "error_code": "chat_generation_failed",
            "seconds": 0.1,
            "chat_mode": kwargs["chat_mode"],
            "output_contract": "image_generation_json",
        }

    monkeypatch.setattr("gemmanima.api.run_tipo_text_chat", fake_text_chat)
    result = handle_chat_payload(
        {
            "task": "auto",
            "message": "draw a small moonlit garden",
            "renderer": "dry-run",
        },
        base_dir=tmp_path,
    )

    assert result["mode"] == "generate_image"
    assert result["status"] == "completed"
    assert result["contract_error"] == "image generation contract is empty"
    assert "contract:fallback" in result["progress"]


def test_handle_chat_payload_accepts_generation_overrides(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "draw a small moonlit garden",
            "session_id": "api-test",
            "steps": 8,
            "size": 512,
            "cfg": 4.5,
            "seed": 123,
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["output_path"]


def test_handle_chat_payload_clamps_low_cfg_by_default(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "draw a small moonlit garden",
            "session_id": "api-low-cfg-test",
            "steps": 8,
            "size": 512,
            "cfg": 1.0,
            "renderer": "dry-run",
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["plan"]["cfg"] == 4.5


def test_handle_chat_payload_can_allow_low_cfg_for_experiments(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "draw a small moonlit garden",
            "session_id": "api-allow-low-cfg-test",
            "steps": 8,
            "size": 512,
            "cfg": 1.0,
            "allow_low_cfg": True,
            "renderer": "dry-run",
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["plan"]["cfg"] == 1.0


def test_handle_chat_payload_treats_empty_ui_number_fields_as_defaults(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "task": "generate",
            "chat_mode": "image_generation_request",
            "message": "draw a small moonlit garden",
            "generation_preset": "anima_balanced",
            "resolution_preset": "square_1024",
            "steps": "",
            "cfg": "",
            "seed": "",
            "renderer": "dry-run",
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["output_path"]


def test_build_config_accepts_hiddenstage_bridge_override(tmp_path) -> None:
    bridge = tmp_path / "kv_proj_text_delta_300k_from_epoch1_a0p35.pt"

    config = build_config({"hiddenstage_bridge": str(bridge)})

    assert config.models.hiddenstage_bridge == bridge


def test_build_config_accepts_explicit_bridge_profile() -> None:
    config = build_config({"bridge_profile": "style_artist"})

    assert config.models.hiddenstage_bridge == config.models.hiddenstage_bridge_style_artist


def test_build_config_routes_style_hints_to_style_bridge() -> None:
    config = build_config({"message": "draw 1girl, by soft watercolor style, gentle forest light"})

    assert select_bridge_profile_name({"message": "draw by soft watercolor style"}) == "style_artist"
    assert config.models.hiddenstage_bridge == config.models.hiddenstage_bridge_style_artist


def test_build_config_routes_text_prompts_to_text_bridge() -> None:
    config = build_config({"message": 'make a shop sign that clearly reads "LUNA GATE"'})

    assert select_bridge_profile_name({"message": 'logo text "TEA"'}) == "text_exact"
    assert config.models.hiddenstage_bridge == config.models.hiddenstage_bridge_text_exact


def test_build_config_hiddenstage_bridge_path_wins_over_profile(tmp_path) -> None:
    bridge = tmp_path / "manual_bridge.pt"

    config = build_config({"hiddenstage_bridge": str(bridge), "bridge_profile": "style_artist"})

    assert config.models.hiddenstage_bridge == bridge


def test_handle_chat_payload_rejects_unknown_renderer(tmp_path) -> None:
    result = handle_chat_payload(
        {"task": "generate", "message": "draw a small moonlit garden", "renderer": "missing"},
        base_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert "unknown renderer" in result["error"]


def test_handle_generate_payload_preserves_attached_reference_image(tmp_path) -> None:
    reference = tmp_path / "uploads" / "reference.png"
    reference.parent.mkdir()
    reference.write_bytes(b"fake png")

    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "make a brighter variation of the attached image",
            "renderer": "dry-run",
            "image_path": str(reference),
            "steps": 5,
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["plan"]["reference_image_path"] == str(reference)
    assert "attachment:reference" in result["progress"]


def test_explicit_generate_payload_does_not_fall_back_to_chat_for_korean_attachment_request(tmp_path) -> None:
    reference = tmp_path / "uploads" / "reference.png"
    reference.parent.mkdir()
    reference.write_bytes(b"fake png")

    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "이 첨부 이미지를 참고해서 더 밝은 숲 장면으로 새로 구성해줘",
            "renderer": "dry-run",
            "image_path": str(reference),
        },
        base_dir=tmp_path,
    )

    assert result["mode"] == "generate_image"
    assert result["status"] == "completed"
    assert result["plan"]["reference_image_path"] == str(reference)


def test_build_renderer_accepts_local_worker(tmp_path) -> None:
    from gemmanima.api import build_renderer
    from gemmanima.modules.local_worker_anima_renderer import LocalWorkerAnimaRendererAdapter

    renderer = build_renderer({"renderer": "local-worker"}, image_root=tmp_path, config=build_config({}))

    assert isinstance(renderer, LocalWorkerAnimaRendererAdapter)


def test_build_renderer_forwards_vram_saving_options(tmp_path) -> None:
    from gemmanima.api import build_renderer

    renderer = build_renderer(
        {
            "renderer": "local-worker",
            "tiled_vae": True,
            "cpu_vae": True,
            "reserve_vram": 1.5,
            "memory_mode": "lowvram",
        },
        image_root=tmp_path,
        config=build_config({}),
    )

    assert renderer.tiled_vae is True
    assert renderer.comfy_args == ("--lowvram", "--cpu-vae", "--reserve-vram", "1.5")


def test_handle_chat_payload_exposes_conflict_clarification(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "draw the same character in a ruined cathedral but make her black-haired",
            "history": [
                {
                    "role": "user",
                    "content": "Reference image: silver hair, blue cloak, calm face",
                }
            ],
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "ask_clarify"
    assert result["clarification_required"] is True
    assert result["output_path"] is None
    assert result["conflict"]["requires_user_confirmation"] is True
    assert result["conflict"]["fields"] == ["hair_color"]


def test_handle_chat_payload_allows_resolved_conflict(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "task": "generate",
            "message": "draw the same character in a ruined cathedral but make her black-haired",
            "history": [
                {
                    "role": "user",
                    "content": "Reference image: silver hair, blue cloak, calm face",
                },
                {
                    "role": "assistant",
                    "content": "The reference has silver hair, but the request asks for black hair.",
                },
                {
                    "role": "user",
                    "content": "Change it to black hair.",
                },
            ],
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["clarification_required"] is False
    assert result["conflict"] is None
    assert result["output_path"]


def test_handle_chat_payload_resumes_conflict_from_history_reply(tmp_path) -> None:
    result = handle_chat_payload(
        {
            "message": "Change it to black hair.",
            "history": [
                {
                    "role": "user",
                    "content": "Reference image: silver hair, blue cloak, calm face",
                },
                {
                    "role": "user",
                    "content": "draw the same character in a ruined cathedral but make her black-haired",
                },
                {
                    "role": "assistant",
                    "content": "The reference has silver hair, but the request asks for black hair. Should I preserve the reference hair color or change it?",
                },
            ],
        },
        base_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["clarification_required"] is False
    assert "clarification:resume" in result["progress"]
    assert result["output_path"]


def test_handle_health_payload_reports_models() -> None:
    result = handle_health_payload()

    assert result["status"] == "ok"
    assert "ready" in result
    assert "preflight" in result
    assert "gemma_core.shared_base_gguf" in result["models"]
    assert "anima_image_core.diffusion_model" in result["models"]
    assert "hiddenstage_bridge.bridge_checkpoint" in result["models"]
    assert "hiddenstage_bridge.bridge_style_artist" in result["models"]
    assert "style_artist" in result["bridge_profiles"]
    assert "anima_image_core.text_encoder" not in result["models"]
    assert isinstance(result["hiddenstage_bridge"]["passed_mse_gate"], bool)
    assert "in_process" in result["renderers"]
