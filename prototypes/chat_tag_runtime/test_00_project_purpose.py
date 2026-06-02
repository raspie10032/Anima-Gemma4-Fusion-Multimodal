from __future__ import annotations

from purpose import INTERNAL_GENERATOR_ID, MODEL_ARCHITECTURE_ID, REQUIRED_IMAGE_ENGINE, project_contract
from runtime import route_request


def test_project_purpose_is_local_chatbot_with_builtin_image_generation() -> None:
    contract = project_contract()

    assert "Gemma4 hidden states directly into Anima synthesis" in contract["purpose"]
    assert contract["required_capabilities"] == ["chat", "image_vision_understanding", "image_to_tags", "image_generation"]
    assert contract["auxiliary_modes"] == ["tag", "planner"]
    assert contract["required_vision_modules"] == ["image_understander", "vision_tagger"]
    assert contract["model_architecture"] == MODEL_ARCHITECTURE_ID
    assert contract["single_gemma4_core_required"] is True
    assert contract["preserve_gemma4_chat"] is True
    assert contract["preserve_gemma4_multimodal_vision"] is True
    assert contract["separate_image_understanding_and_tagging"] is True
    assert contract["hidden_state_bridge_required"] is True
    assert contract["direct_anima_synthesis_required"] is True
    assert contract["role_split_ggufs_are_temporary"] is True
    assert contract["runtime_branch"] == "quantized-llamacpp"
    assert contract["future_runtime_branch"] == "unquantized-transformers-or-native"
    assert contract["quantized_runtime_required_now"] is True
    assert contract["unquantized_runtime_allowed_later"] is True
    assert contract["planner_module_strategy"] == "quantized Gemma4 base GGUF plus llama.cpp-compatible LoRA GGUF"
    assert contract["planner_bias_handling"] == "retrain the planner model; do not patch with runtime tag filtering"
    assert contract["internal_generator_id"] == INTERNAL_GENERATOR_ID
    assert contract["required_image_engine"] == REQUIRED_IMAGE_ENGINE
    assert contract["anima_gemmanima_required"] is True
    assert contract["external_generation_backends_allowed"] is False


def test_image_route_never_points_to_codex_or_external_backend() -> None:
    result = route_request({
        "task": "image",
        "message": "draw a blue-eyed cat-ear wizard",
    })

    assert result["mode"] == "image"
    assert result["generator"] == INTERNAL_GENERATOR_ID
    assert result["external_backend"] is False
    assert "codex" not in str(result).lower()
    assert "comfyui" not in str(result).lower()
    assert "novelai" not in str(result).lower()


def test_auto_route_detects_natural_language_image_requests() -> None:
    result = route_request({
        "task": "auto",
        "message": "blue-eyed cat-ear wizard image please",
    })

    assert result["mode"] == "image"
    assert result["generator"] == INTERNAL_GENERATOR_ID
