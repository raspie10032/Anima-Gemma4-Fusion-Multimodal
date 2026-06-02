from __future__ import annotations

PROJECT_PURPOSE = (
    "Build one local Gemma4 + Anima/GEMMANIMA model that preserves Gemma4 chat "
    "and multimodal vision while feeding Gemma4 hidden states directly into "
    "Anima synthesis, so chat, image vision, and image generation live in one "
    "runtime."
)

NON_GOALS = (
    "Do not use Codex image generation.",
    "Do not treat ComfyUI, NovelAI, or another server as a later external backend.",
    "Do not collapse normal chat and image tag output into one contaminated mode.",
    "Do not hide planner model defects with runtime tag blacklists or bias filters.",
    "Do not make a prompt-only or tag-only handoff the final image generation path.",
)

REQUIRED_CAPABILITIES = ("chat", "image_vision_understanding", "image_to_tags", "image_generation")
AUXILIARY_MODES = ("tag", "planner")
REQUIRED_VISION_MODULES = ("image_understander", "vision_tagger")
INTERNAL_GENERATOR_ID = "anima-gemmanima-image-generator"
MODEL_ARCHITECTURE_ID = "gemma4-hidden-state-to-anima-synthesis"
REQUIRED_IMAGE_ENGINE = "gemma4-hidden-state-anima-synthesizer"
CURRENT_RUNTIME_BRANCH = "quantized-llamacpp"
FUTURE_RUNTIME_BRANCH = "unquantized-transformers-or-native"


def project_contract() -> dict[str, object]:
    return {
        "purpose": PROJECT_PURPOSE,
        "required_capabilities": list(REQUIRED_CAPABILITIES),
        "auxiliary_modes": list(AUXILIARY_MODES),
        "required_vision_modules": list(REQUIRED_VISION_MODULES),
        "model_architecture": MODEL_ARCHITECTURE_ID,
        "single_gemma4_core_required": True,
        "preserve_gemma4_chat": True,
        "preserve_gemma4_multimodal_vision": True,
        "separate_image_understanding_and_tagging": True,
        "hidden_state_bridge_required": True,
        "direct_anima_synthesis_required": True,
        "role_split_ggufs_are_temporary": True,
        "runtime_branch": CURRENT_RUNTIME_BRANCH,
        "future_runtime_branch": FUTURE_RUNTIME_BRANCH,
        "quantized_runtime_required_now": True,
        "unquantized_runtime_allowed_later": True,
        "planner_module_strategy": "quantized Gemma4 base GGUF plus llama.cpp-compatible LoRA GGUF",
        "planner_bias_handling": "retrain the planner model; do not patch with runtime tag filtering",
        "internal_generator_id": INTERNAL_GENERATOR_ID,
        "required_image_engine": REQUIRED_IMAGE_ENGINE,
        "anima_gemmanima_required": True,
        "non_goals": list(NON_GOALS),
        "external_generation_backends_allowed": False,
    }
