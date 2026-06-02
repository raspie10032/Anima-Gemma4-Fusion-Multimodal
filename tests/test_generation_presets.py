from gemmanima.core.generation_presets import (
    COMFYUI_SAMPLERS,
    COMFYUI_SCHEDULERS,
    GENERATION_PRESETS,
    RESOLUTION_PRESETS,
    apply_generation_preset,
    generation_preset_options,
    normalize_anima_cfg,
    resolution_preset_options,
    sampler_options,
    scheduler_options,
)
from gemmanima.core.schemas import GenerationPlan


def test_resolution_preset_options_match_supported_sizes() -> None:
    assert resolution_preset_options() == [
        {"name": "square_1024", "label": "1024 x 1024", "width": 1024, "height": 1024, "custom": False},
        {"name": "portrait_832_1216", "label": "832 x 1216", "width": 832, "height": 1216, "custom": False},
        {"name": "portrait_768_1344", "label": "768 x 1344", "width": 768, "height": 1344, "custom": False},
        {"name": "custom", "label": "Custom", "width": None, "height": None, "custom": True},
    ]


def test_apply_generation_preset_keeps_prompt_and_sets_render_defaults() -> None:
    plan = apply_generation_preset(
        GenerationPlan(prompt="1girl, solo", width=512, height=512, steps=8, cfg=3.0),
        preset_name="anima_balanced",
        resolution_name="portrait_832_1216",
    )

    assert plan.prompt == "1girl, solo"
    assert plan.width == 832
    assert plan.height == 1216
    assert plan.steps == GENERATION_PRESETS["anima_balanced"].steps
    assert plan.cfg == GENERATION_PRESETS["anima_balanced"].cfg
    assert plan.sampler == GENERATION_PRESETS["anima_balanced"].sampler
    assert plan.scheduler == GENERATION_PRESETS["anima_balanced"].scheduler
    assert plan.lora_stack == GENERATION_PRESETS["anima_balanced"].lora_stack


def test_apply_generation_preset_supports_swapped_orientation() -> None:
    plan = apply_generation_preset(
        GenerationPlan(prompt="wide scene"),
        preset_name="anima_balanced",
        resolution_name="portrait_832_1216",
        orientation="landscape",
    )

    assert plan.width == 1216
    assert plan.height == 832


def test_apply_generation_preset_uses_custom_dimensions_when_requested() -> None:
    plan = apply_generation_preset(
        GenerationPlan(prompt="custom scene"),
        preset_name="anima_draft",
        resolution_name="custom",
        custom_width=1152,
        custom_height=896,
    )

    assert plan.width == 1152
    assert plan.height == 896
    assert plan.steps == GENERATION_PRESETS["anima_draft"].steps


def test_apply_generation_preset_allows_explicit_payload_overrides() -> None:
    plan = apply_generation_preset(
        GenerationPlan(prompt="override scene"),
        preset_name="anima_final",
        resolution_name="square_1024",
        overrides={"steps": 9, "sampler": "euler", "scheduler": "normal", "lora_stack": ("custom-anima",)},
    )

    assert plan.steps == 9
    assert plan.sampler == "euler"
    assert plan.scheduler == "normal"
    assert plan.lora_stack == ("custom-anima",)


def test_apply_generation_preset_clamps_low_cfg_by_default() -> None:
    plan = apply_generation_preset(
        GenerationPlan(prompt="low cfg scene"),
        preset_name="anima_balanced",
        overrides={"cfg": 1.0},
    )

    assert plan.cfg == 4.5


def test_apply_generation_preset_allows_low_cfg_for_experiments() -> None:
    plan = apply_generation_preset(
        GenerationPlan(prompt="low cfg experiment"),
        preset_name="anima_balanced",
        overrides={"cfg": 1.0},
        allow_low_cfg=True,
    )

    assert plan.cfg == 1.0
    assert normalize_anima_cfg(1.0, allow_low_cfg=True) == 1.0


def test_generation_preset_options_are_payload_safe() -> None:
    options = generation_preset_options()

    assert options[0]["name"] == "anima_draft"
    assert options[0]["sampler"]
    assert options[0]["scheduler"]
    assert "anima_lora" in {option["name"] for option in options}
    assert "custom" in RESOLUTION_PRESETS


def test_sampler_and_scheduler_options_are_curated_from_comfyui_ksampler() -> None:
    assert set(sampler_options()).issubset(COMFYUI_SAMPLERS)
    assert set(scheduler_options()).issubset(COMFYUI_SCHEDULERS)
    assert sampler_options() == ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde_gpu"]
    assert scheduler_options() == ["normal", "karras", "sgm_uniform"]
    assert "euler" in sampler_options()
    assert "euler_ancestral" in sampler_options()
    assert "dpmpp_2m_sde_gpu" in sampler_options()
    assert "normal" in scheduler_options()
    assert "sgm_uniform" in scheduler_options()
