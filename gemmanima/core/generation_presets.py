from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from gemmanima.core.schemas import GenerationPlan


# Snapshot from ComfyUI `comfy/samplers.py`:
# `KSampler.SAMPLERS = SAMPLER_NAMES` and `KSampler.SCHEDULERS = SCHEDULER_NAMES`.
COMFYUI_SAMPLERS: tuple[str, ...] = (
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)


COMFYUI_SCHEDULERS: tuple[str, ...] = (
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)


SUPPORTED_SAMPLERS: tuple[str, ...] = (
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde_gpu",
)


SUPPORTED_SCHEDULERS: tuple[str, ...] = (
    "normal",
    "karras",
    "sgm_uniform",
)

MIN_EFFECTIVE_ANIMA_CFG = 3.0
DEFAULT_EFFECTIVE_ANIMA_CFG = 5.0


@dataclass(frozen=True)
class ResolutionPreset:
    name: str
    label: str
    width: int | None
    height: int | None
    custom: bool = False


@dataclass(frozen=True)
class GenerationPreset:
    name: str
    label: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    renderer_profile: str = "anima_fp16_final"
    lora_stack: tuple[str, ...] = ()


RESOLUTION_PRESETS: dict[str, ResolutionPreset] = {
    "square_1024": ResolutionPreset("square_1024", "1024 x 1024", 1024, 1024),
    "portrait_832_1216": ResolutionPreset("portrait_832_1216", "832 x 1216", 832, 1216),
    "portrait_768_1344": ResolutionPreset("portrait_768_1344", "768 x 1344", 768, 1344),
    "custom": ResolutionPreset("custom", "Custom", None, None, custom=True),
}


GENERATION_PRESETS: dict[str, GenerationPreset] = {
    "anima_draft": GenerationPreset(
        name="anima_draft",
        label="Anima Draft",
        steps=16,
        cfg=3.5,
        sampler="euler_ancestral",
        scheduler="sgm_uniform",
        renderer_profile="anima_int8_draft",
    ),
    "anima_balanced": GenerationPreset(
        name="anima_balanced",
        label="Anima Balanced",
        steps=28,
        cfg=5.0,
        sampler="euler_ancestral",
        scheduler="sgm_uniform",
        renderer_profile="anima_fp16_final",
    ),
    "anima_final": GenerationPreset(
        name="anima_final",
        label="Anima Final",
        steps=36,
        cfg=5.0,
        sampler="euler_ancestral",
        scheduler="sgm_uniform",
        renderer_profile="anima_fp16_final",
    ),
    "anima_lora": GenerationPreset(
        name="anima_lora",
        label="Anima LoRA",
        steps=28,
        cfg=5.0,
        sampler="euler_ancestral",
        scheduler="sgm_uniform",
        renderer_profile="anima_fp16_final",
        lora_stack=("anima_lora",),
    ),
}


def generation_preset_options() -> list[dict[str, Any]]:
    return [
        {
            "name": preset.name,
            "label": preset.label,
            "steps": preset.steps,
            "cfg": preset.cfg,
            "sampler": preset.sampler,
            "scheduler": preset.scheduler,
            "renderer_profile": preset.renderer_profile,
            "lora_stack": list(preset.lora_stack),
        }
        for preset in GENERATION_PRESETS.values()
    ]


def resolution_preset_options() -> list[dict[str, Any]]:
    return [
        {
            "name": preset.name,
            "label": preset.label,
            "width": preset.width,
            "height": preset.height,
            "custom": preset.custom,
        }
        for preset in RESOLUTION_PRESETS.values()
    ]


def sampler_options() -> list[str]:
    return list(SUPPORTED_SAMPLERS)


def scheduler_options() -> list[str]:
    return list(SUPPORTED_SCHEDULERS)


def normalize_anima_cfg(cfg: float, *, allow_low_cfg: bool = False) -> float:
    if allow_low_cfg or cfg >= MIN_EFFECTIVE_ANIMA_CFG:
        return cfg
    return DEFAULT_EFFECTIVE_ANIMA_CFG


def apply_generation_preset(
    plan: GenerationPlan,
    *,
    preset_name: str | None = None,
    resolution_name: str | None = None,
    orientation: str | None = None,
    custom_width: int | None = None,
    custom_height: int | None = None,
    overrides: dict[str, object] | None = None,
    allow_low_cfg: bool = False,
) -> GenerationPlan:
    preset = GENERATION_PRESETS.get((preset_name or "anima_balanced").strip()) or GENERATION_PRESETS["anima_balanced"]
    resolution = RESOLUTION_PRESETS.get((resolution_name or "square_1024").strip()) or RESOLUTION_PRESETS["square_1024"]
    width, height = _resolution_dimensions(
        resolution,
        orientation=orientation,
        custom_width=custom_width,
        custom_height=custom_height,
        default_size=(plan.width, plan.height),
    )
    data: dict[str, object] = {
        "width": width,
        "height": height,
        "steps": preset.steps,
        "cfg": preset.cfg,
        "sampler": preset.sampler,
        "scheduler": preset.scheduler,
        "renderer_profile": preset.renderer_profile,
        "lora_stack": preset.lora_stack,
    }
    data.update({key: value for key, value in (overrides or {}).items() if value not in (None, "")})
    data["cfg"] = normalize_anima_cfg(float(data["cfg"]), allow_low_cfg=allow_low_cfg)
    return replace(plan, **data)


def _resolution_dimensions(
    resolution: ResolutionPreset,
    *,
    orientation: str | None,
    custom_width: int | None,
    custom_height: int | None,
    default_size: tuple[int, int],
) -> tuple[int, int]:
    if resolution.custom:
        width = int(custom_width or default_size[0])
        height = int(custom_height or default_size[1])
    else:
        assert resolution.width is not None
        assert resolution.height is not None
        width = resolution.width
        height = resolution.height
    if (orientation or "").strip().lower() == "landscape":
        width, height = max(width, height), min(width, height)
    return width, height
