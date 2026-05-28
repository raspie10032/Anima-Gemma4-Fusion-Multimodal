from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    gemma_planner_adapter: Path = Path(
        r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\adapter_model.safetensors"
    )
    gemma_vision_embedding: Path = Path(
        r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\embed_vision.pt"
    )
    anima_diffusion_model: Path = Path(
        r"E:\ComfyUI_sage\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors"
    )
    anima_text_encoder: Path = Path(
        r"E:\ComfyUI_sage\ComfyUI\models\text_encoders\qwen_3_06b_base.safetensors"
    )
    anima_vae: Path = Path(r"E:\ComfyUI_sage\ComfyUI\models\vae\qwen_image_vae.safetensors")
    hiddenstage_bridge: Path = Path(r"E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt")


@dataclass(frozen=True)
class HardwareConfig:
    primary_gpu: str = "RTX 4070 Ti SUPER"
    secondary_gpu: str | None = None


@dataclass(frozen=True)
class RendererProfile:
    name: str
    precision: str
    width: int
    height: int
    steps: int
    cfg: float


@dataclass(frozen=True)
class EngineConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    renderer_profiles: dict[str, RendererProfile] = field(
        default_factory=lambda: {
            "anima_fp16_final": RendererProfile(
                name="anima_fp16_final",
                precision="fp16",
                width=1024,
                height=1024,
                steps=28,
                cfg=4.0,
            ),
            "anima_int8_draft": RendererProfile(
                name="anima_int8_draft",
                precision="int8",
                width=768,
                height=768,
                steps=16,
                cfg=3.5,
            ),
        }
    )

    def profile(self, name: str) -> RendererProfile:
        try:
            return self.renderer_profiles[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.renderer_profiles))
            raise ValueError(f"unknown renderer profile {name!r}; known profiles: {known}") from exc
