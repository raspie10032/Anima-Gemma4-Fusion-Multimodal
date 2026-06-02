from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gemmanima.core.model_paths import model_path


@dataclass(frozen=True)
class BridgeProfile:
    name: str
    label: str
    checkpoint: Path
    role: str
    notes: str


@dataclass(frozen=True)
class ModelConfig:
    gemma_planner_adapter: Path = model_path(
        "hiddenstage_bridge",
        "hiddenstage-planner-adapter.safetensors",
        r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\adapter_model.safetensors",
    )
    gemma_vision_embedding: Path = model_path(
        "hiddenstage_bridge",
        "hiddenstage-planner-embed-vision.pt",
        r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2\embed_vision.pt",
    )
    anima_diffusion_model: Path = model_path(
        "anima_image_core",
        "anima-base-v1.0.safetensors",
        r"E:\ComfyUI_sage\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors",
    )
    anima_vae: Path = model_path(
        "anima_image_core",
        "qwen_image_vae.safetensors",
        r"E:\ComfyUI_sage\ComfyUI\models\vae\qwen_image_vae.safetensors",
    )
    hiddenstage_bridge: Path = model_path(
        "hiddenstage_bridge",
        "kv_proj_hiddenstage_planner_v2.pt",
        r"E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt",
    )
    hiddenstage_bridge_balanced_pose: Path = model_path(
        "hiddenstage_bridge",
        "kv_proj_balanced_pose_153k_pose10k_a0p35.pt",
        r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\bridge_quality_1p53m_v2_153k_pose10k\bridge\kv_proj_bridge_quality_1p53m_v2_153k_pose10k_from_a0p35.pt",
    )
    hiddenstage_bridge_style_artist: Path = model_path(
        "hiddenstage_bridge",
        "kv_proj_style_artist_v37a_10k.pt",
        r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\text_preservation_blended_v37\bridge\text_preservation_blended_v37a_10k_bridge.pt",
    )
    hiddenstage_bridge_text_exact: Path = model_path(
        "hiddenstage_bridge",
        "kv_proj_text_exact_v27_alpha35.pt",
        r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\cache\text_preservation_blended_v27\bridge\text_preservation_blended_v27_alpha35_bridge.pt",
    )

    def bridge_profiles(self) -> dict[str, BridgeProfile]:
        return {
            "balanced_pose": BridgeProfile(
                name="balanced_pose",
                label="Balanced Pose",
                checkpoint=self.hiddenstage_bridge_balanced_pose,
                role="general image generation, composition, and pose-sensitive prompts",
                notes="Default automatic image bridge for normal prompts; prototype routing checkpoint.",
            ),
            "style_artist": BridgeProfile(
                name="style_artist",
                label="Style",
                checkpoint=self.hiddenstage_bridge_style_artist,
                role="style tags and rare surface-token prompt response",
                notes="Prototype style-specialist bridge; not a text-rendering promotion checkpoint.",
            ),
            "text_exact": BridgeProfile(
                name="text_exact",
                label="Text Exact",
                checkpoint=self.hiddenstage_bridge_text_exact,
                role="signs, labels, logos, captions, and prompts that require readable text",
                notes="Best available fixed-text preservation bridge from the Qwen baseline summaries.",
            ),
            "legacy_mse": BridgeProfile(
                name="legacy_mse",
                label="Legacy MSE",
                checkpoint=self.hiddenstage_bridge,
                role="compatibility bridge and engineering MSE-gated baseline",
                notes="Original HiddenStage bridge retained for compatibility and explicit override.",
            ),
        }


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
