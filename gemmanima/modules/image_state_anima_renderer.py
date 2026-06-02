from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import torch

from gemmanima.core.config import EngineConfig
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.rendering.anima_sampler import AnimaSamplerRuntime, SamplerRequest, build_comfy_sampler, load_anima_model_vae
from gemmanima.rendering.comfy_bootstrap import bootstrap_comfy
from gemmanima.rendering.image_state_engine import (
    FusionMode,
    ImageStateConditioningConfig,
    ImageStateConditioningEngine,
)


class ImageStateAnimaRendererAdapter(AnimaRendererAdapter):
    dry_run = False

    def __init__(
        self,
        output_root: str | Path = "runs/images",
        *,
        engine: ImageStateConditioningEngine | None = None,
        sampler_runtime: AnimaSamplerRuntime | None = None,
        config: EngineConfig | None = None,
        checkpoint: str | Path = r"runs\cache\image_state_conditioning_v2_full\bridge\image_state_conditioning_v2_full_image_translator.pt",
        unet_dtype: str = "fp8_e4m3fn_fast",
    ) -> None:
        super().__init__(output_root)
        self.config = config or EngineConfig()
        self.engine = engine or ImageStateConditioningEngine(
            ImageStateConditioningConfig(checkpoint=Path(checkpoint), device="cuda", dtype="bfloat16")
        )
        self.sampler_runtime = sampler_runtime
        self.unet_dtype = unet_dtype

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        conditioning.validate()
        plan.validate()
        metadata = conditioning.metadata
        record = _record_from_metadata(metadata)
        text_state = _optional_tensor_from_metadata(metadata, "text_state")
        mode = str(metadata.get("fusion_mode") or "image_only")
        return self.generate_from_record(plan, record, mode=_as_fusion_mode(mode), text_state=text_state)

    def generate_from_record(
        self,
        plan: GenerationPlan,
        record: dict,
        *,
        mode: FusionMode = "image_only",
        text_state: torch.Tensor | None = None,
    ) -> RenderResult:
        plan.validate()
        self._ensure_runtime(mode)
        assert self.sampler_runtime is not None
        payload = self.engine.condition_from_record(
            record,
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            mode=mode,
            text_state=text_state,
        )
        seed = plan.seed if plan.seed is not None else self._stable_seed(plan.prompt)
        image_id = uuid4().hex
        output_path = self.output_root / f"{image_id}.png"
        self.sampler_runtime.sample_to_file(
            SamplerRequest(
                positive=payload.positive,
                negative=payload.negative,
                output_path=output_path,
                seed=seed,
                width=plan.width,
                height=plan.height,
                steps=plan.steps,
                cfg=plan.cfg,
                sampler=plan.sampler,
                scheduler=plan.scheduler,
            )
        )
        warnings = []
        if mode == "conditioning_fusion":
            warnings.append("conditioning_fusion passes precomputed cross-attn and does not invoke llm_adapter in Comfy")
        return RenderResult(image_id=image_id, output_path=output_path, seed=seed, warnings=tuple(warnings))

    def _ensure_runtime(self, mode: FusionMode) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("GEMMA_EMBED_ON_GPU", "1")
        if self.sampler_runtime is not None:
            return
        bootstrap_comfy()
        model, vae = load_anima_model_vae(
            diffusion_model_path=self.config.models.anima_diffusion_model,
            vae_path=self.config.models.anima_vae,
            unet_dtype=self.unet_dtype,
        )
        if mode in {"image_only", "hidden_fusion"}:
            self.engine.attach_to_model(model, mode=mode)
        self.sampler_runtime = AnimaSamplerRuntime(model=model, vae=vae, sampler=build_comfy_sampler())


def _record_from_metadata(metadata: dict) -> dict:
    if "image_state_record" in metadata:
        path = Path(metadata["image_state_record"])
        return json.loads(path.read_text(encoding="utf-8"))
    if "image_embed_pre" in metadata:
        return {
            "idx": metadata.get("source_idx"),
            "source_id": metadata.get("source_id"),
            "image": metadata.get("source_image"),
            "image_embed_pre": metadata["image_embed_pre"],
            "text": metadata.get("text") or metadata.get("prompt"),
            "visible_prompt": metadata.get("visible_prompt"),
            "teacher_prompt": metadata.get("teacher_prompt"),
        }
    raise ValueError("image-state renderer requires metadata.image_state_record or metadata.image_embed_pre")


def _optional_tensor_from_metadata(metadata: dict, key: str) -> torch.Tensor | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    return torch.load(value, map_location="cpu", weights_only=False).float()


def _as_fusion_mode(value: str) -> FusionMode:
    if value in {"image_only", "hidden_fusion", "conditioning_fusion"}:
        return value  # type: ignore[return-value]
    raise ValueError(f"unsupported image-state fusion mode: {value}")
