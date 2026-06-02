from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from gemmanima.core.config import EngineConfig
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.rendering.anima_sampler import AnimaSamplerRuntime, SamplerRequest, build_comfy_sampler, load_anima_model_vae
from gemmanima.rendering.anima_adapter import adapter_dtype_for_unet, attach_hiddenstage_adapter
from gemmanima.rendering.comfy_bootstrap import bootstrap_comfy
from gemmanima.rendering.gemma_hidden import GemmaHiddenProvider, GemmaTextRuntime
from gemmanima.rendering.t5_tokenizer import T5TokenizerProvider, build_t5_tokenizer_provider


NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, text"


class InProcessAnimaRendererAdapter(AnimaRendererAdapter):
    dry_run = False

    def __init__(
        self,
        output_root: str | Path = "runs/images",
        *,
        config: EngineConfig | None = None,
        hidden_provider: GemmaHiddenProvider | None = None,
        t5_provider: T5TokenizerProvider | None = None,
        sampler_runtime: AnimaSamplerRuntime | None = None,
        image_id_factory: Callable[[], str] | None = None,
        unet_dtype: str = "fp8_e4m3fn_fast",
        tiled_vae: bool = True,
        comfy_args: tuple[str, ...] = (),
    ) -> None:
        super().__init__(output_root)
        self.config = config or EngineConfig()
        self.hidden_provider = hidden_provider
        self.t5_provider = t5_provider
        self.sampler_runtime = sampler_runtime
        self.image_id_factory = image_id_factory or (lambda: uuid4().hex)
        self.unet_dtype = unet_dtype
        self.tiled_vae = tiled_vae
        self.comfy_args = comfy_args

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        conditioning.validate()
        plan.validate()
        self._ensure_runtime()
        assert self.hidden_provider is not None
        assert self.t5_provider is not None
        assert self.sampler_runtime is not None

        seed = plan.seed if plan.seed is not None else self._stable_seed(plan.prompt)
        image_id = self.image_id_factory()
        output_path = self.output_root / f"{image_id}.png"
        source_text = str(conditioning.metadata.get("hiddenstage_source_text") or plan.prompt)
        positive = self._conditioning(source_text, plan.prompt)
        negative = self._conditioning(NEGATIVE_PROMPT, NEGATIVE_PROMPT)
        self.sampler_runtime.sample_to_file(
            SamplerRequest(
                positive=positive,
                negative=negative,
                output_path=output_path,
                seed=seed,
                width=plan.width,
                height=plan.height,
                steps=plan.steps,
                cfg=plan.cfg,
                sampler=plan.sampler,
                scheduler=plan.scheduler,
                tiled_vae=self.tiled_vae,
            )
        )
        return RenderResult(image_id=image_id, output_path=output_path, seed=seed)

    def _conditioning(self, source_text: str, prompt: str) -> list[list[object]]:
        assert self.hidden_provider is not None
        assert self.t5_provider is not None
        hidden = self.hidden_provider.encode_image_intent(source_text, prompt)
        ids, weights = self.t5_provider.encode_ids_weights(prompt)
        return [[hidden, {"t5xxl_ids": ids, "t5xxl_weights": weights}]]

    def _ensure_runtime(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("GEMMA_EMBED_ON_GPU", "1")
        if self.hidden_provider is None or self.t5_provider is None or self.sampler_runtime is None:
            bootstrap_comfy(comfy_args=self.comfy_args)
        if self.hidden_provider is None:
            self.hidden_provider = GemmaHiddenProvider(GemmaTextRuntime())
        if self.t5_provider is None:
            self.t5_provider = build_t5_tokenizer_provider()
        if self.sampler_runtime is None:
            model, vae = load_anima_model_vae(
                diffusion_model_path=self.config.models.anima_diffusion_model,
                vae_path=self.config.models.anima_vae,
                unet_dtype=self.unet_dtype,
            )
            attach_hiddenstage_adapter(
                model,
                diffusion_model_path=str(self.config.models.anima_diffusion_model),
                checkpoint=self.config.models.hiddenstage_bridge,
                adapter_dtype=adapter_dtype_for_unet(model, self.unet_dtype),
            )
            self.sampler_runtime = AnimaSamplerRuntime(model=model, vae=vae, sampler=build_comfy_sampler())
