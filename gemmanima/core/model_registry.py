from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gemmanima.core.config import EngineConfig
from gemmanima.core.model_sources import (
    ModelSource,
    ProgressCallback,
    adapter_source,
    download_hf_source,
    hf_source,
)
from gemmanima.modules.tipo_runtime import (
    DEFAULT_TIPO_BASE_MODEL,
    DEFAULT_TIPO_TEXT_LORA,
    DEFAULT_TIPO_VISION_LORA,
    DEFAULT_TIPO_VISION_MMPROJ,
)


COMPONENT_LABELS: dict[str, str] = {
    "gemma_core": "Gemma Core",
    "anima_image_core": "Anima Image Core",
    "vision_tagger": "Vision Tagger",
    "hiddenstage_bridge": "HiddenStage Bridge",
}


@dataclass(frozen=True)
class ModelAsset:
    name: str
    component: str
    role: str
    path: Path
    source: ModelSource
    required: bool = True

    @property
    def component_label(self) -> str:
        return COMPONENT_LABELS[self.component]

    @property
    def exists(self) -> bool:
        return self.path.exists()


class ModelRegistry:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def assets(self) -> tuple[ModelAsset, ...]:
        models = self.config.models
        return (
            ModelAsset(
                "gemma_core.shared_base_gguf",
                "gemma_core",
                "language_core",
                DEFAULT_TIPO_BASE_MODEL,
                hf_source(
                    origin="original_model_page",
                    repo_id="mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF",
                    filename="gemma-4-E2B-it-heretic-ara-custom.Q4_K_M.gguf",
                    license_id="apache-2.0",
                    license_note="License tag reported by the upstream GGUF model page.",
                    note="Original GGUF quantization page for the Gemma Core base.",
                ),
            ),
            ModelAsset(
                "gemma_core.text_lora",
                "gemma_core",
                "chat_adapter",
                DEFAULT_TIPO_TEXT_LORA,
                adapter_source("gemma_core/text-adapter-model-f16.gguf"),
            ),
            ModelAsset(
                "gemma_core.vision_lora",
                "gemma_core",
                "vision_tag_adapter",
                DEFAULT_TIPO_VISION_LORA,
                adapter_source("gemma_core/vision-tagger-adapter-model-f16.gguf"),
            ),
            ModelAsset(
                "gemma_core.vision_mmproj",
                "gemma_core",
                "vision_projector",
                DEFAULT_TIPO_VISION_MMPROJ,
                adapter_source("gemma_core/gemma4-tipo-vision.mmproj-f16.gguf"),
            ),
            ModelAsset(
                "vision_tagger.wd_swinv2_model",
                "vision_tagger",
                "local_danbooru_tagger",
                models.wd_tagger_model,
                hf_source(
                    origin="original_model_page",
                    repo_id="SmilingWolf/wd-swinv2-tagger-v3",
                    filename="model.onnx",
                    license_id="apache-2.0",
                    license_note="The upstream Hugging Face model page reports Apache-2.0.",
                    note="Local ONNX Danbooru tagger used as the default image-tag correlation evaluator.",
                ),
            ),
            ModelAsset(
                "vision_tagger.wd_swinv2_tags",
                "vision_tagger",
                "local_danbooru_tagger_tags",
                models.wd_tagger_tags,
                hf_source(
                    origin="original_model_page",
                    repo_id="SmilingWolf/wd-swinv2-tagger-v3",
                    filename="selected_tags.csv",
                    license_id="apache-2.0",
                    license_note="The upstream Hugging Face model page reports Apache-2.0.",
                    note="Tag vocabulary paired with wd-swinv2-tagger-v3/model.onnx.",
                ),
            ),
            ModelAsset(
                "anima_image_core.diffusion_model",
                "anima_image_core",
                "image_diffusion",
                models.anima_diffusion_model,
                hf_source(
                    origin="original_model_page",
                    repo_id="circlestone-labs/Anima",
                    filename="split_files/diffusion_models/anima-base-v1.0.safetensors",
                    license_id="circlestone-labs-non-commercial-license",
                    license_note="The upstream Anima page also states that the model is a derivative of NVIDIA Cosmos-Predict2-2B-Text2Image and is subject to the NVIDIA Open Model License Agreement where applicable.",
                    note="Original Anima diffusion model page.",
                ),
            ),
            ModelAsset(
                "anima_image_core.vae",
                "anima_image_core",
                "image_vae",
                models.anima_vae,
                hf_source(
                    origin="original_model_page",
                    repo_id="circlestone-labs/Anima",
                    filename="split_files/vae/qwen_image_vae.safetensors",
                    license_id="circlestone-labs-non-commercial-license",
                    license_note="Distributed from the upstream Anima repository; follow the Anima page and any upstream VAE/model notices.",
                    note="Original VAE file distributed with Anima.",
                ),
            ),
            ModelAsset(
                "hiddenstage_bridge.planner_adapter",
                "hiddenstage_bridge",
                "fusion_planner",
                models.gemma_planner_adapter,
                adapter_source("hiddenstage_bridge/hiddenstage-planner-adapter.safetensors"),
            ),
            ModelAsset(
                "hiddenstage_bridge.planner_vision_embedding",
                "hiddenstage_bridge",
                "fusion_planner",
                models.gemma_vision_embedding,
                adapter_source("hiddenstage_bridge/hiddenstage-planner-embed-vision.pt"),
            ),
            ModelAsset(
                "hiddenstage_bridge.bridge_checkpoint",
                "hiddenstage_bridge",
                "fusion_bridge",
                models.hiddenstage_bridge,
                adapter_source("hiddenstage_bridge/kv_proj_hiddenstage_planner_v2.pt"),
            ),
            ModelAsset(
                "hiddenstage_bridge.bridge_balanced_pose",
                "hiddenstage_bridge",
                "fusion_bridge_profile",
                models.hiddenstage_bridge_balanced_pose,
                adapter_source("hiddenstage_bridge/kv_proj_text_exact_v27_alpha35.pt"),
            ),
            ModelAsset(
                "hiddenstage_bridge.bridge_style_artist",
                "hiddenstage_bridge",
                "fusion_bridge_profile",
                models.hiddenstage_bridge_style_artist,
                adapter_source("hiddenstage_bridge/kv_proj_text_exact_v27_alpha35.pt"),
            ),
            ModelAsset(
                "hiddenstage_bridge.bridge_text_exact",
                "hiddenstage_bridge",
                "fusion_bridge_profile",
                models.hiddenstage_bridge_text_exact,
                adapter_source("hiddenstage_bridge/kv_proj_text_exact_v27_alpha35.pt"),
            ),
        )

    def health(self) -> dict[str, dict[str, object]]:
        return {
            asset.name: {
                "component": asset.component,
                "component_label": asset.component_label,
                "role": asset.role,
                "path": str(asset.path),
                "required": asset.required,
                "exists": asset.exists,
                "source": asset.source.to_json_dict(),
            }
            for asset in self.assets()
        }

    def grouped_health(self) -> dict[str, dict[str, object]]:
        flat = self.health()
        grouped: dict[str, dict[str, object]] = {
            component: {"label": label, "assets": {}}
            for component, label in COMPONENT_LABELS.items()
        }
        for name, payload in flat.items():
            grouped[str(payload["component"])]["assets"][name] = payload
        return grouped

    def missing_required(self) -> tuple[ModelAsset, ...]:
        return tuple(asset for asset in self.assets() if asset.required and not asset.exists)

    def download_plan(self) -> dict[str, object]:
        assets = [
            {
                "name": asset.name,
                "component": asset.component,
                "component_label": asset.component_label,
                "role": asset.role,
                "path": str(asset.path),
                "exists": asset.exists,
                "required": asset.required,
                "source": asset.source.to_json_dict(),
            }
            for asset in self.assets()
        ]
        return {
            "status": "planned",
            "model_root_hint": "%LOCALAPPDATA%\\GemmAnima\\models or GEMMANIMA_MODEL_ROOT",
            "assets": assets,
        }

    def ensure_assets(
        self,
        *,
        overwrite: bool = False,
        names: set[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        selected = [asset for asset in self.assets() if names is None or asset.name in names]
        results = []
        total_assets = len(selected)
        for index, asset in enumerate(selected, start=1):
            if on_progress:
                on_progress(
                    {
                        "status": "asset_started",
                        "name": asset.name,
                        "component": asset.component,
                        "role": asset.role,
                        "asset_index": index,
                        "total_assets": total_assets,
                        "path": str(asset.path),
                    }
                )

            def forward_progress(event: dict[str, object], *, asset=asset, index=index) -> None:
                if not on_progress:
                    return
                payload = dict(event)
                payload.update(
                    {
                        "name": asset.name,
                        "component": asset.component,
                        "role": asset.role,
                        "asset_index": index,
                        "total_assets": total_assets,
                    }
                )
                on_progress(payload)

            result = download_hf_source(
                asset.source,
                asset.path,
                overwrite=overwrite,
                on_progress=forward_progress,
            )
            results.append({"name": asset.name, **result})
            if on_progress:
                on_progress(
                    {
                        "status": result["status"],
                        "name": asset.name,
                        "component": asset.component,
                        "role": asset.role,
                        "asset_index": index,
                        "total_assets": total_assets,
                        "path": str(asset.path),
                        "downloaded_bytes": result.get("bytes", 0),
                        "total_bytes": result.get("bytes", 0),
                    }
                )
        return {
            "status": "completed",
            "assets": results,
        }
