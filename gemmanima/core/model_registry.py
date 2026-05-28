from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gemmanima.core.config import EngineConfig


@dataclass(frozen=True)
class ModelAsset:
    name: str
    role: str
    path: Path
    required: bool = True

    @property
    def exists(self) -> bool:
        return self.path.exists()


class ModelRegistry:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def assets(self) -> tuple[ModelAsset, ...]:
        models = self.config.models
        return (
            ModelAsset("gemma_planner_adapter", "planner", models.gemma_planner_adapter),
            ModelAsset("gemma_vision_embedding", "planner", models.gemma_vision_embedding),
            ModelAsset("anima_diffusion_model", "renderer", models.anima_diffusion_model),
            ModelAsset("anima_text_encoder", "teacher_or_baseline", models.anima_text_encoder),
            ModelAsset("anima_vae", "renderer", models.anima_vae),
            ModelAsset("hiddenstage_bridge", "hiddenstage_exit", models.hiddenstage_bridge),
        )

    def health(self) -> dict[str, dict[str, object]]:
        return {
            asset.name: {
                "role": asset.role,
                "path": str(asset.path),
                "required": asset.required,
                "exists": asset.exists,
            }
            for asset in self.assets()
        }

    def missing_required(self) -> tuple[ModelAsset, ...]:
        return tuple(asset for asset in self.assets() if asset.required and not asset.exists)
