from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from gemmanima.core.config import EngineConfig, ModelConfig
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter
from gemmanima.modules.local_worker_anima_renderer import WORKER_RESULT_PREFIX


def main(*, hard_exit_on_success: bool = False) -> int:
    try:
        payload = json.load(sys.stdin)
        result = render_payload(payload)
        print(WORKER_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
        if hard_exit_on_success:
            sys.stdout.flush()
            os._exit(0)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=20),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


def render_payload(payload: dict[str, Any]) -> dict[str, object]:
    plan = GenerationPlan.from_dict(dict(payload["plan"]))
    conditioning = ConditioningBundle.from_dict(dict(payload["conditioning"]))
    output_root = Path(str(payload["output_root"]))
    renderer = InProcessAnimaRendererAdapter(
        output_root=output_root,
        config=_config_from_payload(payload.get("config")),
        image_id_factory=lambda: str(payload.get("image_id") or ""),
        unet_dtype=str(payload.get("unet_dtype") or "fp8_e4m3fn_fast"),
        tiled_vae=bool(payload.get("tiled_vae", True)),
        comfy_args=tuple(str(item) for item in payload.get("comfy_args") or ()),
    )
    result = renderer.generate(plan, conditioning)
    return {
        "image_id": result.image_id,
        "output_path": str(result.output_path),
        "seed": result.seed,
        "warnings": list(result.warnings),
    }


def _config_from_payload(data: object) -> EngineConfig:
    config = EngineConfig()
    if not isinstance(data, dict):
        return config
    models = ModelConfig(
        gemma_planner_adapter=config.models.gemma_planner_adapter,
        gemma_vision_embedding=config.models.gemma_vision_embedding,
        anima_diffusion_model=Path(str(data.get("anima_diffusion_model") or config.models.anima_diffusion_model)),
        anima_vae=Path(str(data.get("anima_vae") or config.models.anima_vae)),
        hiddenstage_bridge=Path(str(data.get("hiddenstage_bridge") or config.models.hiddenstage_bridge)),
    )
    return EngineConfig(models=models, hardware=config.hardware, renderer_profiles=config.renderer_profiles)


if __name__ == "__main__":
    raise SystemExit(main())
