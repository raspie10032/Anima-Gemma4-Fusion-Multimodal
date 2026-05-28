from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemmanima.core.config import EngineConfig, ModelConfig
from gemmanima.modules.bridge_runtime import TrainedBridgeRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test trained HiddenStage bridge forward pass.")
    parser.add_argument("--checkpoint", default=r"E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt")
    parser.add_argument("--anima-dm", default=r"E:\ComfyUI_sage\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors")
    parser.add_argument("--comfy-root", default=r"E:\ComfyUI_sage\ComfyUI")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--t5-len", type=int, default=32)
    args = parser.parse_args()

    model_config = ModelConfig(
        hiddenstage_bridge=args.checkpoint,
        anima_diffusion_model=args.anima_dm,
    )
    runtime = TrainedBridgeRuntime(
        config=EngineConfig(models=model_config),
        comfy_root=args.comfy_root,
        device=args.device,
    )
    payload = runtime.smoke(seq_len=args.seq_len, t5_len=args.t5_len).to_json_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["finite"] and payload["output_shape"] == [1, args.t5_len, 1024] else 1


if __name__ == "__main__":
    raise SystemExit(main())
