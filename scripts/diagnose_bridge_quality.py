from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemmanima.training.bridge_quality_diagnosis import (
    DEFAULT_TEXT_DELTA_ADAPTER,
    BridgeQualityDiagnosisConfig,
    default_output_root,
    run_diagnosis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen baseline vs Gemma bridge image-quality diagnosis.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--adapter", default=str(DEFAULT_TEXT_DELTA_ADAPTER))
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--sampler", default="euler_ancestral")
    parser.add_argument("--scheduler", default="sgm_uniform")
    parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    report = run_diagnosis(
        BridgeQualityDiagnosisConfig(
            prompt=args.prompt,
            output_root=Path(args.output_root),
            adapter=Path(args.adapter),
            gpu_index=args.gpu_index,
            seed=args.seed,
            size=args.size,
            steps=args.steps,
            cfg=args.cfg,
            sampler=args.sampler,
            scheduler=args.scheduler,
            unet_dtype=args.unet_dtype,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
