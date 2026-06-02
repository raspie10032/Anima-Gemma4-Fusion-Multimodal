from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemmanima.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the repo-native in-process Anima renderer.")
    parser.add_argument(
        "--prompt",
        default="Draw Nahida from Genshin Impact as a bright forest anime illustration, gentle expression.",
    )
    parser.add_argument("--image-root", default="runs/images")
    parser.add_argument("--manifest-root", default="runs/manifests")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=19375672098)
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["GEMMA_EMBED_ON_GPU"] = "1"

    return cli_main(
        [
            "run",
            args.prompt,
            "--renderer",
            "in-process",
            "--image-root",
            args.image_root,
            "--manifest-root",
            args.manifest_root,
            "--steps",
            str(args.steps),
            "--size",
            str(args.size),
            "--cfg",
            str(args.cfg),
            "--seed",
            str(args.seed),
            "--json",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
