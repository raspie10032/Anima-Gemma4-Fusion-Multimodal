from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def compare_images(reference: Path, candidate: Path) -> dict[str, object]:
    ref = load_rgb(reference)
    cand = load_rgb(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"image shapes differ: {ref.shape} != {cand.shape}")
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "shape": list(ref.shape),
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two generated RGB images with simple pixel metrics.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = compare_images(Path(args.reference), Path(args.candidate))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
