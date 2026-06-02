from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GenerationCompareReport:
    prompt: str
    seed: int
    teacher_image: Path
    student_image: Path
    student_checkpoint: Path
    conditioning_mse: float | None = None
    image_metrics: dict[str, Any] | None = None
    text_roi: dict[str, Any] | None = None
    text_roi_metrics: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "prompt": self.prompt,
            "seed": self.seed,
            "images": {
                "teacher": str(self.teacher_image),
                "student": str(self.student_image),
            },
            "student_checkpoint": str(self.student_checkpoint),
            "conditioning": {
                "mse": self.conditioning_mse,
            },
            "image_metrics": self.image_metrics
            if self.image_metrics is not None
            else compare_images_if_available(self.teacher_image, self.student_image),
        }
        if self.text_roi is not None or self.text_roi_metrics is not None:
            payload["text_roi_metrics"] = (
                self.text_roi_metrics
                if self.text_roi_metrics is not None
                else compare_text_roi_if_available(self.teacher_image, self.student_image, roi=self.text_roi or {})
            )
        return payload


def compare_images_if_available(reference: str | Path, candidate: str | Path) -> dict[str, Any] | None:
    ref_path = Path(reference)
    cand_path = Path(candidate)
    if not ref_path.exists() or not cand_path.exists():
        return None
    ref = _load_rgb(ref_path)
    cand = _load_rgb(cand_path)
    if ref.shape != cand.shape:
        return {
            "reference": str(ref_path),
            "candidate": str(cand_path),
            "shape_mismatch": [list(ref.shape), list(cand.shape)],
        }
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    return {
        "reference": str(ref_path),
        "candidate": str(cand_path),
        "shape": list(ref.shape),
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
    }


def compare_text_roi_if_available(
    reference: str | Path,
    candidate: str | Path,
    *,
    roi: dict[str, Any],
) -> dict[str, Any] | None:
    ref_path = Path(reference)
    cand_path = Path(candidate)
    if not ref_path.exists() or not cand_path.exists():
        return None
    ref = _load_rgb(ref_path)
    cand = _load_rgb(cand_path)
    if ref.shape != cand.shape:
        return {
            "reference": str(ref_path),
            "candidate": str(cand_path),
            "shape_mismatch": [list(ref.shape), list(cand.shape)],
            "roi": roi,
        }
    box = _normalize_roi_box(roi.get("box"), width=ref.shape[1], height=ref.shape[0])
    ref_crop = ref[box[1] : box[3], box[0] : box[2], :]
    cand_crop = cand[box[1] : box[3], box[0] : box[2], :]
    diff = ref_crop - cand_crop
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    return {
        "reference": str(ref_path),
        "candidate": str(cand_path),
        "roi": {
            "name": roi.get("name") or "text_roi",
            "box": box,
        },
        "shape": list(ref_crop.shape),
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
    }


def write_compare_report(report: GenerationCompareReport, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _normalize_roi_box(box: Any, *, width: int, height: int) -> list[int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("roi box must be a four-item [left, top, right, bottom] list")
    left, top, right, bottom = [int(value) for value in box]
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        raise ValueError("roi box must have positive width and height after clipping")
    return [left, top, right, bottom]
