from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gemmanima.core.config import EngineConfig


DEFAULT_GENERAL_THRESHOLD = 0.25
DEFAULT_CHARACTER_THRESHOLD = 0.35
DEFAULT_MAX_TAGS = 80


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


@dataclass(frozen=True)
class WdTaggerConfig:
    model: Path = field(default_factory=lambda: _env_path("GEMMANIMA_WD_TAGGER_MODEL", EngineConfig().models.wd_tagger_model))
    tags: Path = field(default_factory=lambda: _env_path("GEMMANIMA_WD_TAGGER_TAGS", EngineConfig().models.wd_tagger_tags))
    general_threshold: float = field(default_factory=lambda: _env_float("GEMMANIMA_WD_TAGGER_GENERAL_THRESHOLD", DEFAULT_GENERAL_THRESHOLD))
    character_threshold: float = field(default_factory=lambda: _env_float("GEMMANIMA_WD_TAGGER_CHARACTER_THRESHOLD", DEFAULT_CHARACTER_THRESHOLD))
    max_tags: int = DEFAULT_MAX_TAGS
    use_bgr: bool = True


@dataclass(frozen=True)
class WdTag:
    name: str
    score: float
    category: int


class WdTagger:
    def __init__(self, config: WdTaggerConfig | None = None) -> None:
        self.config = config or WdTaggerConfig()
        self._session: Any | None = None
        self._input_name = ""
        self._input_size = 448
        self._tags: list[dict[str, str]] = []

    def _load(self) -> None:
        if self._session is not None:
            return
        missing = [str(path) for path in (self.config.model, self.config.tags) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("; ".join(f"missing {path}" for path in missing))
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self.config.model), providers=["CPUExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) >= 3 and isinstance(shape[1], int):
            self._input_size = int(shape[1])
        with Path(self.config.tags).open("r", encoding="utf-8", newline="") as handle:
            self._tags = list(csv.DictReader(handle))

    def tag_image(self, image_path: str | Path) -> list[WdTag]:
        self._load()
        assert self._session is not None
        image = _preprocess_image(Path(image_path), self._input_size, use_bgr=self.config.use_bgr)
        scores = self._session.run(None, {self._input_name: image})[0][0]
        tags: list[WdTag] = []
        for row, score_value in zip(self._tags, scores):
            category = int(row.get("category") or 0)
            if category == 9:
                continue
            threshold = self.config.character_threshold if category == 4 else self.config.general_threshold
            score = float(score_value)
            if score < threshold:
                continue
            name = str(row.get("name") or "").replace("_", " ").strip()
            if name:
                tags.append(WdTag(name=name, score=score, category=category))
        tags.sort(key=lambda item: item.score, reverse=True)
        return tags[: self.config.max_tags]


_DEFAULT_TAGGER: WdTagger | None = None


def get_wd_tagger(config: WdTaggerConfig | None = None) -> WdTagger:
    global _DEFAULT_TAGGER
    if config is not None:
        return WdTagger(config)
    if _DEFAULT_TAGGER is None:
        _DEFAULT_TAGGER = WdTagger()
    return _DEFAULT_TAGGER


def run_wd_vision_tag(
    *,
    image_path: str | Path,
    config: WdTaggerConfig | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    cfg = config or WdTaggerConfig()
    try:
        tagger = get_wd_tagger(cfg if config is not None else None)
        tags = tagger.tag_image(image_path)
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "tags": "",
            "raw": "",
            "seconds": round(time.perf_counter() - start, 3),
            "model": str(cfg.model),
            "tagger": "wd-swinv2-tagger-v3",
        }
    tag_text = ", ".join(tag.name for tag in tags)
    raw = ", ".join(f"{tag.name}:{tag.score:.4f}" for tag in tags)
    return {
        "status": "completed",
        "error": "",
        "tags": tag_text,
        "raw": raw,
        "seconds": round(time.perf_counter() - start, 3),
        "model": str(cfg.model),
        "tagger": "wd-swinv2-tagger-v3",
    }


def wd_tagger_health(config: WdTaggerConfig | None = None) -> dict[str, Any]:
    cfg = config or WdTaggerConfig()
    assets = {
        "model": cfg.model,
        "tags": cfg.tags,
    }
    issues = [
        {
            "code": f"missing_wd_tagger_{name}",
            "scope": "wd_tagger",
            "asset": name,
            "path": str(path),
            "severity": "error",
            "message_ko": "WD 태거 파일을 찾을 수 없습니다.",
            "message_en": "WD tagger asset is missing.",
        }
        for name, path in assets.items()
        if not Path(path).is_file()
    ]
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        onnxruntime_available = True
    except Exception as exc:
        providers = []
        onnxruntime_available = False
        issues.append(
            {
                "code": "missing_onnxruntime",
                "scope": "wd_tagger",
                "asset": "onnxruntime",
                "path": "onnxruntime",
                "severity": "error",
                "message_ko": "ONNX Runtime을 가져올 수 없습니다.",
                "message_en": f"ONNX Runtime is not importable: {exc}",
            }
        )
    return {
        "ready": not issues,
        "backend": "onnxruntime",
        "tagger": "wd-swinv2-tagger-v3",
        "onnxruntime_available": onnxruntime_available,
        "providers": providers,
        "assets": {
            name: {"path": str(path), "exists": Path(path).is_file()}
            for name, path in assets.items()
        },
        "issues": issues,
    }


def _preprocess_image(path: Path, size: int, *, use_bgr: bool) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    rgb = background.convert("RGB")
    width, height = rgb.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(rgb, ((side - width) // 2, (side - height) // 2))
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32)
    if use_bgr:
        array = array[:, :, ::-1]
    return array[None, :, :, :]
