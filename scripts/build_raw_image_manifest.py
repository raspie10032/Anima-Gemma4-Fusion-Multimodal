from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
QUALITY_PREFIX = "masterpiece, best quality, score_7, safe"
SCHEMA = "hiddenstage_raw_image_planner_anima_v2"
IMAGE_INTENT_OPEN = "<image_intent>"
IMAGE_INTENT_CLOSE = "</image_intent>"


def stable_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:20]


def split_tags(text: str, *, max_tags: int = 96) -> list[str]:
    chunks = re.split(r"[,;\n\r\t|]+", text)
    tags: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        tag = " ".join(chunk.replace("_", " ").strip().split()).strip(" ,.;:")
        low = tag.lower()
        if not low or len(low) < 2:
            continue
        if low in seen:
            continue
        seen.add(low)
        tags.append(low)
        if len(tags) >= max_tags:
            break
    return tags


def prompt_from_json_text(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        for key in ("prompt", "Prompt", "description", "Description", "positive_prompt", "Positive prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        comment = payload.get("Comment")
        if isinstance(comment, str):
            return prompt_from_json_text(comment)
    return None


def prompt_from_sidecar(path: Path) -> str | None:
    for suffix in (".txt", ".caption", ".prompt"):
        sidecar = path.with_suffix(suffix)
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
    json_sidecar = path.with_suffix(".json")
    if json_sidecar.exists():
        text = json_sidecar.read_text(encoding="utf-8", errors="ignore")
        prompt = prompt_from_json_text(text)
        if prompt:
            return prompt
    return None


def prompt_from_image_info(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            info = dict(image.info)
    except Exception:
        return None
    for key in ("prompt", "Prompt", "parameters", "Description", "description", "Comment", "comment"):
        value = info.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if key.lower() == "comment":
            parsed = prompt_from_json_text(value)
            if parsed:
                return parsed
        text = value.strip()
        if "Negative prompt:" in text:
            text = text.split("Negative prompt:", 1)[0].strip()
        if text:
            return text
    return None


def fallback_prompt(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root)
    parts = [part for part in rel.parts[:-1] if part and not re.fullmatch(r"new folder( \(\d+\))?", part, re.I)]
    stem = re.sub(r"[_\-]+", " ", path.stem)
    stem = re.sub(r"\b[0-9a-f]{8,}\b", " ", stem, flags=re.I)
    raw = ", ".join([*parts[-4:], stem])
    tags = split_tags(raw, max_tags=48)
    if not tags:
        tags = ["anime style", "image reference"]
    return ", ".join(tags)


def build_prompt(path: Path, data_root: Path) -> tuple[str, str]:
    for source_name, getter in (
        ("sidecar", lambda: prompt_from_sidecar(path)),
        ("image_metadata", lambda: prompt_from_image_info(path)),
    ):
        prompt = getter()
        if prompt:
            return prompt, source_name
    return fallback_prompt(path, data_root), "path_fallback"


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_rows(data_roots: list[Path], output: Path, embed_root: Path, limit: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    embed_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "stage": "raw_image_manifest",
        "data_roots": [str(root) for root in data_roots],
        "output": str(output),
        "embed_root": str(embed_root),
        "read": 0,
        "written": 0,
        "prompt_sources": {},
        "skipped_bad_image": 0,
    }
    with output.open("w", encoding="utf-8") as handle:
        for data_root in data_roots:
            for image_path in iter_images(data_root):
                stats["read"] += 1
                width, height = image_size(image_path)
                if not width or not height:
                    stats["skipped_bad_image"] += 1
                    continue
                sid = stable_id(image_path)
                prompt, prompt_source = build_prompt(image_path, data_root)
                stats["prompt_sources"][prompt_source] = stats["prompt_sources"].get(prompt_source, 0) + 1
                tags = split_tags(prompt)
                visible_prompt = prompt
                if not visible_prompt.lower().startswith("masterpiece"):
                    visible_prompt = f"{QUALITY_PREFIX}, {visible_prompt}"
                row = {
                    "schema": SCHEMA,
                    "id": sid,
                    "src_root": str(data_root),
                    "src_manifest": "raw_recursive_image_scan",
                    "action": "generate",
                    "input_modalities": ["image", "text"],
                    "chat_context": [],
                    "user_request": "Create an Anima image generation prompt from this reference image.",
                    "assistant_response": f"{IMAGE_INTENT_OPEN}\n{visible_prompt}\n{IMAGE_INTENT_CLOSE}",
                    "visible_prompt": visible_prompt,
                    "teacher_prompt": visible_prompt,
                    "hidden_span_policy": "image_intent_sentinel_span",
                    "target_crossattn_emb": None,
                    "image": str(image_path),
                    "image_rel": str(image_path.relative_to(data_root)).replace("\\", "/"),
                    "image_embed_pre": str(embed_root / f"{sid}.pt"),
                    "rating": "unknown",
                    "width": width,
                    "height": height,
                    "score": None,
                    "tags": tags,
                    "target_tags": ", ".join(tags),
                    "prompt_source": prompt_source,
                    "negative_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written"] += 1
                if limit and stats["written"] >= limit:
                    return stats
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build planner-style rows from arbitrary recursive image folders.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embed-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stats", default=None)
    args = parser.parse_args()

    stats = build_rows([Path(p) for p in args.data], Path(args.out), Path(args.embed_root), args.limit)
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)
    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
