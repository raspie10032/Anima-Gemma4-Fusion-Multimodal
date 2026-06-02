import argparse
import json
from pathlib import Path


QUALITY_PREFIX = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"


def compact_tags(*parts: str) -> str:
    seen = set()
    out = []
    for part in parts:
        for raw in str(part or "").replace("_", " ").split():
            tag = raw.strip().strip(",")
            if not tag:
                continue
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return " ".join(out)


def comma_tags(value: str) -> str:
    tags = []
    for raw in str(value or "").split():
        tag = raw.strip()
        if tag:
            tags.append(tag.replace("_", " "))
    return ", ".join(tags)


def build_text(record: dict) -> str:
    pose = str(record.get("pose_query") or "").replace("_", " ").strip()
    character = comma_tags(record.get("tag_string_character", ""))
    copyright_tags = comma_tags(record.get("tag_string_copyright", ""))
    general = comma_tags(record.get("tag_string_general", ""))
    parts = [QUALITY_PREFIX]
    if pose:
        parts.append(f"pose focus, {pose}, clear body pose")
    if character:
        parts.append(character)
    if copyright_tags:
        parts.append(copyright_tags)
    if general:
        parts.append(general)
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--curriculum-out", required=True)
    parser.add_argument("--idx-start", type=int, default=2_000_000)
    parser.add_argument("--weight", type=float, default=3.0)
    parser.add_argument("--skip-missing-images", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = Path(args.manifest)
    out = Path(args.out)
    curriculum_out = Path(args.curriculum_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    curriculum_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_images = 0
    seen_ids = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            image_rel = record.get("image") or ""
            image_path = root / image_rel
            if image_rel and not image_path.exists():
                missing_images += 1
                if args.skip_missing_images:
                    continue
            danbooru_id = record.get("id")
            if danbooru_id in seen_ids:
                continue
            seen_ids.add(danbooru_id)
            idx = args.idx_start + len(rows)
            rows.append(
                {
                    "idx": idx,
                    "id": f"pose_{danbooru_id}",
                    "src": "danbooru_pose_alltags_allratings_10",
                    "danbooru_id": danbooru_id,
                    "rating": record.get("rating"),
                    "pose_query": record.get("pose_query"),
                    "image": str(image_path),
                    "text": build_text(record),
                }
            )

    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    curriculum = {
        "name": "pose_alltags_allratings_10_curriculum",
        "source_manifest": str(manifest),
        "prompt_records": [
            {
                "idx": row["idx"],
                "id": row["id"],
                "pose_query": row["pose_query"],
                "rating": row["rating"],
                "sample_weight": args.weight,
            }
            for row in rows
        ],
    }
    curriculum_out.write_text(json.dumps(curriculum, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "output": str(out),
        "curriculum_output": str(curriculum_out),
        "rows": len(rows),
        "idx_start": args.idx_start,
        "idx_end": args.idx_start + max(0, len(rows) - 1),
        "missing_images": missing_images,
        "skip_missing_images": args.skip_missing_images,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
