from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_danbooru_unified_dataset import (  # noqa: E402
    Taxonomy,
    bucket_tags,
    is_identity_like,
    norm_tag,
    ordered_visual_tags,
    split_tags,
)


DEFAULT_TAXONOMY = Path(
    r"E:\1\danbooru-e621-tag-list-processor-main"
    r"\output\tag_taxonomy\llm_2026-05-22"
    r"\tag_taxonomy_llm_classified_merged.jsonl"
)

SAFE_RATINGS = {"g", "s"}
EXPLICIT_DENY_FOR_SAFE = {
    "anus",
    "areolae",
    "cameltoe",
    "clitoris",
    "cum",
    "cunnilingus",
    "deepthroat",
    "doggystyle",
    "ejaculation",
    "fellatio",
    "fingering",
    "footjob",
    "handjob",
    "masturbation",
    "missionary",
    "nipples",
    "oral",
    "paizuri",
    "penetration",
    "penis",
    "prone bone",
    "pubic hair",
    "pussy",
    "sex",
    "testicles",
    "vagina",
    "vaginal",
}
SAFE_PRECISION_RISK_TAGS = EXPLICIT_DENY_FOR_SAFE | {
    "breasts",
    "large breasts",
    "huge breasts",
    "medium breasts",
    "small breasts",
    "cleavage",
    "navel",
    "stomach",
    "thighs",
}
POSE_TARGET_RISK_TAGS = SAFE_PRECISION_RISK_TAGS | {
    "bare breasts",
    "bottomless",
    "naked",
    "nude",
    "panties",
    "topless",
    "underwear",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rating(row: dict) -> str:
    return str(row.get("rating") or "").strip().lower()[:1]


def tag_set(row: dict) -> set[str]:
    if isinstance(row.get("flat_tags"), list):
        return {norm_tag(tag) for tag in row["flat_tags"] if norm_tag(tag)}
    return set(split_tags(row.get("target_tags") or row.get("tags") or ""))


def sanitize_tags(tags: list[str], rating: str, *, mode: str) -> list[str]:
    deny = set()
    if rating in SAFE_RATINGS:
        deny |= EXPLICIT_DENY_FOR_SAFE
    if mode == "pose_boost":
        deny |= POSE_TARGET_RISK_TAGS
    out = []
    seen = set()
    for tag in tags:
        tag = norm_tag(tag)
        if not tag or tag in seen or tag in deny:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def visual_record_from_existing(row: dict, *, mode: str, source: str) -> dict | None:
    rating = normalize_rating(row)
    tags = [norm_tag(tag) for tag in (row.get("flat_tags") or split_tags(row.get("target_tags") or row.get("tags")))]
    tags = sanitize_tags(tags, rating, mode=mode)
    if len(tags) < 8:
        return None
    groups = row.get("taxonomy_groups") or {}
    if not groups:
        groups = {"other": tags}
    target = ", ".join(tags)
    return {
        "id": row.get("id"),
        "split": f"train_{mode}",
        "rating": rating,
        "score": row.get("score"),
        "image": str(row.get("image") or ""),
        "width": row.get("width"),
        "height": row.get("height"),
        "n_tags": len(tags),
        "tags": target,
        "target_tags": target,
        "flat_tags": tags,
        "taxonomy_groups": groups,
        "core_tags": tags,
        "support_tags": [],
        "identity_tags": row.get("identity_tags") or [],
        "blocked_tags": sorted(set(row.get("blocked_tags") or []) | (set(tag_set(row)) - set(tags))),
        "source_dataset": source,
        "preprocess": f"tagger_precision_pose_v2:{mode}",
    }


def raw_pose_to_visual(row: dict, root: Path, taxonomy: Taxonomy) -> dict | None:
    rating = normalize_rating(row)
    if rating not in SAFE_RATINGS:
        return None
    tags = []
    for tag in split_tags(row.get("tag_string_general")):
        if taxonomy.is_drop(tag) or is_identity_like(tag, taxonomy):
            continue
        tags.append(tag)
    pose_query = norm_tag(row.get("pose_query"))
    if pose_query and pose_query not in tags:
        tags.append(pose_query)
    tags = sanitize_tags(tags, rating, mode="pose_boost")
    groups = bucket_tags(tags, taxonomy)
    tags = ordered_visual_tags(groups, 48)
    if pose_query and pose_query not in tags:
        tags.append(pose_query)
        groups = bucket_tags(tags, taxonomy)
    if len(tags) < 8:
        return None
    image = root / str(row.get("image") or "")
    if not image.exists() or image.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    target = ", ".join(tags)
    return {
        "id": row.get("id"),
        "split": "train_pose_boost_v2",
        "rating": rating,
        "score": row.get("score"),
        "image": str(image),
        "width": row.get("image_width") or row.get("width"),
        "height": row.get("image_height") or row.get("height"),
        "n_tags": len(tags),
        "tags": target,
        "target_tags": target,
        "flat_tags": tags,
        "taxonomy_groups": groups,
        "core_tags": tags,
        "support_tags": [],
        "identity_tags": split_tags(row.get("tag_string_character")) + split_tags(row.get("tag_string_copyright")),
        "blocked_tags": split_tags(row.get("tag_string_artist")) + split_tags(row.get("tag_string_meta")),
        "pose_query": pose_query,
        "source_dataset": "danbooru_pose_alltags_allratings_10",
        "preprocess": "tagger_precision_pose_v2:raw_pose_safe",
    }


def tag_freq(rows: list[dict], source: str) -> dict:
    df = Counter()
    core_df = Counter()
    dropped_df = Counter()
    for row in rows:
        tags = set(row.get("flat_tags") or split_tags(row.get("target_tags")))
        for tag in tags:
            df[tag] += 1
            core_df[tag] += 1
        for tag in set(row.get("blocked_tags") or []):
            dropped_df[norm_tag(tag)] += 1
    return {
        "n_docs": len(rows),
        "df": dict(sorted(df.items())),
        "core_df": dict(sorted(core_df.items())),
        "support_df": {},
        "dropped_df": dict(sorted(dropped_df.items())),
        "preprocess": "tagger_precision_pose_v2",
        "source": source,
    }


def write_root(root: Path, rows: list[dict], source: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "manifest_visual_expand.jsonl", rows)
    (root / "tag_freq_visual_expand.json").write_text(
        json.dumps(tag_freq(rows, source), ensure_ascii=False),
        encoding="utf-8",
    )
    ratings = Counter(normalize_rating(row) for row in rows)
    groups = Counter()
    for row in rows:
        for group, values in (row.get("taxonomy_groups") or {}).items():
            groups[group] += len(values)
    return {
        "root": str(root),
        "rows": len(rows),
        "ratings": dict(sorted(ratings.items())),
        "group_tag_counts": dict(sorted(groups.items())),
    }


def build_eval_manifest(output: Path, safe_rows: list[dict], pose_rows: list[dict], *, seed: int) -> dict:
    rng = random.Random(seed)
    selected = []
    for source, rows, limit in (
        ("safe_precision", safe_rows, 16),
        ("pose_boost", pose_rows, 16),
    ):
        pool = list(rows)
        rng.shuffle(pool)
        for row in pool[:limit]:
            selected.append(
                {
                    "id": row.get("id"),
                    "rating": row.get("rating"),
                    "image": row.get("image"),
                    "tags_general": " ".join(str(tag).replace(" ", "_") for tag in (row.get("flat_tags") or [])),
                    "eval_source": source,
                }
            )
    write_jsonl(output, selected)
    return {"path": str(output), "rows": len(selected)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-root", default=r"D:\Projects\danbooru_unified_balanced_v1")
    parser.add_argument("--e10k-root", default=r"D:\Projects\danbooru_unified_e10k_v1")
    parser.add_argument("--pose-root", default=r"D:\Projects\danbooru_pose_alltags_allratings_10")
    parser.add_argument("--out-root", default=r"D:\Projects")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--safe-limit", type=int, default=30000)
    parser.add_argument("--mixed-limit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=44203)
    parser.add_argument("--report-dir", default=r"reports\tagger_precision_pose_v2")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    taxonomy = Taxonomy(Path(args.taxonomy))
    balanced_root = Path(args.balanced_root)
    e10k_root = Path(args.e10k_root)
    pose_root = Path(args.pose_root)
    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    for root, source in ((balanced_root, "balanced_v1"), (e10k_root, "e10k_v1")):
        manifest = root / "manifest_visual_expand.jsonl"
        for row in read_jsonl(manifest):
            existing_rows.append((row, source))

    safe_pool = []
    mixed_pool = []
    counters = Counter()
    for row, source in existing_rows:
        rating = normalize_rating(row)
        tags = tag_set(row)
        if rating in SAFE_RATINGS and not (tags & SAFE_PRECISION_RISK_TAGS):
            rec = visual_record_from_existing(row, mode="safe_precision", source=source)
            if rec:
                safe_pool.append(rec)
        mixed = visual_record_from_existing(row, mode="mixed_stability", source=source)
        if mixed:
            mixed_pool.append(mixed)
        counters[f"existing_{source}"] += 1

    pose_rows = []
    for row in read_jsonl(pose_root / "manifest.jsonl"):
        rec = raw_pose_to_visual(row, pose_root, taxonomy)
        if rec:
            pose_rows.append(rec)
        counters["pose_raw"] += 1

    rng.shuffle(safe_pool)
    rng.shuffle(mixed_pool)
    rng.shuffle(pose_rows)
    safe_rows = safe_pool[: args.safe_limit]
    mixed_rows = mixed_pool[: args.mixed_limit]

    roots = {
        "safe_precision": out_root / "danbooru_tagger_safe_precision_v2",
        "pose_boost": out_root / "danbooru_tagger_pose_boost_v2",
        "mixed_stability": out_root / "danbooru_tagger_mixed_stability_v2",
    }
    summaries = {
        "safe_precision": write_root(roots["safe_precision"], safe_rows, "balanced_v1+e10k_v1"),
        "pose_boost": write_root(roots["pose_boost"], pose_rows, str(pose_root / "manifest.jsonl")),
        "mixed_stability": write_root(roots["mixed_stability"], mixed_rows, "balanced_v1+e10k_v1"),
    }
    eval_summary = build_eval_manifest(
        Path(r"re-learning\eval\tagger_precision_pose_v2_manifest.jsonl"),
        safe_rows,
        pose_rows,
        seed=args.seed,
    )
    readme = report_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Tagger precision/pose v2 preparation",
                "",
                "Purpose: reduce G/S false explicit anatomy tags before pose/action boosting.",
                "",
                "Training order:",
                "1. safe_precision warmup from danbooru_tagger_safe_precision_v2",
                "2. pose_boost from danbooru_tagger_pose_boost_v2",
                "3. mixed_stability from danbooru_tagger_mixed_stability_v2 plus the previous two roots",
                "",
                "Promotion gate: run v18 on re-learning/eval/tagger_precision_pose_v2_manifest.jsonl and reject if G/S explicit false positives recur.",
            ]
        ),
        encoding="utf-8",
    )
    report = {
        "stage": "tagger_precision_pose_v2_preparation",
        "seed": args.seed,
        "inputs": {
            "balanced_root": str(balanced_root),
            "e10k_root": str(e10k_root),
            "pose_root": str(pose_root),
            "taxonomy": str(Path(args.taxonomy)),
        },
        "counters": dict(counters),
        "outputs": summaries,
        "eval_manifest": eval_summary,
        "next": {
            "cache_script": r"re-learning\09_cache_vision_tagger_precision_pose_v2_4070.ps1",
            "train_script": r"re-learning\10_train_vision_tagger_precision_pose_v2_4070.ps1",
            "eval_script": r"re-learning\11_eval_vision_tagger_precision_pose_v2_4070.ps1",
        },
    }
    (report_dir / "manifest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
