from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_tagger_precision_pose_v2 import (
    EXPLICIT_DENY_FOR_SAFE,
    SAFE_RATINGS,
    normalize_rating,
    split_tags,
    tag_freq,
)


DEFAULT_SOURCE_ROOT = Path(r"D:\Projects\danbooru_tagger_mixed_stability_v2")
DEFAULT_OUTPUT_ROOT = Path(r"D:\Projects\danbooru_tagger_mixed_pose_front_v2")
DEFAULT_REPORT_DIR = Path(r"reports\tagger_mixed_pose_front_v2")
DEFAULT_SAFETY_EVAL = Path(r"re-learning\eval\tagger_mixed_pose_front_v2_safety_manifest.jsonl")


def norm_tag(tag: object) -> str:
    return str(tag or "").strip().replace("_", " ").lower()


def unique_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        tag = norm_tag(tag)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def row_tags(row: dict) -> list[str]:
    if isinstance(row.get("flat_tags"), list):
        return unique_tags(list(row["flat_tags"]))
    return unique_tags(split_tags(row.get("target_tags") or row.get("tags") or ""))


def filtered_groups(groups: dict, allowed: set[str]) -> dict:
    out: dict[str, list[str]] = {}
    for group, values in (groups or {}).items():
        kept = [tag for tag in unique_tags(list(values or [])) if tag in allowed]
        if kept:
            out[str(group)] = kept
    return out


def prepare_row(row: dict, *, subject_lead: int = 8) -> tuple[dict, list[str]]:
    rating = normalize_rating(row)
    original = row_tags(row)
    groups = row.get("taxonomy_groups") or {}
    deny = EXPLICIT_DENY_FOR_SAFE if rating in SAFE_RATINGS else set()

    removed: list[str] = []
    sanitized: list[str] = []
    for tag in original:
        if tag in deny:
            removed.append(tag)
            continue
        sanitized.append(tag)

    allowed = set(sanitized)
    groups = filtered_groups(groups, allowed)
    subject = [tag for tag in groups.get("subject", []) if tag in allowed][:subject_lead]
    pose_action = [tag for tag in groups.get("pose_action", []) if tag in allowed]
    front = unique_tags(subject + pose_action)
    rest = [tag for tag in sanitized if tag not in set(front)]
    ordered = unique_tags(front + rest)

    prepared = dict(row)
    target = ", ".join(ordered)
    prepared["flat_tags"] = ordered
    prepared["tags"] = target
    prepared["target_tags"] = target
    prepared["core_tags"] = ordered
    prepared["n_tags"] = len(ordered)
    prepared["taxonomy_groups"] = filtered_groups(groups, set(ordered))
    blocked = unique_tags(list(row.get("blocked_tags") or []) + removed)
    prepared["blocked_tags"] = blocked
    prepared["preprocess"] = "tagger_mixed_pose_front_v2"
    return prepared, removed


def first_position_stats(rows: list[dict], group_name: str) -> dict:
    positions: list[int] = []
    for row in rows:
        tags = row_tags(row)
        wanted = set(row.get("taxonomy_groups", {}).get(group_name, []) or [])
        row_positions = [index + 1 for index, tag in enumerate(tags) if tag in wanted]
        if row_positions:
            positions.append(min(row_positions))
    if not positions:
        return {"rows": 0, "min": None, "median": None, "p75": None, "max": None, "le10": 0, "le15": 0, "le25": 0}
    ordered = sorted(positions)
    p75_index = min(len(ordered) - 1, int(len(ordered) * 0.75))
    return {
        "rows": len(positions),
        "min": ordered[0],
        "median": median(ordered),
        "p75": ordered[p75_index],
        "max": ordered[-1],
        "le10": sum(1 for pos in ordered if pos <= 10),
        "le15": sum(1 for pos in ordered if pos <= 15),
        "le25": sum(1 for pos in ordered if pos <= 25),
    }


def explicit_tags_in_safe_rows(rows: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    for row in rows:
        if normalize_rating(row) not in SAFE_RATINGS:
            continue
        for tag in set(row_tags(row)) & EXPLICIT_DENY_FOR_SAFE:
            counts[tag] += 1
    return dict(sorted(counts.items()))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_cache_rows(source_root: Path) -> dict[object, dict]:
    cache_manifest = source_root / "img_embeds_pre" / "cache_manifest.jsonl"
    have: dict[object, dict] = {}
    if not cache_manifest.exists():
        return have
    with cache_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "pre_proj":
                have[row.get("id")] = row
    return have


def link_or_copy(source: Path, dest: Path) -> str | None:
    if not source.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return "existing"
    try:
        os.link(source, dest)
        return "linked"
    except OSError:
        shutil.copy2(source, dest)
        return "copied"


def sync_cache(source_root: Path, output_root: Path, rows: list[dict]) -> dict:
    source_cache = source_root / "img_embeds_pre"
    output_cache = output_root / "img_embeds_pre"
    source_entries = load_cache_rows(source_root)
    manifest_entries: list[dict] = []
    counts: Counter[str] = Counter()
    wanted_ids = {row.get("id") for row in rows}
    rows_by_id = {row.get("id"): row for row in rows}

    for image_id, entry in source_entries.items():
        if image_id not in wanted_ids:
            continue
        embed_name = str(entry.get("embed") or f"{image_id}.pt")
        source_file = source_cache / embed_name
        dest_file = output_cache / embed_name
        result = link_or_copy(source_file, dest_file)
        if result is None:
            counts["missing_file"] += 1
            continue
        counts[result] += 1
        row = rows_by_id.get(image_id) or {}
        manifest_entries.append(
            {
                "id": image_id,
                "embed": embed_name,
                "kind": "pre_proj",
                "n_tok": entry.get("n_tok"),
                "hidden": entry.get("hidden"),
                "tags": row.get("target_tags") or entry.get("tags"),
            }
        )

    manifest_entries.sort(key=lambda item: str(item.get("id")))
    write_jsonl_atomic(output_cache / "cache_manifest.jsonl", manifest_entries)
    counts["manifest_rows"] = len(manifest_entries)
    counts["source_cached_ids"] = len(source_entries)
    counts["wanted_rows"] = len(rows)
    counts["not_yet_cached"] = max(0, len(rows) - len(manifest_entries))
    return dict(counts)


def build_safety_eval(rows: list[dict], output: Path, *, seed: int, limit: int = 64) -> dict:
    rng = random.Random(seed)
    pool = [row for row in rows if normalize_rating(row) in SAFE_RATINGS]
    rng.shuffle(pool)
    selected = []
    for row in pool[:limit]:
        selected.append(
            {
                "id": row.get("id"),
                "rating": row.get("rating"),
                "image": row.get("image"),
                "tags_general": " ".join(tag.replace(" ", "_") for tag in row_tags(row)),
                "eval_source": "mixed_pose_front_safe",
            }
        )
    write_jsonl_atomic(output, selected)
    return {"path": str(output), "rows": len(selected)}


def build_report(
    *,
    source_root: Path,
    output_root: Path,
    rows: list[dict],
    removed_safe_explicit: Counter[str],
    cache_summary: dict | None,
    safety_eval: dict | None,
) -> dict:
    ratings = Counter(normalize_rating(row) for row in rows)
    pose_rows = sum(1 for row in rows if row.get("taxonomy_groups", {}).get("pose_action"))
    action_rows = pose_rows
    return {
        "stage": "tagger_mixed_pose_front_v2_preparation",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "rows": len(rows),
        "ratings": dict(sorted(ratings.items())),
        "pose_rows": pose_rows,
        "action_rows": action_rows,
        "removed_safe_explicit_tags": dict(sorted(removed_safe_explicit.items())),
        "remaining_safe_explicit_tags": explicit_tags_in_safe_rows(rows),
        "pose_action_first_position": first_position_stats(rows, "pose_action"),
        "cache": cache_summary,
        "safety_eval": safety_eval,
        "next": {
            "prepare": r"re-learning\15_prepare_mixed_pose_front_v2.ps1",
            "train": r"re-learning\16_train_mixed_pose_front_v2_4070.ps1",
            "eval": r"re-learning\17_eval_mixed_pose_front_v2_4070.ps1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--safety-eval", default=str(DEFAULT_SAFETY_EVAL))
    parser.add_argument("--subject-lead", type=int, default=8)
    parser.add_argument("--seed", type=int, default=44203)
    parser.add_argument("--sync-cache", action="store_true")
    parser.add_argument("--no-safety-eval", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if source_root.resolve() == output_root.resolve():
        raise SystemExit("source-root and output-root must be different")
    manifest = source_root / "manifest_visual_expand.jsonl"
    if not manifest.exists():
        raise SystemExit(f"missing source manifest: {manifest}")

    rows: list[dict] = []
    removed_safe_explicit: Counter[str] = Counter()
    for row in read_jsonl(manifest):
        prepared, removed = prepare_row(row, subject_lead=args.subject_lead)
        rows.append(prepared)
        if normalize_rating(row) in SAFE_RATINGS:
            removed_safe_explicit.update(removed)

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "manifest_visual_expand.jsonl", rows)
    (output_root / "tag_freq_visual_expand.json").write_text(
        json.dumps(tag_freq(rows, str(source_root / "manifest_visual_expand.jsonl")), ensure_ascii=False),
        encoding="utf-8",
    )

    safety_eval = None
    if not args.no_safety_eval:
        safety_eval = build_safety_eval(rows, Path(args.safety_eval), seed=args.seed)
    cache_summary = sync_cache(source_root, output_root, rows) if args.sync_cache else None
    report = build_report(
        source_root=source_root,
        output_root=output_root,
        rows=rows,
        removed_safe_explicit=removed_safe_explicit,
        cache_summary=cache_summary,
        safety_eval=safety_eval,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "manifest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
