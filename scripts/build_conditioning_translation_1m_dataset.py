import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


GROUP_ORDER = [
    "subject",
    "identity",
    "appearance",
    "body_focus",
    "clothing",
    "pose_action",
    "composition",
    "setting",
    "objects",
    "mood",
    "style",
    "time_weather",
    "text_graphics",
]

VARIANT_ORDER = [
    "full_prompt",
    "taxonomy_balanced",
    "pose_composition",
    "visual_grounding",
    "subject_identity",
    "action_scene",
    "style_mood",
    "compact_anchor",
]

QUALITY_BY_RATING = {
    "g": "masterpiece, best quality, score_7, safe",
    "s": "masterpiece, best quality, score_7, sensitive",
    "q": "masterpiece, best quality, score_7, questionable",
    "e": "masterpiece, best quality, score_7, explicit",
}


def split_tags(value):
    if isinstance(value, list):
        return [clean_tag(v) for v in value if clean_tag(v)]
    return [clean_tag(v) for v in str(value or "").split(",") if clean_tag(v)]


def clean_tag(value):
    return str(value or "").strip().replace("_", " ")


def canonical_tag(value):
    return str(value or "").strip().replace(" ", "_")


def uniq(items):
    seen = set()
    out = []
    for item in items:
        item = clean_tag(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_prompt(parts):
    tags = []
    for part in parts:
        if isinstance(part, str):
            tags.extend(split_tags(part))
        else:
            tags.extend(part or [])
    return ", ".join(uniq(tags))


def load_taxonomy(paths):
    tag_to_group = {}
    stats = Counter()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("keep_policy") == "drop":
                    continue
                tag = canonical_tag(row.get("tag"))
                group = row.get("main_group") or "other"
                if not tag:
                    continue
                tag_to_group[tag] = group
                stats[group] += 1
                for alias in str(row.get("aliases") or "").split(","):
                    alias = canonical_tag(alias)
                    if alias:
                        tag_to_group.setdefault(alias, group)
    return tag_to_group, stats


def row_groups(row, tag_to_group):
    groups = defaultdict(list)
    source_groups = row.get("taxonomy_groups")
    if isinstance(source_groups, dict):
        for group, values in source_groups.items():
            for tag in values or []:
                groups[str(group)].append(clean_tag(tag))
    for tag in split_tags(row.get("target_tags") or row.get("tags") or row.get("text")):
        group = tag_to_group.get(canonical_tag(tag), "other")
        groups[group].append(tag)
    return {group: uniq(values) for group, values in groups.items() if values}


def take(groups, names, limit):
    values = []
    for name in names:
        values.extend(groups.get(name, []))
    return uniq(values)[:limit]


def rating_prefix(row):
    return QUALITY_BY_RATING.get(str(row.get("rating") or "").lower(), "masterpiece, best quality, score_7")


def full_prompt(row, groups):
    text = row.get("teacher_prompt") or row.get("visible_prompt") or row.get("text") or row.get("target_tags")
    return join_prompt([text])


def taxonomy_balanced(row, groups):
    parts = [rating_prefix(row)]
    for group in GROUP_ORDER:
        limit = 12 if group in {"appearance", "clothing", "pose_action"} else 8
        values = groups.get(group, [])[:limit]
        if values:
            parts.append(values)
    return join_prompt(parts)


def pose_composition(row, groups):
    parts = [
        rating_prefix(row),
        take(groups, ["subject", "identity"], 8),
        take(groups, ["pose_action", "body_focus", "composition"], 24),
        take(groups, ["appearance", "clothing"], 12),
        take(groups, ["setting", "objects"], 8),
    ]
    prompt = join_prompt(parts)
    if "pose focus" not in prompt:
        prompt = join_prompt(["pose focus, clear body pose", prompt])
    return prompt


def visual_grounding(row, groups):
    parts = [
        rating_prefix(row),
        take(groups, ["subject", "identity"], 8),
        take(groups, ["appearance", "body_focus"], 18),
        take(groups, ["clothing"], 16),
        take(groups, ["setting", "objects", "time_weather"], 20),
        take(groups, ["composition", "mood", "style"], 16),
    ]
    return join_prompt(parts)


def subject_identity(row, groups):
    parts = [
        rating_prefix(row),
        take(groups, ["subject"], 10),
        take(groups, ["identity"], 12),
        take(groups, ["appearance"], 18),
        take(groups, ["clothing"], 12),
    ]
    return join_prompt(parts)


def action_scene(row, groups):
    parts = [
        rating_prefix(row),
        take(groups, ["subject"], 6),
        take(groups, ["pose_action"], 18),
        take(groups, ["composition"], 12),
        take(groups, ["setting", "objects", "time_weather"], 22),
    ]
    return join_prompt(parts)


def style_mood(row, groups):
    parts = [
        rating_prefix(row),
        take(groups, ["subject", "identity"], 8),
        take(groups, ["mood", "style", "text_graphics"], 18),
        take(groups, ["composition"], 12),
        take(groups, ["appearance", "clothing"], 14),
    ]
    return join_prompt(parts)


def compact_anchor(row, groups):
    parts = [
        rating_prefix(row),
        take(groups, ["subject", "identity"], 8),
        take(groups, ["appearance"], 10),
        take(groups, ["clothing"], 8),
        take(groups, ["pose_action"], 8),
        take(groups, ["composition", "setting"], 8),
    ]
    return join_prompt(parts)


VARIANT_BUILDERS = {
    "full_prompt": full_prompt,
    "taxonomy_balanced": taxonomy_balanced,
    "pose_composition": pose_composition,
    "visual_grounding": visual_grounding,
    "subject_identity": subject_identity,
    "action_scene": action_scene,
    "style_mood": style_mood,
    "compact_anchor": compact_anchor,
}


def remap_image_embed(row, image_embed_root):
    image_embed = row.get("image_embed_pre")
    if image_embed and Path(image_embed).exists():
        return str(Path(image_embed))
    if image_embed_root:
        source_id = row.get("source_id", row.get("id"))
        if source_id is not None:
            candidate = Path(image_embed_root) / f"{source_id}.pt"
            if candidate.exists():
                return str(candidate)
    return image_embed


def valid_source(row, require_embed, image_embed_root):
    prompt = row.get("teacher_prompt") or row.get("visible_prompt") or row.get("text") or row.get("target_tags")
    if not prompt:
        return False
    image_embed = remap_image_embed(row, image_embed_root)
    if require_embed and (not image_embed or not Path(image_embed).exists()):
        return False
    row["image_embed_pre"] = image_embed
    return True


def source_key(row):
    return row.get("source_id", row.get("id", row.get("idx")))


def build_dataset(args):
    taxonomy, taxonomy_stats = load_taxonomy(args.taxonomy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    variants = Counter()
    ratings = Counter()
    source_files = []
    idx = args.idx_start

    with out.open("w", encoding="utf-8", newline="\n") as fout:
        for src in args.source:
            src_path = Path(src)
            source_files.append(str(src_path))
            with src_path.open("r", encoding="utf-8-sig") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    stats["scanned"] += 1
                    row = json.loads(line)
                    if not valid_source(row, args.require_image_embed, args.image_embed_root):
                        stats["skipped_invalid_source"] += 1
                        continue
                    groups = row_groups(row, taxonomy)
                    for variant in VARIANT_ORDER:
                        prompt = VARIANT_BUILDERS[variant](row, groups)
                        if not prompt:
                            stats["skipped_empty_prompt"] += 1
                            continue
                        record = {
                            "idx": idx,
                            "source_idx": row.get("idx"),
                            "source_id": source_key(row),
                            "variant": variant,
                            "text": prompt,
                            "visible_prompt": prompt,
                            "teacher_prompt": prompt,
                            "image": row.get("image"),
                            "image_embed_pre": row.get("image_embed_pre"),
                            "input_modalities": ["image", "text"],
                            "source_schema": "conditioning_translation_1m_v1",
                            "source_dataset_schema": row.get("source_schema") or row.get("schema"),
                            "rating": row.get("rating"),
                            "width": row.get("width"),
                            "height": row.get("height"),
                            "target_tags": row.get("target_tags"),
                            "taxonomy_groups": groups,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        idx += 1
                        stats["written"] += 1
                        variants[variant] += 1
                        ratings[str(row.get("rating") or "unknown")] += 1
                        if stats["written"] >= args.limit:
                            break
                    if stats["written"] >= args.limit:
                        break
            if stats["written"] >= args.limit:
                break

    summary = {
        "stage": "conditioning_translation_1m_v1",
        "output": str(out),
        "sources": source_files,
        "taxonomy_files": [str(Path(p)) for p in args.taxonomy if Path(p).exists()],
        "taxonomy_groups_loaded": dict(sorted(taxonomy_stats.items())),
        "idx_start": args.idx_start,
        "idx_end": idx - 1 if stats["written"] else None,
        "limit": args.limit,
        "stats": dict(stats),
        "variant_counts": dict(variants),
        "rating_counts": dict(ratings),
        "require_image_embed": args.require_image_embed,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--taxonomy", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--idx-start", type=int, default=5_000_000)
    parser.add_argument("--image-embed-root")
    parser.add_argument("--require-image-embed", action="store_true")
    build_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
