from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\Projects\danbooru_unified")
DEFAULT_TAXONOMY = Path(
    r"E:\1\danbooru-e621-tag-list-processor-main"
    r"\output\tag_taxonomy\llm_2026-05-22"
    r"\tag_taxonomy_llm_classified_merged.jsonl"
)

DROP_ALWAYS = {
    "absurdres",
    "artist name",
    "bad id",
    "character request",
    "chinese commentary",
    "commentary",
    "commentary request",
    "commentary typo",
    "commission",
    "copyright request",
    "english commentary",
    "highres",
    "image sample",
    "korean commentary",
    "lowres",
    "partial commentary",
    "pixiv commission",
    "scan",
    "signature",
    "source request",
    "tagme",
    "translated",
    "translation request",
    "twitter username",
    "voice drama available",
    "watermark",
}

IDENTITY_GROUPS = {"identity", "copyright", "artist", "metadata", "quality_noise"}
COPYRIGHT_TAGS = {
    "original",
    "blue archive",
    "genshin impact",
    "honkai star rail",
    "honkai impact 3rd",
    "zenless zone zero",
    "reverse:1999",
    "touhou",
    "touhou project",
    "fate",
    "fate/grand order",
    "azur lane",
    "arknights",
    "girls frontline",
    "goddess of victory: nikke",
    "nikke",
    "pokemon",
    "uma musume",
    "granblue fantasy",
    "project sekai",
    "idolmaster",
    "love live!",
    "hololive",
    "nijisanji",
    "vocaloid",
    "virtual youtuber",
}
VISUAL_PAREN_QUALIFIERS = {
    "animal",
    "body part",
    "clothes",
    "food",
    "medium",
    "object",
    "plant",
    "print",
    "symbol",
}
VISUAL_GROUP_ORDER = [
    "subject",
    "appearance",
    "body_focus",
    "nsfw",
    "clothing",
    "pose_action",
    "objects",
    "setting",
    "mood",
    "style",
    "composition",
    "time_weather",
    "text_graphics",
    "other",
]

GROUP_PATTERNS = [
    ("subject", [r"^\d+(girl|boy|other)s?$", r"\bsolo\b", r"\bmultiple\b"]),
    ("appearance", [r"\b(hair|eyes|skin|ears|horns|tail|wings)\b"]),
    (
        "clothing",
        [
            r"\b(dress|shirt|skirt|shorts|pants|pantyhose|thighhighs|socks)\b",
            r"\b(shoes|boots|heels|uniform|swimsuit|bikini|bra|panties)\b",
            r"\b(jacket|hoodie|coat|apron|sweater|bodysuit|leotard|sailor)\b",
        ],
    ),
    (
        "pose_action",
        [
            r"\b(standing|sitting|lying|kneeling|crouching|walking|running)\b",
            r"\b(jumping|holding|looking|smile|crying|arms|hands|legs|feet)\b",
            r"\b(open mouth|closed mouth|from side|from behind|looking at viewer)\b",
        ],
    ),
    (
        "setting",
        [
            r"\b(indoors|outdoors|sky|cloud|beach|water|forest|room|bed)\b",
            r"\b(chair|table|desk|window|street|school|classroom|background)\b",
        ],
    ),
    (
        "objects",
        [
            r"\b(weapon|sword|gun|bag|book|phone|umbrella|food|flower|plant)\b",
            r"\b(ribbon|bow|collar|necktie|belt|bracelet|earrings|jewelry)\b",
            r"\b(hair ornament|hairclip|halo)\b",
        ],
    ),
    ("composition", [r"\b(close-up|portrait|upper body|full body|cowboy shot)\b"]),
    ("style", [r"\b(monochrome|greyscale|lineart|sketch|flat color|traditional media)\b"]),
    ("time_weather", [r"\b(night|sunset|rain|snow|day|sunlight)\b"]),
    ("text_graphics", [r"\b(text|speech bubble|symbol|logo)\b"]),
]


class Taxonomy:
    def __init__(self, path: Path):
        self.path = path
        self.group: dict[str, str] = {}
        self.policy: dict[str, str] = {}
        self.role: dict[str, str] = {}
        self._classify_cache: dict[str, str] = {}
        self._drop_cache: dict[str, bool] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tag = norm_tag(row.get("tag"))
                    if not tag:
                        continue
                    self.group[tag] = str(row.get("main_group") or "other")
                    self.policy[tag] = str(row.get("keep_policy") or "keep")
                    self.role[tag] = str(row.get("caption_role") or "descriptive")

    def classify(self, tag: str) -> str:
        cached = self._classify_cache.get(tag)
        if cached is not None:
            return cached
        if tag in self.group:
            result = self.group[tag]
            self._classify_cache[tag] = result
            return result
        for group, patterns in GROUP_PATTERNS:
            if any(re.search(pattern, tag) for pattern in patterns):
                self._classify_cache[tag] = group
                return group
        self._classify_cache[tag] = "other"
        return "other"

    def is_drop(self, tag: str) -> bool:
        cached = self._drop_cache.get(tag)
        if cached is not None:
            return cached
        result = False
        if tag in DROP_ALWAYS:
            result = True
        elif self.policy.get(tag) == "drop":
            result = True
        elif self.group.get(tag) in {"metadata", "quality_noise"}:
            result = True
        elif self.role.get(tag) == "noise":
            result = True
        self._drop_cache[tag] = result
        return result


def norm_tag(tag: object) -> str:
    value = str(tag or "").strip().lower().replace("_", " ")
    value = " ".join(value.split()).strip(" ,.;:")
    return value


def split_tags(raw: object) -> list[str]:
    tags = []
    seen = set()
    text = str(raw or "").strip()
    if not text:
        return []
    parts = text.split(",") if "," in text else text.split()
    for part in parts:
        tag = norm_tag(part)
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def row_tag_sources(raw: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    all_tags = split_tags(raw.get("tags"))
    has_category_tags = any(
        raw.get(key)
        for key in ("tags_general", "tags_character", "tags_copyright", "tags_artist", "tags_meta")
    )
    if not has_category_tags:
        return all_tags, all_tags, [], []
    visual_candidates = split_tags(raw.get("tags_general"))
    identity_candidates = []
    for key in ("tags_character", "tags_copyright", "tags_artist"):
        identity_candidates.extend(split_tags(raw.get(key)))
    meta_candidates = split_tags(raw.get("tags_meta"))
    return all_tags, visual_candidates, identity_candidates, meta_candidates


def stable_split(post_id: object, train: float, val: float) -> str:
    digest = hashlib.sha1(str(post_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def bucket_tags(tags: list[str], taxonomy: Taxonomy) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for tag in tags:
        buckets[taxonomy.classify(tag)].append(tag)
    return dict(buckets)


def is_identity_like(tag: str, taxonomy: Taxonomy) -> bool:
    if not hasattr(taxonomy, "_identity_cache"):
        taxonomy._identity_cache = {}
    cached = taxonomy._identity_cache.get(tag)
    if cached is not None:
        return cached
    group = taxonomy.classify(tag)
    if group in IDENTITY_GROUPS:
        taxonomy._identity_cache[tag] = True
        return True
    if tag in COPYRIGHT_TAGS:
        taxonomy._identity_cache[tag] = True
        return True
    if any(seed in tag for seed in COPYRIGHT_TAGS if seed not in {"original", "fate"}):
        taxonomy._identity_cache[tag] = True
        return True
    if ":" in tag and not re.search(r"\b(:3|:d|:o|:p|:q)\b", tag):
        taxonomy._identity_cache[tag] = True
        return True
    match = re.search(r"\(([^)]+)\)", tag)
    if match and match.group(1).strip().lower() not in VISUAL_PAREN_QUALIFIERS:
        taxonomy._identity_cache[tag] = True
        return True
    taxonomy._identity_cache[tag] = False
    return False


def identity_context(tags: list[str]) -> set[str]:
    values = set()
    for tag in tags:
        match = re.search(r"\(([^)]+)\)", tag)
        if match:
            value = match.group(1).strip().lower()
            if value and value not in VISUAL_PAREN_QUALIFIERS:
                values.add(value)
        if ":" in tag:
            values.add(tag)
    return values


def ordered_visual_tags(groups: dict[str, list[str]], max_tags: int) -> list[str]:
    out = []
    seen = set()
    for group in VISUAL_GROUP_ORDER:
        for tag in groups.get(group, []):
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
                if len(out) >= max_tags:
                    return out
    return out


def describe_from_groups(groups: dict[str, list[str]], lang: str) -> str:
    def joined(group: str, limit: int) -> str:
        return ", ".join(groups.get(group, [])[:limit])

    subject = joined("subject", 4) or "the visible subject"
    appearance = joined("appearance", 8)
    clothing = joined("clothing", 8)
    pose = joined("pose_action", 6)
    setting = joined("setting", 6)
    objects = joined("objects", 6)

    if lang == "ko":
        parts = [f"이미지에는 {subject}가 보인다."]
        if appearance:
            parts.append(f"외형 단서는 {appearance}이다.")
        if clothing:
            parts.append(f"의상 단서는 {clothing}이다.")
        if pose:
            parts.append(f"포즈와 동작 단서는 {pose}이다.")
        if setting:
            parts.append(f"배경 단서는 {setting}이다.")
        if objects:
            parts.append(f"소품 단서는 {objects}이다.")
        return " ".join(parts)
    if lang == "ja":
        parts = [f"画像には {subject} が見える。"]
        if appearance:
            parts.append(f"外見の手がかりは {appearance}。")
        if clothing:
            parts.append(f"服装の手がかりは {clothing}。")
        if pose:
            parts.append(f"ポーズや動作は {pose}。")
        if setting:
            parts.append(f"背景の手がかりは {setting}。")
        if objects:
            parts.append(f"小物の手がかりは {objects}。")
        return " ".join(parts)
    parts = [f"The image shows {subject}."]
    if appearance:
        parts.append(f"Visible appearance cues include {appearance}.")
    if clothing:
        parts.append(f"Clothing cues include {clothing}.")
    if pose:
        parts.append(f"Pose and action cues include {pose}.")
    if setting:
        parts.append(f"Setting cues include {setting}.")
    if objects:
        parts.append(f"Object cues include {objects}.")
    return " ".join(parts)


def planner_record(row: dict, lang: str) -> dict:
    tags = row["target_tags"]
    if lang == "ko":
        user = (
            "다음 이미지 설명을 Danbooru 태그로만 변환해줘. "
            "영어 소문자 태그만 쉼표로 구분하고 설명문은 쓰지 마.\n"
            f"설명: {row['descriptions']['ko']}"
        )
    elif lang == "ja":
        user = (
            "次の画像説明をDanbooruタグだけに変換してください。"
            "英小文字タグをカンマ区切りで出力し、説明文は書かないでください。\n"
            f"説明: {row['descriptions']['ja']}"
        )
    else:
        user = (
            "Convert the following image description into Danbooru tags only. "
            "Use lowercase English tags separated by commas. Do not write prose.\n"
            f"Description: {row['descriptions']['en']}"
        )
    return {
        "id": row["id"],
        "lang": lang,
        "split": row["split"],
        "source_dataset": row.get("source_dataset"),
        "image": row["image"],
        "user": user,
        "assistant": tags,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": tags},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--out-dir", default="processed_v1")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--min-tags", type=int, default=8)
    parser.add_argument("--max-tags", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--keep-identity-in-visual",
        action="store_true",
        help="Keep identity/copyright/artist-like tags in visual target_tags.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    src = root / args.manifest
    out_dir = root / args.out_dir
    split_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = Taxonomy(Path(args.taxonomy))
    counters = Counter()
    split_counts = Counter()
    tag_df = Counter()
    core_df = Counter()
    support_df = Counter()
    dropped_df = Counter()

    files = {
        "canonical": (out_dir / "canonical_manifest.jsonl").open("w", encoding="utf-8"),
        "visual": (out_dir / "manifest_visual_expand.jsonl").open("w", encoding="utf-8"),
        "planner_all": (out_dir / "text_tag_planner_all.jsonl").open("w", encoding="utf-8"),
    }
    split_files = {}
    try:
        for split in ("train", "val", "test"):
            split_files[("vision", split)] = (
                split_dir / f"vision_tagger_{split}.jsonl"
            ).open("w", encoding="utf-8")
            split_files[("planner", split)] = (
                split_dir / f"text_tag_planner_{split}.jsonl"
            ).open("w", encoding="utf-8")

        with src.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                if args.limit and counters["rows"] >= args.limit:
                    break
                counters["rows"] += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    counters["bad_json"] += 1
                    continue

                image_rel = str(raw.get("image") or "")
                if not image_rel or not (root / image_rel).exists():
                    counters["missing_image"] += 1
                    continue

                source_tags, visual_candidates, explicit_identity, explicit_meta = row_tag_sources(raw)
                if not source_tags:
                    counters["missing_tags"] += 1
                    continue

                canonical = []
                dropped = list(explicit_meta)
                explicit_identity = list(dict.fromkeys(explicit_identity))
                for tag in visual_candidates:
                    if taxonomy.is_drop(tag):
                        dropped.append(tag)
                    else:
                        canonical.append(tag)
                groups_all = bucket_tags(canonical, taxonomy)
                row_identity_context = identity_context(canonical)

                visual_pool = []
                identity_tags = list(explicit_identity)
                for tag in canonical:
                    group = taxonomy.classify(tag)
                    if (
                        not args.keep_identity_in_visual
                        and (is_identity_like(tag, taxonomy) or tag in row_identity_context)
                    ):
                        if tag not in identity_tags:
                            identity_tags.append(tag)
                    else:
                        visual_pool.append(tag)
                groups_visual = bucket_tags(visual_pool, taxonomy)
                visual_tags = ordered_visual_tags(groups_visual, args.max_tags)
                if len(visual_tags) < args.min_tags:
                    counters["too_few_visual_tags"] += 1
                    continue

                split = stable_split(raw.get("id"), args.train_ratio, args.val_ratio)
                split_counts[split] += 1
                target = ", ".join(visual_tags)
                canonical_record = {
                    "id": raw.get("id"),
                    "split": split,
                    "image": image_rel,
                    "image_abs": str(root / image_rel),
                    "width": raw.get("width"),
                    "height": raw.get("height"),
                    "rating": raw.get("rating"),
                    "score": raw.get("score"),
                    "source_dataset": raw.get("source_dataset"),
                    "source_image": raw.get("source_image"),
                    "source_tags": source_tags,
                    "canonical_tags": canonical,
                    "visual_tags": visual_tags,
                    "identity_tags": identity_tags,
                    "dropped_tags": dropped,
                    "taxonomy_groups": groups_visual,
                    "all_taxonomy_groups": groups_all,
                    "target_tags": target,
                    "n_tags": len(visual_tags),
                    "descriptions": {
                        "en": describe_from_groups(groups_visual, "en"),
                        "ko": describe_from_groups(groups_visual, "ko"),
                        "ja": describe_from_groups(groups_visual, "ja"),
                    },
                    "preprocess": "danbooru_unified_canonical_v1",
                }
                visual_record = {
                    "id": canonical_record["id"],
                    "split": split,
                    "rating": canonical_record["rating"],
                    "score": canonical_record["score"],
                    "image": image_rel,
                    "width": canonical_record["width"],
                    "height": canonical_record["height"],
                    "n_tags": len(visual_tags),
                    "tags": target,
                    "target_tags": target,
                    "flat_tags": visual_tags,
                    "taxonomy_groups": groups_visual,
                    "core_tags": visual_tags,
                    "support_tags": [],
                    "identity_tags": identity_tags,
                    "blocked_tags": dropped + identity_tags,
                    "source_dataset": canonical_record["source_dataset"],
                    "preprocess": "danbooru_unified_visual_flat_v1",
                }

                files["canonical"].write(json.dumps(canonical_record, ensure_ascii=False) + "\n")
                files["visual"].write(json.dumps(visual_record, ensure_ascii=False) + "\n")
                split_files[("vision", split)].write(
                    json.dumps(visual_record, ensure_ascii=False) + "\n"
                )
                for tag in set(visual_tags):
                    tag_df[tag] += 1
                    core_df[tag] += 1
                for tag in set(dropped + identity_tags):
                    dropped_df[tag] += 1

                for lang in ("ko", "ja", "en"):
                    rec = planner_record(canonical_record, lang)
                    files["planner_all"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    split_files[("planner", split)].write(
                        json.dumps(rec, ensure_ascii=False) + "\n"
                    )
                counters["kept"] += 1
                if counters["kept"] % 10000 == 0:
                    print(f"[unified] kept {counters['kept']:,} / read {counters['rows']:,}", flush=True)
    finally:
        for f in list(files.values()) + list(split_files.values()):
            f.close()

    freq = {
        "n_docs": counters["kept"],
        "df": dict(sorted(tag_df.items())),
        "core_df": dict(sorted(core_df.items())),
        "support_df": dict(sorted(support_df.items())),
        "dropped_df": dict(sorted(dropped_df.items())),
        "preprocess": "danbooru_unified_visual_flat_v1",
        "source": str(src),
    }
    (out_dir / "tag_freq_visual_expand.json").write_text(
        json.dumps(freq, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "source": str(src),
        "out_dir": str(out_dir),
        "taxonomy": str(Path(args.taxonomy)),
        "taxonomy_loaded": bool(taxonomy.group),
        "min_tags": args.min_tags,
        "max_tags": args.max_tags,
        "keep_identity_in_visual": args.keep_identity_in_visual,
        "counters": dict(counters),
        "splits": dict(split_counts),
        "unique_visual_tags": len(tag_df),
        "top_visual_tags": tag_df.most_common(50),
        "top_dropped_or_blocked_tags": dropped_df.most_common(50),
        "outputs": {
            "canonical_manifest": str(out_dir / "canonical_manifest.jsonl"),
            "manifest_visual_expand": str(out_dir / "manifest_visual_expand.jsonl"),
            "tag_freq_visual_expand": str(out_dir / "tag_freq_visual_expand.json"),
            "text_tag_planner_all": str(out_dir / "text_tag_planner_all.jsonl"),
            "splits": str(split_dir),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
