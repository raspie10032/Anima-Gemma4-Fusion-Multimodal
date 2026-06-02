from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemmanima.training.text_preservation import (
    build_general_scene_regression_prompt_records,
    build_text_preservation_heldout_prompt_records,
)


BUCKET_COUNTS = {
    "00_fixed6_hard": 600,
    "10_fragile_text": 2000,
    "20_heldout_text": 3000,
    "30_general_scene": 3000,
    "40_synthetic_text": 1400,
}

DEFAULT_OUTPUT_DIR = Path("reports/text_rendering_qwen_baseline")
DEFAULT_COUNT = sum(BUCKET_COUNTS.values())
DEFAULT_SEED = 37000
DEFAULT_START_IDX = 993000

FIXED6_CASES = (
    {
        "case_id": "text_eval_001_sign_luna_gate",
        "target": "LUNA GATE",
        "surface": "moonlit street sign",
        "setting": "a close-up sign mounted on a quiet stone wall",
        "style": "large crisp serif letters",
        "weight": 8.0,
    },
    {
        "case_id": "text_eval_002_book_cover_star_atlas",
        "target": "STAR ATLAS",
        "surface": "fantasy book cover",
        "setting": "a book lying flat on a clean desk",
        "style": "centered serif title letters",
        "weight": 8.0,
    },
    {
        "case_id": "text_eval_003_magic_circle_aether",
        "target": "AETHER",
        "surface": "glowing magic circle",
        "setting": "a flat carved stone floor",
        "style": "repeated glyph-like letters around the ring",
        "weight": 18.0,
    },
    {
        "case_id": "text_eval_004_ui_panel_hp_42",
        "target": "HP 42",
        "surface": "anime game UI panel",
        "setting": "a neutral dark background",
        "style": "simple block UI lettering",
        "weight": 8.0,
    },
    {
        "case_id": "text_eval_005_handwritten_note_meet_at_dawn",
        "target": "MEET AT DAWN",
        "surface": "handwritten paper note",
        "setting": "a wooden door with the note pinned at center",
        "style": "readable natural handwriting",
        "weight": 14.0,
    },
    {
        "case_id": "text_eval_006_label_tea",
        "target": "TEA",
        "surface": "small glass jar label",
        "setting": "a cozy kitchen shelf",
        "style": "plain dark ink on a pale paper label",
        "weight": 14.0,
    },
)

TEXT_SURFACES = (
    "neon sign",
    "book cover",
    "poster",
    "glass jar label",
    "paper note",
    "game UI panel",
    "street banner",
    "ticket stub",
    "shop awning",
    "warning label",
    "map title",
    "magic seal",
    "wooden plaque",
    "menu board",
    "shipping label",
    "arcade marquee",
    "train platform sign",
    "museum placard",
    "bottle label",
    "library card",
)

TEXT_SETTINGS = (
    "a rainy storefront window",
    "a blue hardcover book on a maple desk",
    "a clean transit poster on a tiled wall",
    "a small kitchen jar in morning light",
    "a pinned note on cork board",
    "a compact fantasy game status panel",
    "a fabric banner stretched between posts",
    "a cream concert ticket on a table",
    "a striped awning above a closed shop",
    "a simple label on a metal kettle",
    "a folded illustrated map",
    "a glowing circle carved into dark stone",
    "a polished wooden door",
    "a chalk menu in a quiet cafe",
    "a cardboard package label",
    "a retro arcade cabinet header",
    "a clean train platform at dusk",
    "a quiet museum wall",
    "a glass bottle on white cloth",
    "an old library checkout card",
)

TEXT_STYLES = (
    "center the text and make every letter readable",
    "use bold block letters with high contrast",
    "use serif letters with clean spacing",
    "use handwritten letters that remain readable",
    "use simple sans-serif letters with generous spacing",
    "use clear white letters on a dark background",
    "use dark ink on a pale label",
    "use glowing cyan letters without distorting the word",
    "make the target text the only prominent readable wording",
    "keep decorative borders away from the letters",
)

TEXT_VALUES = (
    "AETHER",
    "MEET AT DAWN",
    "TEA",
    "HP 42",
    "LUNA GATE",
    "STAR ATLAS",
    "NOVA CAFE",
    "MOON INDEX",
    "SKY RAIL",
    "MINT",
    "CALL LUNA",
    "MP 88",
    "EAST GATE",
    "ROW 7",
    "SUN BAKERY",
    "HOT",
    "OLD HARBOR",
    "BLUE STAR",
    "LEVEL 12",
    "OPEN",
    "CLOUD NINE",
    "NOON BELL",
    "GOLD KEY",
    "ROOM 305",
    "SALT",
    "NORTH EXIT",
    "RED MOON",
    "KITE SHOP",
    "ZONE 4",
    "PIXEL BAR",
    "JAZZ NIGHT",
    "EMBER",
    "MILK",
    "BOOK 17",
    "RIVER WALK",
    "ALPHA",
    "BETA 9",
    "LILAC",
    "CAFE 24",
    "DREAM LOG",
    "FROST",
    "SUNSET",
    "BLOOM",
    "SILVER LINE",
    "ARCADE",
    "TICKET A",
    "HONEY",
    "VIOLET",
    "ORBIT",
    "DELTA",
    "SPARK",
    "GREEN TEA",
    "WISH",
    "MAPLE",
    "CROWN",
    "BRIGHT",
    "QUIET HILL",
    "STATION 8",
    "MORNING",
    "NIGHT BUS",
    "RUBY",
    "PEACH",
    "COSMOS",
    "LIGHT",
)

LAYOUTS = (
    "large centered crop",
    "flat front view",
    "close-up readable crop",
    "slight angle with text still centered",
    "straight-on product-style crop",
)


def _now_local() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _path(path: str | Path) -> str:
    return Path(path).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _tag_record(
    record: dict[str, Any],
    *,
    bucket: str,
    record_index: int,
    sample_weight: float,
    idx: int,
    source_case_id: str | None = None,
    target_text: str | None = None,
) -> dict[str, Any]:
    tagged = dict(record)
    tagged["idx"] = idx
    tagged["v37_record_index"] = record_index
    tagged["cache_source_bucket"] = bucket
    tagged["sample_weight"] = sample_weight
    tagged["src"] = f"v37_{bucket}_{record_index:05d}"
    tagged["id"] = f"v37_{bucket}_{record_index:05d}"
    if source_case_id is not None:
        tagged["source_case_id"] = source_case_id
    elif "source_case_id" not in tagged:
        tagged["source_case_id"] = tagged["src"]
    if target_text is not None:
        tagged["target_text"] = target_text
    return tagged


def _fixed6_records(count: int, *, start_idx: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    modifiers = (
        "keep the object centered and large enough to inspect the letters",
        "use a clean uncluttered background and preserve the exact spelling",
        "use high contrast between the letters and the surrounding surface",
        "avoid extra readable words and keep the target text dominant",
        "make the crop close enough that every letter can be checked",
    )
    for index in range(count):
        case = FIXED6_CASES[index % len(FIXED6_CASES)]
        variant = index // len(FIXED6_CASES)
        modifier = modifiers[variant % len(modifiers)]
        text = (
            "Object-only text rendering fixed6 guard: draw "
            f"{case['setting']} with a {case['surface']} that clearly reads {case['target']}; "
            f"{case['style']}; {modifier}. "
            "No people, no characters, no faces, no bodies, no hands."
        )
        weight = float(case["weight"]) + (2.0 if case["target"] in {"AETHER", "MEET AT DAWN", "TEA"} else 0.0)
        records.append(
            _tag_record(
                {
                    "text": text,
                    "eval_idx": index,
                    "curriculum_bucket": "fixed6_hard_guard_10k",
                    "fixed6_hard_guard": True,
                },
                bucket="00_fixed6_hard",
                record_index=index,
                sample_weight=weight,
                idx=start_idx + index,
                source_case_id=str(case["case_id"]),
                target_text=str(case["target"]),
            )
        )
    return records


def _fragile_records(count: int, *, start_idx: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    fragile_targets = (
        ("AETHER", 18.0, "aether_fixed6_guard"),
        ("MEET AT DAWN", 14.0, "meet_handwriting_guard"),
        ("TEA", 14.0, "tea_label_guard"),
        ("HP 42", 9.0, "ui_label_guard"),
        ("LUNA GATE", 8.0, "sign_label_guard"),
        ("STAR ATLAS", 8.0, "book_title_guard"),
        ("GREEN TEA", 6.0, "short_label_guard"),
        ("NOON BELL", 6.0, "phrase_spacing_guard"),
    )
    target_cycle = list(fragile_targets)
    rng.shuffle(target_cycle)
    records: list[dict[str, Any]] = []
    for index in range(count):
        target, weight, curriculum = target_cycle[index % len(target_cycle)]
        surface = TEXT_SURFACES[(index * 7 + 3) % len(TEXT_SURFACES)]
        setting = TEXT_SETTINGS[(index * 11 + 5) % len(TEXT_SETTINGS)]
        style = TEXT_STYLES[(index * 13 + 7) % len(TEXT_STYLES)]
        layout = LAYOUTS[(index * 17) % len(LAYOUTS)]
        text = (
            "Object-only fragile text preservation: draw "
            f"{setting} with a {surface} that clearly reads {target}; "
            f"{style}; {layout}. "
            "No people, no characters, no faces, no hands. Do not add any other readable words."
        )
        records.append(
            _tag_record(
                {
                    "text": text,
                    "eval_idx": index,
                    "curriculum_bucket": curriculum,
                    "fragile_text_guard": True,
                },
                bucket="10_fragile_text",
                record_index=index,
                sample_weight=weight,
                idx=start_idx + index,
                source_case_id=f"v37_fragile_{target.lower().replace(' ', '_')}_{index:05d}",
                target_text=target,
            )
        )
    return records


def _heldout_records(count: int, *, start_idx: int) -> list[dict[str, Any]]:
    base_records = build_text_preservation_heldout_prompt_records(
        count=count,
        start_idx=start_idx,
        prompt_index_offset=60000,
        src_prefix="v37_heldout_base",
        include_sample_marker=True,
    )
    records: list[dict[str, Any]] = []
    for index, record in enumerate(base_records):
        records.append(
            _tag_record(
                {
                    **record,
                    "curriculum_bucket": "heldout_text_repair_10k",
                    "heldout_text_guard": True,
                    "source_original_idx": record["idx"],
                },
                bucket="20_heldout_text",
                record_index=index,
                sample_weight=4.0,
                idx=start_idx + index,
                source_case_id=f"v37_heldout_{index:05d}",
                target_text=record.get("target_text"),
            )
        )
    return records


def _general_records(count: int, *, start_idx: int) -> list[dict[str, Any]]:
    base_records = build_general_scene_regression_prompt_records(
        count=count,
        start_idx=start_idx,
        src_prefix="v37_general_base",
    )
    records: list[dict[str, Any]] = []
    for index, record in enumerate(base_records):
        records.append(
            _tag_record(
                {
                    **record,
                    "curriculum_bucket": "general_scene_regularizer_10k",
                    "general_regularizer": True,
                    "source_original_idx": record["idx"],
                },
                bucket="30_general_scene",
                record_index=index,
                sample_weight=1.25,
                idx=start_idx + index,
                source_case_id=f"v37_general_{record['category']}_{index:05d}",
            )
        )
    return records


def _synthetic_records(count: int, *, start_idx: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 41)
    values = list(TEXT_VALUES)
    rng.shuffle(values)
    records: list[dict[str, Any]] = []
    for index in range(count):
        target = values[index % len(values)]
        surface = TEXT_SURFACES[(index * 5 + 2) % len(TEXT_SURFACES)]
        setting = TEXT_SETTINGS[(index * 3 + 1) % len(TEXT_SETTINGS)]
        style = TEXT_STYLES[(index * 19 + 4) % len(TEXT_STYLES)]
        layout = LAYOUTS[(index * 23 + 1) % len(LAYOUTS)]
        text = (
            "Object-only synthetic text rendering: draw "
            f"{setting} with a {surface} that clearly reads {target}; "
            f"{style}; {layout}; clean readable lettering. "
            "No people, no characters, no faces, no hands. Keep only the requested text readable."
        )
        weight = 3.5 if target in {"AETHER", "MEET AT DAWN", "TEA", "HP 42"} else 2.5
        records.append(
            _tag_record(
                {
                    "text": text,
                    "eval_idx": index,
                    "curriculum_bucket": "synthetic_text_surface_10k",
                    "synthetic_text_guard": True,
                },
                bucket="40_synthetic_text",
                record_index=index,
                sample_weight=weight,
                idx=start_idx + index,
                source_case_id=f"v37_synthetic_{target.lower().replace(' ', '_')}_{index:05d}",
                target_text=target,
            )
        )
    return records


def _split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        bucket: [record for record in records if record["cache_source_bucket"] == bucket]
        for bucket in BUCKET_COUNTS
    }


def _surface_curriculum(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "text_preservation_v37_10k_surface_curriculum_weights",
        "protected_baseline": "v5",
        "record_count": len(records),
        "bucket_weight_policy": {
            "00_fixed6_hard": "highest replay weight, especially AETHER/MEET/TEA",
            "10_fragile_text": "fragile text retention and exact spelling",
            "20_heldout_text": "broad heldout text repair",
            "30_general_scene": "general scene regularization to reduce collateral damage",
            "40_synthetic_text": "new surface/word combinations for volume",
        },
        "records": [
            {
                "id": record["id"],
                "idx": record["idx"],
                "cache_source_bucket": record["cache_source_bucket"],
                "sample_weight": record["sample_weight"],
                "target_text": record.get("target_text"),
            }
            for record in records
        ],
    }


def _loss_config() -> dict[str, Any]:
    return {
        "stage": "text_preservation_v37_10k_artifact_gate_loss_config",
        "protected_baseline": "v5",
        "default_changed": False,
        "sample_weight_field": "sample_weight",
        "artifact_gate": {
            "enabled": True,
            "fixed6_first": True,
            "hard_stop_delta_vs_v5_gt": 0.0,
            "guard_targets": ["AETHER", "MEET AT DAWN", "TEA", "HP 42"],
        },
        "recommended_training": {
            "resume_from": "runs/cache/text_preservation_blended_v28/bridge/text_preservation_blended_v28_block0_a100_bridge.pt",
            "fallback_resume_from": "runs/cache/text_preservation_blended_v5/bridge/text_preservation_blended_v5_bridge.pt",
            "lr_range": [1e-08, 5e-08],
            "anchor_lambda_range": [0.05, 0.12],
        },
    }


def _manifest(
    *,
    output_dir: Path,
    count: int,
    seed: int,
    records: list[dict[str, Any]],
    split_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": "text_preservation_v37_10k_curriculum_manifest",
        "created_local": _now_local(),
        "mode": "ten_k_shared_cache_for_fixed6_first_training",
        "protected_baseline": "v5",
        "v5_default_changed": False,
        "promotion_allowed": False,
        "seed": seed,
        "record_count": count,
        "record_buckets": dict(BUCKET_COUNTS),
        "prompt_records": _path(output_dir / "v37_10k_prompt_records.jsonl"),
        "split_prompt_files": split_files,
        "surface_curriculum": _path(output_dir / "v37_10k_surface_curriculum_weights.json"),
        "artifact_gate_loss_config": _path(output_dir / "v37_10k_artifact_gate_loss_config.json"),
        "source_reports": {
            "v36_sweep_status": _path(
                "reports/text_rendering_qwen_baseline/workflow_status_v36_big_batch_sweep.json"
            ),
            "v36_manifest": _path(
                "reports/text_rendering_qwen_baseline/v36_big_batch_curriculum_manifest.json"
            ),
            "v35_objective_manifest": _path(
                "reports/text_rendering_qwen_baseline/v35_objective_manifest.json"
            ),
        },
        "required_gpu_policy": {
            "CUDA_VISIBLE_DEVICES": "0",
            "allowed_gpu": "RTX 4070 Ti SUPER",
            "disallowed_gpu": "RTX 5060",
        },
        "cache_plan": {
            "target_cache_outdir": "runs/cache/text_preservation_blended_v37/targets",
            "gemma_cache_outdir": "runs/cache/text_preservation_blended_v37/gemma",
            "training_policy": (
                "Build target and Gemma caches by bucket, train only v37 candidates, "
                "run fixed6 render/compare before heldout or general expansion."
            ),
        },
        "hard_stop_conditions": [
            "Any fixed6 candidate delta_vs_v5 > 0.0 blocks that candidate from expansion",
            "Do not overwrite v5/default/release metadata",
            "CUDA_VISIBLE_DEVICES must be exactly 0 for GPU work",
            "Do not use RTX 5060",
        ],
        "sample_weight_summary": {
            bucket: {
                "count": sum(record["cache_source_bucket"] == bucket for record in records),
                "total_weight": round(
                    sum(
                        float(record["sample_weight"])
                        for record in records
                        if record["cache_source_bucket"] == bucket
                    ),
                    4,
                ),
            }
            for bucket in BUCKET_COUNTS
        },
    }


def _audit(records: list[dict[str, Any]], split_files: dict[str, str]) -> dict[str, Any]:
    bucket_counts = Counter(record["cache_source_bucket"] for record in records)
    idx_counts = Counter(record["idx"] for record in records)
    id_counts = Counter(record["id"] for record in records)
    fixed6_cases = sorted(
        {
            record["source_case_id"]
            for record in records
            if record["cache_source_bucket"] == "00_fixed6_hard"
        }
    )
    target_counter = Counter(record.get("target_text") for record in records if record.get("target_text"))
    required_targets = {"AETHER", "MEET AT DAWN", "TEA", "HP 42"}
    status = "pass"
    failures: list[str] = []
    if dict(bucket_counts) != BUCKET_COUNTS:
        status = "fail"
        failures.append("bucket_counts_mismatch")
    if len(idx_counts) != len(records):
        status = "fail"
        failures.append("duplicate_idx")
    if len(id_counts) != len(records):
        status = "fail"
        failures.append("duplicate_id")
    if not required_targets <= set(target_counter):
        status = "fail"
        failures.append("required_fragile_targets_missing")
    if len(fixed6_cases) < len(FIXED6_CASES):
        status = "fail"
        failures.append("fixed6_cases_missing")
    return {
        "stage": "text_preservation_v37_10k_manifest_audit",
        "created_local": _now_local(),
        "status": status,
        "failures": failures,
        "record_count": len(records),
        "unique_idx_count": len(idx_counts),
        "unique_id_count": len(id_counts),
        "bucket_counts": {bucket: bucket_counts[bucket] for bucket in BUCKET_COUNTS},
        "split_prompt_files": split_files,
        "fixed6_source_case_ids": fixed6_cases,
        "required_target_counts": {target: target_counter[target] for target in sorted(required_targets)},
        "protected_baseline": "v5",
        "v5_default_changed": False,
        "promotion_allowed": False,
        "gpu_permission": "eligible_for_4070_v37_cache_build_after_user_requested_10k",
        "required_gpu_policy": {
            "CUDA_VISIBLE_DEVICES": "0",
            "allowed_gpu": "RTX 4070 Ti SUPER",
            "disallowed_gpu": "RTX 5060",
        },
        "next_step": "build v37 target cache shards by bucket on 4070",
    }


def build_v37_10k_curriculum(
    *,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    start_idx: int = DEFAULT_START_IDX,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    if count != DEFAULT_COUNT:
        raise ValueError(f"v37 10k curriculum requires count={DEFAULT_COUNT}; got {count}")
    output_path = Path(output_dir)
    records: list[dict[str, Any]] = []
    current_idx = start_idx
    records.extend(_fixed6_records(BUCKET_COUNTS["00_fixed6_hard"], start_idx=current_idx))
    current_idx += BUCKET_COUNTS["00_fixed6_hard"]
    records.extend(_fragile_records(BUCKET_COUNTS["10_fragile_text"], start_idx=current_idx, seed=seed))
    current_idx += BUCKET_COUNTS["10_fragile_text"]
    records.extend(_heldout_records(BUCKET_COUNTS["20_heldout_text"], start_idx=current_idx))
    current_idx += BUCKET_COUNTS["20_heldout_text"]
    records.extend(_general_records(BUCKET_COUNTS["30_general_scene"], start_idx=current_idx))
    current_idx += BUCKET_COUNTS["30_general_scene"]
    records.extend(_synthetic_records(BUCKET_COUNTS["40_synthetic_text"], start_idx=current_idx, seed=seed))

    splits = _split_records(records)
    split_files = {
        bucket: _path(output_path / f"v37_{bucket}_prompts.jsonl")
        for bucket in BUCKET_COUNTS
    }
    return {
        "records": records,
        "splits": splits,
        "surface_curriculum": _surface_curriculum(records),
        "artifact_gate_loss_config": _loss_config(),
        "manifest": _manifest(
            output_dir=output_path,
            count=count,
            seed=seed,
            records=records,
            split_files=split_files,
        ),
        "audit": _audit(records, split_files),
    }


def write_v37_10k_curriculum(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    start_idx: int = DEFAULT_START_IDX,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = build_v37_10k_curriculum(
        count=count,
        seed=seed,
        start_idx=start_idx,
        output_dir=output_path,
    )
    _write_jsonl(output_path / "v37_10k_prompt_records.jsonl", payload["records"])
    for bucket, records in payload["splits"].items():
        _write_jsonl(output_path / f"v37_{bucket}_prompts.jsonl", records)
    _write_json(output_path / "v37_10k_surface_curriculum_weights.json", payload["surface_curriculum"])
    _write_json(output_path / "v37_10k_artifact_gate_loss_config.json", payload["artifact_gate_loss_config"])
    _write_json(output_path / "v37_10k_curriculum_manifest.json", payload["manifest"])
    _write_json(output_path / "v37_10k_manifest_audit.json", payload["audit"])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the GEMMANIMA v37 10k text curriculum.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-idx", type=int, default=DEFAULT_START_IDX)
    parser.add_argument("--json", action="store_true", help="Print the manifest and audit as JSON.")
    args = parser.parse_args(argv)

    payload = write_v37_10k_curriculum(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        start_idx=args.start_idx,
    )
    summary = {
        "manifest": payload["manifest"],
        "audit": payload["audit"],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote v37 10k curriculum to {Path(args.output_dir).as_posix()}")
        print(f"Audit status: {payload['audit']['status']}")
    return 0 if payload["audit"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
