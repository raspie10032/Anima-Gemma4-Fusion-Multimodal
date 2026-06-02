from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_MANIFEST = Path(r"E:\anima_gemma_swap\dataset_manifests\hiddenstage_multimodal_planner_anima_v2.jsonl")
DEFAULT_TEXT_TRANSLATOR = Path(r"E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt")
DEFAULT_STAGE = "image_state_conditioning_v1"
DEFAULT_OUTPUT_ROOT = Path(r"runs\cache\image_state_conditioning_v1")
DEFAULT_TRAIN_SCRIPT = Path("scripts/train_image_state_translator.py")
DEFAULT_TARGET_CACHE_SCRIPT = Path(r"E:\anima_gemma_swap\scripts\core\06_cache_targets.py")
DEFAULT_EMBEDDED_PYTHON = Path(r"E:\ComfyUI_sage\python_embeded\python.exe")


def build_image_state_subset(
    *,
    source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
    output: str | Path,
    limit: int = 10_000,
    start: int = 0,
    require_image_embed: bool = True,
) -> dict[str, Any]:
    src = Path(source_manifest)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "stage": DEFAULT_STAGE,
        "source_manifest": str(src),
        "subset": str(out),
        "limit": limit,
        "start": start,
        "scanned": 0,
        "written": 0,
        "skipped_missing_prompt": 0,
        "skipped_missing_image_embed": 0,
    }
    with src.open("r", encoding="utf-8-sig") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            stats["scanned"] += 1
            if stats["scanned"] <= start:
                continue
            row = json.loads(line)
            prompt = row.get("teacher_prompt") or row.get("visible_prompt") or row.get("text")
            if not prompt:
                stats["skipped_missing_prompt"] += 1
                continue
            image_embed = row.get("image_embed_pre")
            if require_image_embed and (not image_embed or not Path(image_embed).exists()):
                stats["skipped_missing_image_embed"] += 1
                continue
            record = {
                "idx": stats["written"],
                "source_id": row.get("id"),
                "text": prompt,
                "visible_prompt": row.get("visible_prompt") or prompt,
                "teacher_prompt": prompt,
                "image": row.get("image"),
                "image_embed_pre": image_embed,
                "input_modalities": ["image", "text"],
                "source_schema": row.get("schema"),
                "rating": row.get("rating"),
                "width": row.get("width"),
                "height": row.get("height"),
                "target_tags": row.get("target_tags"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if limit and stats["written"] >= limit:
                break
    stats["ready"] = stats["written"] > 0
    return stats


def build_image_state_conditioning_plan(
    *,
    source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
    subset: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    text_translator: str | Path = DEFAULT_TEXT_TRANSLATOR,
    sample_count: int = 10_000,
    stage: str = DEFAULT_STAGE,
    target_shard: int = 1000,
    batch_size: int = 4,
    epochs: int = 1,
    lr: float = 2e-4,
) -> dict[str, Any]:
    root = Path(output_root)
    subset_path = Path(subset) if subset is not None else root / f"{stage}_subset_{sample_count}.jsonl"
    target_dir = root / "targets"
    out_checkpoint = root / "bridge" / f"{stage}_image_translator.pt"
    train_report = root / "reports" / f"{stage}_train_report.json"
    source = Path(source_manifest)
    translator = Path(text_translator)
    return {
        "stage": stage,
        "goal": "train image/multimodal state to Anima-compatible conditioning",
        "executes_gpu_commands": False,
        "source_manifest": str(source),
        "subset": str(subset_path),
        "sample_count": sample_count,
        "architecture": {
            "text_path_status": "treated_as_functional_anchor",
            "text_translator_anchor": str(translator),
            "text_translator_anchor_exists": translator.exists(),
            "source_state": "image_embed_pre tensor, typically [image_tokens, 768]",
            "target_state": "Anima llm_adapter conditioning target [t5_tokens, 1024]",
            "translator": "ImageStateToConditioningTranslator",
            "fusion_policy": "learn image-state conditioning in the same target space as the text translator",
        },
        "outputs": {
            "target_dir": str(target_dir),
            "checkpoint": str(out_checkpoint),
            "train_report": str(train_report),
        },
        "gpu_policy": {
            "required_gpu": "RTX 4070 Ti SUPER",
            "cuda_visible_devices": "0",
            "forbidden_gpu": "RTX 5060",
        },
        "commands": {
            "write_subset": (
                "python -m gemmanima.cli image-state-conditioning-subset "
                f"--source-manifest {_quote_pwsh(str(source))} "
                f"--output {_quote_pwsh(str(subset_path))} "
                f"--limit {sample_count} --json"
            ),
            "cache_targets": (
                "$env:CUDA_VISIBLE_DEVICES='0'; "
                f"& {_quote_pwsh(str(DEFAULT_EMBEDDED_PYTHON))} {_quote_pwsh(str(DEFAULT_TARGET_CACHE_SCRIPT))} "
                f"--subset {_quote_pwsh(str(subset_path))} "
                f"--outdir {_quote_pwsh(str(target_dir))} "
                f"--shard {target_shard} --resume"
            ),
            "train_image_translator": (
                "$env:CUDA_VISIBLE_DEVICES='0'; "
                "$env:GEMMA_EMBED_ON_GPU='1'; "
                f"& {_quote_pwsh(str(DEFAULT_EMBEDDED_PYTHON))} {_quote_pwsh(str(DEFAULT_TRAIN_SCRIPT))} "
                f"--subset {_quote_pwsh(str(subset_path))} "
                f"--targets {_quote_pwsh(str(target_dir))} "
                f"--out {_quote_pwsh(str(out_checkpoint))} "
                f"--text-translator-anchor {_quote_pwsh(str(translator))} "
                f"--epochs {epochs} --batch-size {batch_size} --lr {lr} "
                f"--report {_quote_pwsh(str(train_report))}"
            ),
        },
        "gates": {
            "first": "image_reference_smoke",
            "second": "text_prompt_adherence_guard",
            "third": "fixed6_no_default_regression",
            "default_promotion_allowed": False,
        },
    }


def write_image_state_conditioning_plan(*, output: str | Path, **kwargs: Any) -> Path:
    payload = build_image_state_conditioning_plan(**kwargs)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _quote_pwsh(value: str) -> str:
    return f"\"{value.replace('`', '``').replace('$', '`$').replace(chr(34), '`' + chr(34))}\""
