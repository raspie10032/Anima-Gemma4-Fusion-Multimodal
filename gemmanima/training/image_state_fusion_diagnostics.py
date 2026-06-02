from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageStat

DEFAULT_TRAIN_SCRIPT = Path("scripts/train_image_state_translator.py")
DEFAULT_EMBEDDED_PYTHON = Path(r"E:\ComfyUI_sage\python_embeded\python.exe")
DEFAULT_TEXT_TRANSLATOR = Path(r"E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt")
DEFAULT_CURRENT_IMAGE_CHECKPOINT = Path(
    r"runs\cache\image_state_conditioning_v2_full\bridge\image_state_conditioning_v2_full_image_translator.pt"
)


def build_conditioning_fusion_guard_manifest(
    *,
    sweep_report: str | Path,
    subset: str | Path,
    failed_indices: Iterable[int],
    guard_reason: str = "conditioning_fusion_noise_or_instability",
) -> dict[str, Any]:
    report_path = Path(sweep_report)
    subset_path = Path(subset)
    failed = {int(idx) for idx in failed_indices}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_subset_rows(subset_path)
    failed_cases = []
    passed_cases = []
    replay_rows = []
    for output in report.get("outputs", []):
        idx = int(output["idx"])
        row = rows.get(idx, {})
        case = {
            "idx": idx,
            "mode": output.get("mode"),
            "seed": output.get("seed"),
            "output": output.get("output"),
            "source_id": row.get("source_id"),
            "rating": row.get("rating"),
            "prompt": row.get("visible_prompt") or row.get("teacher_prompt") or row.get("text"),
            "image_embed_pre": row.get("image_embed_pre"),
            "image_state": _tensor_file_stats(row.get("image_embed_pre")),
            "output_image": _image_file_stats(output.get("output")),
        }
        if idx in failed:
            case["guard_reason"] = guard_reason
            failed_cases.append(case)
            replay = dict(row)
            replay.update(
                {
                    "guard_reason": guard_reason,
                    "guard_source_report": str(report_path),
                    "guard_output": output.get("output"),
                    "guard_seed": output.get("seed"),
                    "guard_mode": output.get("mode"),
                }
            )
            replay_rows.append(replay)
        else:
            passed_cases.append(case)
    return {
        "stage": "conditioning_fusion_guard_manifest",
        "sweep_report": str(report_path),
        "subset": str(subset_path),
        "failed_indices": sorted(failed),
        "failed_count": len(failed_cases),
        "passed_count": len(passed_cases),
        "guard_reason": guard_reason,
        "failed_cases": failed_cases,
        "passed_cases": passed_cases,
        "replay_rows": replay_rows,
        "next_training_use": "oversample replay_rows and require no regression on their conditioning_fusion renders",
    }


def write_conditioning_fusion_guard_manifest(
    *,
    output: str | Path,
    replay_output: str | Path,
    sweep_report: str | Path,
    subset: str | Path,
    failed_indices: Iterable[int],
    guard_reason: str = "conditioning_fusion_noise_or_instability",
) -> dict[str, Any]:
    manifest = build_conditioning_fusion_guard_manifest(
        sweep_report=sweep_report,
        subset=subset,
        failed_indices=failed_indices,
        guard_reason=guard_reason,
    )
    output_path = Path(output)
    replay_path = Path(replay_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(manifest)
    serializable["replay_rows"] = manifest["replay_rows"]
    output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    with replay_path.open("w", encoding="utf-8") as handle:
        for row in manifest["replay_rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "guard_manifest": str(output_path),
        "replay_manifest": str(replay_path),
        "failed_count": manifest["failed_count"],
        "passed_count": manifest["passed_count"],
    }


def build_conditioning_fusion_preflight_manifest(
    *,
    fusion_report: str | Path,
    text_only_report: str | Path,
    subset: str | Path,
    fusion_failed_indices: Iterable[int],
    text_only_failed_indices: Iterable[int],
) -> dict[str, Any]:
    fusion_path = Path(fusion_report)
    text_only_path = Path(text_only_report)
    subset_path = Path(subset)
    fusion_failed = {int(idx) for idx in fusion_failed_indices}
    text_failed = {int(idx) for idx in text_only_failed_indices}
    rows = _load_subset_rows(subset_path)
    fusion_outputs = _outputs_by_idx(fusion_path)
    text_outputs = _outputs_by_idx(text_only_path)

    text_conditioning_failed = []
    image_conditioning_failed = []
    passed = []
    image_replay_rows = []
    all_indices = sorted(set(fusion_outputs) | set(text_outputs) | fusion_failed | text_failed)
    for idx in all_indices:
        row = rows.get(idx, {})
        case = {
            "idx": idx,
            "source_id": row.get("source_id"),
            "rating": row.get("rating"),
            "prompt": row.get("visible_prompt") or row.get("teacher_prompt") or row.get("text"),
            "image_embed_pre": row.get("image_embed_pre"),
            "fusion_failed": idx in fusion_failed,
            "text_only_failed": idx in text_failed,
            "fusion_output": fusion_outputs.get(idx),
            "text_only_output": text_outputs.get(idx),
        }
        if idx in text_failed:
            case["failure_class"] = "text_conditioning_failed"
            text_conditioning_failed.append(case)
            continue
        if idx in fusion_failed:
            case["failure_class"] = "image_conditioning_failed_text_only_passed"
            image_conditioning_failed.append(case)
            replay = dict(row)
            replay.update(
                {
                    "guard_reason": "image_conditioning_failed_text_only_passed",
                    "guard_source_report": str(fusion_path),
                    "text_only_preflight_report": str(text_only_path),
                    "guard_output": case["fusion_output"].get("output") if case["fusion_output"] else None,
                    "guard_seed": case["fusion_output"].get("seed") if case["fusion_output"] else None,
                    "guard_mode": case["fusion_output"].get("mode") if case["fusion_output"] else None,
                }
            )
            image_replay_rows.append(replay)
            continue
        case["failure_class"] = "passed"
        passed.append(case)

    return {
        "stage": "conditioning_fusion_text_only_preflight",
        "fusion_report": str(fusion_path),
        "text_only_report": str(text_only_path),
        "subset": str(subset_path),
        "fusion_failed_indices": sorted(fusion_failed),
        "text_only_failed_indices": sorted(text_failed),
        "text_conditioning_failed_count": len(text_conditioning_failed),
        "image_conditioning_failed_count": len(image_conditioning_failed),
        "passed_count": len(passed),
        "text_conditioning_failed": text_conditioning_failed,
        "image_conditioning_failed": image_conditioning_failed,
        "passed": passed,
        "image_replay_rows": image_replay_rows,
        "next_training_use": "only oversample image_replay_rows; text_conditioning_failed rows need text-path diagnosis first",
    }


def write_conditioning_fusion_preflight_manifest(
    *,
    output: str | Path,
    image_replay_output: str | Path,
    fusion_report: str | Path,
    text_only_report: str | Path,
    subset: str | Path,
    fusion_failed_indices: Iterable[int],
    text_only_failed_indices: Iterable[int],
) -> dict[str, Any]:
    manifest = build_conditioning_fusion_preflight_manifest(
        fusion_report=fusion_report,
        text_only_report=text_only_report,
        subset=subset,
        fusion_failed_indices=fusion_failed_indices,
        text_only_failed_indices=text_only_failed_indices,
    )
    output_path = Path(output)
    replay_path = Path(image_replay_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with replay_path.open("w", encoding="utf-8") as handle:
        for row in manifest["image_replay_rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "preflight_manifest": str(output_path),
        "image_replay_manifest": str(replay_path),
        "text_conditioning_failed_count": manifest["text_conditioning_failed_count"],
        "image_conditioning_failed_count": manifest["image_conditioning_failed_count"],
        "passed_count": manifest["passed_count"],
    }


def build_image_state_replay_training_objective(
    *,
    base_subset: str | Path,
    guard_replay: str | Path,
    current_checkpoint: str | Path = DEFAULT_CURRENT_IMAGE_CHECKPOINT,
    target_dir: str | Path = r"runs\cache\image_state_conditioning_v2_full\targets",
    output_root: str | Path = r"runs\cache\image_state_conditioning_v3_guarded",
    text_translator: str | Path = DEFAULT_TEXT_TRANSLATOR,
    stage: str = "image_state_conditioning_v3_guarded",
    replay_weight: int = 12,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-4,
    image_cache_gb: float = 56.0,
) -> dict[str, Any]:
    subset_path = Path(base_subset)
    replay_path = Path(guard_replay)
    root = Path(output_root)
    candidate_checkpoint = root / "bridge" / f"{stage}_image_translator.pt"
    train_report = root / "reports" / f"{stage}_train_report.json"
    guard_count = _count_jsonl(replay_path)
    base_count = _count_jsonl(subset_path)
    return {
        "stage": stage,
        "objective": "stabilize conditioning_fusion image/text runtime while preserving the working v2_full translator",
        "executes_gpu_commands": False,
        "base_subset": str(subset_path),
        "base_count": base_count,
        "guard_replay": str(replay_path),
        "current_checkpoint": str(current_checkpoint),
        "training_policy": {
            "method": "guard_replay_oversampling",
            "guard_replay_count": guard_count,
            "replay_weight": replay_weight,
            "effective_guard_examples_per_epoch": guard_count * replay_weight,
            "base_sampling": "use existing full subset and target cache; inject guard rows with higher sampling weight",
            "initialization": "start from current v2_full image translator weights",
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_cache_gb": image_cache_gb,
        },
        "outputs": {
            "candidate_checkpoint": str(candidate_checkpoint),
            "train_report": str(train_report),
        },
        "gpu_policy": {
            "required_gpu": "RTX 4070 Ti SUPER",
            "cuda_visible_devices": "0",
            "forbidden_gpu": "RTX 5060",
        },
        "promotion_gates": {
            "default_protected": True,
            "conditioning_fusion_guard": {
                "required": True,
                "guard_replay": str(replay_path),
                "fail_on_noise_collapse": True,
                "must_rerender_modes": ["conditioning_fusion"],
                "must_include_seed": [950001],
            },
            "safe_sweep_regression": {
                "required": True,
                "baseline_reports": [
                    r"reports\image_state_conditioning_v2_full\conditioning_fusion_safe4_sweep_report.json",
                    r"reports\image_state_conditioning_v2_full\conditioning_fusion_safe12_b_sweep_report.json",
                ],
                "minimum_coherent_tally": "must not regress below current 15 coherent / 16 safe-rated renders",
            },
            "hidden_fusion": {
                "status": "not_promotable_until_learned_projector_exists",
            },
        },
        "commands": {
            "train_candidate": (
                "$env:CUDA_VISIBLE_DEVICES='0'; "
                "$env:GEMMA_EMBED_ON_GPU='1'; "
                f"& {_quote_pwsh(str(DEFAULT_EMBEDDED_PYTHON))} {_quote_pwsh(str(DEFAULT_TRAIN_SCRIPT))} "
                f"--subset {_quote_pwsh(str(subset_path))} "
                f"--targets {_quote_pwsh(str(target_dir))} "
                f"--out {_quote_pwsh(str(candidate_checkpoint))} "
                f"--text-translator-anchor {_quote_pwsh(str(text_translator))} "
                f"--init-checkpoint {_quote_pwsh(str(current_checkpoint))} "
                f"--guard-replay {_quote_pwsh(str(replay_path))} "
                f"--guard-replay-weight {replay_weight} "
                f"--epochs {epochs} --batch-size {batch_size} --lr {lr} "
                f"--image-cache-gb {image_cache_gb} "
                f"--report {_quote_pwsh(str(train_report))}"
            ),
            "guard_eval": (
                "$env:CUDA_VISIBLE_DEVICES='0'; "
                "$env:GEMMA_EMBED_ON_GPU='1'; "
                f"& {_quote_pwsh(str(DEFAULT_EMBEDDED_PYTHON))} scripts\\render_image_text_fusion_poc.py "
                f"--indices 0 --seeds 950001 --modes conditioning_fusion "
                f"--image-checkpoint {_quote_pwsh(str(candidate_checkpoint))} "
                "--size 512 --steps 12 --cfg 4.5 --unet-dtype fp8_e4m3fn_fast"
            ),
        },
        "next_step": "run guarded candidate training, then rerender conditioning_fusion guard and safe sweep before any promotion",
    }


def write_image_state_replay_training_objective(
    *,
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_image_state_replay_training_objective(**kwargs)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"objective_manifest": str(path), "stage": payload["stage"]}


def _load_subset_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["idx"])] = row
    return rows


def _outputs_by_idx(path: Path) -> dict[int, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    outputs = {}
    for output in report.get("outputs", []):
        outputs[int(output["idx"])] = output
    return outputs


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _quote_pwsh(value: str) -> str:
    return f"\"{value.replace('`', '``').replace('$', '`$').replace(chr(34), '`' + chr(34))}\""


def _tensor_file_stats(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"exists": False}
    path = Path(path_value)
    if not path.exists():
        return {"path": str(path), "exists": False}
    tensor = torch.load(path, map_location="cpu", weights_only=False).float()
    return {
        "path": str(path),
        "exists": True,
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "absmax": float(tensor.abs().max().item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def _image_file_stats(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"exists": False}
    path = Path(path_value)
    if not path.exists():
        return {"path": str(path), "exists": False}
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        return {
            "path": str(path),
            "exists": True,
            "size": list(rgb.size),
            "mean_rgb": [float(v) for v in stat.mean],
            "stddev_rgb": [float(v) for v in stat.stddev],
        }
