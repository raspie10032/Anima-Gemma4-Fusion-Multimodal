from __future__ import annotations

from pathlib import Path
from typing import Any

from gemmanima.training.cache_manifest import build_cache_manifest_write_command
from gemmanima.training.bridge_training import BridgeTrainingPlan
from gemmanima.training.gemma_cache import GemmaCachePlan, default_split_gemma_cache_plans
from gemmanima.training.gemma_cache import audit_cache_pairing
from gemmanima.training.readiness import DEFAULT_MANIFEST
from gemmanima.training.teacher_targets import build_cache_targets_command


DEFAULT_POC1_SUBSET = Path(r"runs\teacher_targets\poc1_1k_teacher_subset.jsonl")
DEFAULT_POC1_TARGET_DIR = Path(r"runs\cache\poc1_1k\targets")
DEFAULT_POC1_GEMMA_DIR = Path(r"runs\cache\poc1_1k\gemma")
DEFAULT_POC1_BRIDGE_OUT = Path(r"runs\cache\poc1_1k\bridge\poc1_bridge.pt")


def build_poc1_cache_plan(
    *,
    manifest: str | Path = DEFAULT_MANIFEST,
    subset: str | Path = DEFAULT_POC1_SUBSET,
    target_dir: str | Path = DEFAULT_POC1_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_POC1_GEMMA_DIR,
    limit: int = 1000,
    gpu_profile: str = "all",
) -> dict[str, Any]:
    manifest_path = Path(manifest)
    subset_path = Path(subset)
    target_root = Path(target_dir)
    gemma_root = Path(gemma_dir)
    stage = _stage_name(limit)
    teacher_manifest_out = target_root / f"{stage}_CACHE_BUILD_MANIFEST.json"
    gemma_plans = _gemma_plans_for_profile(gpu_profile)

    return {
        "stage": stage,
        "limit": limit,
        "gpu_profile": gpu_profile,
        "source_manifest": str(manifest_path),
        "subset": str(subset_path),
        "target_dir": str(target_root),
        "gemma_dir": str(gemma_root),
        "prepare_subset_command": (
            "python -m gemmanima.cli prepare-teacher-targets "
            f"--manifest \"{manifest_path}\" "
            f"--output-subset \"{subset_path}\" "
            f"--target-dir \"{target_root}\" "
            f"--limit {limit} --json"
        ),
        "teacher_target_command": build_cache_targets_command(
            subset_path=subset_path,
            outdir=target_root,
            shard=1000,
            gpu_index=0 if gpu_profile == "4070-only" else None,
        ),
        "teacher_target_manifest_command": build_cache_manifest_write_command(
            stage=stage,
            cache_kind="anima_te_conditioning",
            sample_count=limit,
            source_manifest=subset_path,
            output_dir=target_root,
            manifest_out=teacher_manifest_out,
            success_count=limit,
            shape=(1, 512, 1024),
            dtype="float16",
            device="cuda:0",
        ),
        "teacher_target_manifest_path": str(teacher_manifest_out),
        "gemma_cache_plans": [
            _gemma_plan_dict(plan, subset=subset_path, target_dir=target_root, gemma_dir=gemma_root, limit=limit)
            for plan in gemma_plans
        ],
    }


def _gemma_plans_for_profile(gpu_profile: str) -> tuple[GemmaCachePlan, ...]:
    plans = default_split_gemma_cache_plans()
    if gpu_profile == "all":
        return plans
    if gpu_profile == "4070-only":
        return tuple(plan for plan in plans if plan.gpu_index == 0)
    raise ValueError(f"unknown GPU profile: {gpu_profile}")


def _gemma_plan_dict(
    plan: GemmaCachePlan,
    *,
    subset: Path,
    target_dir: Path,
    gemma_dir: Path,
    limit: int,
) -> dict[str, Any]:
    manifest_path = gemma_dir / f"{_stage_name(limit)}_{plan.name}_CACHE_BUILD_MANIFEST.json"
    return {
        "name": plan.name,
        "gpu_index": plan.gpu_index,
        "target_patterns": list(plan.target_patterns),
        "embed_on_gpu": plan.embed_on_gpu,
        "batch_size": plan.batch_size,
        "command": plan.command(subset=subset, target_dir=target_dir, outdir=gemma_dir),
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_command": build_cache_manifest_write_command(
            stage=_stage_name(limit),
            cache_kind="gemma_text_state",
            sample_count=limit,
            source_manifest=subset,
            output_dir=gemma_dir,
            manifest_out=manifest_path,
            success_count=limit,
            shape=(1, 16, 1536),
            dtype="float32",
            device=f"cuda:{plan.gpu_index}" if plan.embed_on_gpu else "cpu",
        ),
    }


def _stage_name(limit: int) -> str:
    if limit == 1000:
        return "poc1_1k_smoke"
    if limit == 10000:
        return "poc1_10k_pilot"
    return f"poc1_{limit}_sample_plan"


def build_poc1_bridge_plan(
    *,
    target_dir: str | Path = DEFAULT_POC1_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_POC1_GEMMA_DIR,
    output: str | Path = DEFAULT_POC1_BRIDGE_OUT,
    limit_shards: int | None = 1,
) -> dict[str, Any]:
    bridge = BridgeTrainingPlan(
        target_dir=Path(target_dir),
        gemma_dir=Path(gemma_dir),
        output=Path(output),
        epochs=1,
        batch_size=2,
        accum=2,
        val=100,
        lr=5e-4,
        prefetch_gb=4.0,
        save_every_shards=1,
        limit_shards=limit_shards,
    )
    pairing = audit_cache_pairing(target_dir=target_dir, gemma_dir=gemma_dir)
    return {
        "stage": "poc1_bridge_smoke" if limit_shards else "poc1_bridge_pilot",
        "target_dir": str(Path(target_dir)),
        "gemma_dir": str(Path(gemma_dir)),
        "output": str(Path(output)),
        "limit_shards": limit_shards,
        "ready": pairing["ready_for_bridge_training"],
        "cache_pairing": pairing,
        "train_command": bridge.command(),
        "eval_command": f"python -m gemmanima.cli bridge-eval-status --checkpoint \"{Path(output)}\" --json",
        "forward_smoke_command": (
            f"E:\\ComfyUI_sage\\python_embeded\\python.exe "
            f"scripts\\smoke_hiddenstage_bridge_forward.py --checkpoint \"{Path(output)}\""
        ),
    }
