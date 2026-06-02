from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gemmanima.training.evaluation import audit_bridge_checkpoint
from gemmanima.training.gemma_cache import audit_cache_pairing


DEFAULT_RUNTIME_REPORT = Path("reports/poc1_1k_smoke_report.json")
DEFAULT_COMPARE_REPORT = Path("reports/poc1_generation_compare_report.json")
DEFAULT_POC1_10K_TARGET_DIR = Path(r"runs\cache\poc1_10k\targets")
DEFAULT_POC1_10K_GEMMA_DIR = Path(r"runs\cache\poc1_10k\gemma")
DEFAULT_POC1_10K_BRIDGE_CHECKPOINT = Path(r"runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt")


def build_poc1_status(
    *,
    runtime_report: str | Path = DEFAULT_RUNTIME_REPORT,
    compare_report: str | Path = DEFAULT_COMPARE_REPORT,
) -> dict[str, Any]:
    runtime = _read_json(runtime_report)
    compare = _read_json(compare_report)
    bridge = runtime.get("bridge_training", {})
    gemma = runtime.get("gemma_cache", {})
    image_metrics = compare.get("image_metrics") or {}
    ready = bool(bridge.get("passed_mse_gate")) and bool(gemma.get("pairing_ready"))
    return {
        "stage": "poc1_1k_smoke",
        "ready": ready,
        "runtime_report": str(Path(runtime_report)),
        "compare_report": str(Path(compare_report)),
        "bridge": {
            "final_val_mse": bridge.get("final_val_mse"),
            "passed_mse_gate": bool(bridge.get("passed_mse_gate")),
        },
        "cache": {
            "pairing_ready": bool(gemma.get("pairing_ready")),
            "target_examples": runtime.get("teacher_targets", {}).get("examples"),
            "gemma_examples": gemma.get("examples"),
        },
        "comparison": {
            "conditioning_mse": (compare.get("conditioning") or {}).get("mse"),
            "image_mse": image_metrics.get("mse"),
            "image_mae": image_metrics.get("mae"),
            "image_psnr_db": image_metrics.get("psnr_db"),
        },
        "outputs": {
            "student_image": runtime.get("real_render_smoke", {}).get("output"),
            "baseline_image": (compare.get("images") or {}).get("teacher"),
        },
    }


def build_poc1_runtime_status(
    *,
    target_dir: str | Path = DEFAULT_POC1_10K_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_POC1_10K_GEMMA_DIR,
    bridge_checkpoint: str | Path = DEFAULT_POC1_10K_BRIDGE_CHECKPOINT,
) -> dict[str, Any]:
    pairing = audit_cache_pairing(target_dir=target_dir, gemma_dir=gemma_dir)
    bridge = audit_bridge_checkpoint(bridge_checkpoint)
    return {
        "stage": "poc1_10k_pilot",
        "executes_gpu_commands": False,
        "target_dir": str(Path(target_dir)),
        "gemma_dir": str(Path(gemma_dir)),
        "bridge_checkpoint": str(Path(bridge_checkpoint)),
        "ready_for_bridge_training": pairing["ready_for_bridge_training"],
        "cache": {
            "target_shards": pairing["target_shards"],
            "gemma_shards": pairing["gemma_shards"],
            "paired_shards": pairing["paired_shards"],
            "missing_gemma_shards": pairing["missing_gemma_shards"],
            "extra_gemma_shards": pairing["extra_gemma_shards"],
            "first_missing_gemma": pairing["first_missing_gemma"],
            "first_extra_gemma": pairing["first_extra_gemma"],
        },
        "bridge": bridge,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))
