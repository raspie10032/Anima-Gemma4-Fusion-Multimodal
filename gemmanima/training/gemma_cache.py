from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TARGET_DIR = Path(r"E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets")
DEFAULT_GEMMA_DIR = Path(r"D:\anima_gemma_swap_cache_hiddenstage_planner_v2\gemma")
DEFAULT_SUBSET = Path(r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset.jsonl")
DEFAULT_PYTHON = Path(r"E:\ComfyUI_sage\python_embeded\python.exe")
DEFAULT_SCRIPT = Path(r"E:\anima_gemma_swap\scripts\core\07_cache_gemma_batched.py")


@dataclass(frozen=True)
class GemmaCachePlan:
    name: str
    gpu_index: int
    target_patterns: tuple[str, ...]
    embed_on_gpu: bool
    batch_size: int

    def command(
        self,
        *,
        subset: str | Path = DEFAULT_SUBSET,
        target_dir: str | Path = DEFAULT_TARGET_DIR,
        outdir: str | Path = DEFAULT_GEMMA_DIR,
        python_exe: str | Path = DEFAULT_PYTHON,
        script: str | Path = DEFAULT_SCRIPT,
    ) -> str:
        patterns = ",".join(self.target_patterns)
        embed = "1" if self.embed_on_gpu else "0"
        return (
            f"$env:CUDA_VISIBLE_DEVICES='{self.gpu_index}'; "
            f"$env:GEMMA_EMBED_ON_GPU='{embed}'; "
            f"\"{Path(python_exe)}\" \"{Path(script)}\" "
            f"--subset \"{Path(subset)}\" "
            f"--target-dir \"{Path(target_dir)}\" "
            f"--outdir \"{Path(outdir)}\" "
            f"--patterns \"{patterns}\" "
            f"--batch-size {self.batch_size} --resume"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gpu_index": self.gpu_index,
            "target_patterns": list(self.target_patterns),
            "embed_on_gpu": self.embed_on_gpu,
            "batch_size": self.batch_size,
            "command": self.command(),
        }


def default_split_gemma_cache_plans() -> tuple[GemmaCachePlan, ...]:
    return (
        GemmaCachePlan(
            name="gemma_4070_ti_super",
            gpu_index=0,
            target_patterns=(
                "shard_[0-9][0-9][0-9][0-9].pt",
                "shard_re4070_*.pt",
                "shard_4070w2*.pt",
                "shard_missing4070_*.pt",
            ),
            embed_on_gpu=True,
            batch_size=8,
        ),
        GemmaCachePlan(
            name="gemma_5060",
            gpu_index=1,
            target_patterns=("shard_5060_*.pt",),
            embed_on_gpu=False,
            batch_size=8,
        ),
    )


def audit_cache_pairing(
    *,
    target_dir: str | Path = DEFAULT_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_GEMMA_DIR,
) -> dict[str, Any]:
    target_root = Path(target_dir)
    gemma_root = Path(gemma_dir)
    target_files = sorted(p.name for p in target_root.glob("*.pt")) if target_root.exists() else []
    gemma_files = sorted(p.name for p in gemma_root.glob("*.pt")) if gemma_root.exists() else []
    target_set = set(target_files)
    gemma_set = set(gemma_files)
    missing_gemma = sorted(target_set - gemma_set)
    extra_gemma = sorted(gemma_set - target_set)
    return {
        "target_dir": str(target_root),
        "gemma_dir": str(gemma_root),
        "target_shards": len(target_files),
        "gemma_shards": len(gemma_files),
        "paired_shards": len(target_set & gemma_set),
        "missing_gemma_shards": len(missing_gemma),
        "extra_gemma_shards": len(extra_gemma),
        "first_missing_gemma": missing_gemma[:10],
        "first_extra_gemma": extra_gemma[:10],
        "ready_for_bridge_training": bool(target_files) and not missing_gemma,
    }
