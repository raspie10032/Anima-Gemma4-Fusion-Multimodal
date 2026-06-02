from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from gemmanima.training.cache_manifest import build_cache_manifest_write_command


TARGET_COVERAGE_MARKER = "TARGET_COVERAGE_COMPLETE.json"


@dataclass(frozen=True)
class TeacherSubsetExport:
    input_manifest: Path
    output_subset: Path
    rows_written: int
    skipped_rows: int
    command: str
    cache_manifest_path: Path
    cache_manifest_command: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "input_manifest": str(self.input_manifest),
            "output_subset": str(self.output_subset),
            "rows_written": self.rows_written,
            "skipped_rows": self.skipped_rows,
            "command": self.command,
            "cache_manifest_path": str(self.cache_manifest_path),
            "cache_manifest_command": self.cache_manifest_command,
        }


def export_teacher_subset(
    manifest_path: str | Path,
    output_subset: str | Path,
    *,
    limit: int = 0,
    text_key: str = "teacher_prompt",
    target_dir: str | Path = r"E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets",
) -> TeacherSubsetExport:
    src = Path(manifest_path)
    dst = Path(output_subset)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped = 0
    with src.open("r", encoding="utf-8") as handle, dst.open("w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("action") != "generate":
                skipped += 1
                continue
            text = str(row.get(text_key) or row.get("visible_prompt") or "").strip()
            if not text:
                skipped += 1
                continue
            idx = row.get("idx", row.get("id"))
            if idx is None:
                skipped += 1
                continue
            out.write(
                json.dumps(
                    {
                        "idx": int(idx),
                        "id": row.get("id"),
                        "text": text,
                        "source_manifest": str(src),
                        "rating": row.get("rating"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows_written += 1
            if limit and rows_written >= limit:
                break

    target_root = Path(target_dir)
    command = build_cache_targets_command(subset_path=dst, outdir=target_root)
    cache_manifest_path = target_root / "CACHE_BUILD_MANIFEST.json"
    cache_manifest_command = build_cache_manifest_write_command(
        cache_kind="anima_te_conditioning",
        sample_count=rows_written,
        source_manifest=dst,
        output_dir=target_root,
        manifest_out=cache_manifest_path,
        success_count=rows_written,
        shape=(1, 512, 1024),
        dtype="float16",
        device="cuda:0",
    )
    return TeacherSubsetExport(
        input_manifest=src,
        output_subset=dst,
        rows_written=rows_written,
        skipped_rows=skipped,
        command=command,
        cache_manifest_path=cache_manifest_path,
        cache_manifest_command=cache_manifest_command,
    )


def build_cache_targets_command(
    *,
    subset_path: str | Path,
    outdir: str | Path = r"E:\anima_gemma_swap\cache_hiddenstage_planner_v2\targets",
    shard: int = 2000,
    python_exe: str | Path = r"E:\ComfyUI_sage\python_embeded\python.exe",
    gpu_index: int | None = None,
) -> str:
    prefix = "" if gpu_index is None else f"$env:CUDA_VISIBLE_DEVICES='{gpu_index}'; "
    script = Path(r"E:\anima_gemma_swap\scripts\core\06_cache_targets.py")
    return (
        f"{prefix}& \"{Path(python_exe)}\" \"{script}\" "
        f"--subset \"{Path(subset_path)}\" "
        f"--outdir \"{Path(outdir)}\" "
        f"--shard {shard} --resume"
    )


def audit_target_cache(target_dir: str | Path) -> dict[str, Any]:
    root = Path(target_dir)
    files = sorted(root.glob("*.pt")) if root.exists() else []
    gpu0 = [p for p in files if not p.name.startswith("shard_5060_")]
    gpu1 = [p for p in files if p.name.startswith("shard_5060_")]
    return {
        "target_dir": str(root),
        "exists": root.exists(),
        "shard_count": len(files),
        "4070_ti_super_shards": len(gpu0),
        "5060_shards": len(gpu1),
        "first_shard": str(files[0]) if files else None,
        "last_shard": str(files[-1]) if files else None,
    }


def expected_split_shards(total_rows: int = 193258, split_ratio: float = 0.70, shard_size: int = 2000) -> dict[str, int]:
    gpu0_rows = int(total_rows * split_ratio)
    gpu1_rows = total_rows - gpu0_rows
    return {
        "total_rows": total_rows,
        "4070_ti_super_rows": gpu0_rows,
        "5060_rows": gpu1_rows,
        "4070_ti_super_shards": math.ceil(gpu0_rows / shard_size),
        "5060_shards": math.ceil(gpu1_rows / shard_size),
        "total_shards": math.ceil(gpu0_rows / shard_size) + math.ceil(gpu1_rows / shard_size),
    }


def audit_split_target_completion(
    target_dir: str | Path,
    *,
    total_rows: int = 193258,
    split_ratio: float = 0.70,
    shard_size: int = 2000,
) -> dict[str, Any]:
    audit = audit_target_cache(target_dir)
    marker_path = Path(target_dir) / TARGET_COVERAGE_MARKER
    coverage_marker_exists = marker_path.exists()
    expected = expected_split_shards(total_rows=total_rows, split_ratio=split_ratio, shard_size=shard_size)
    complete_4070 = audit["4070_ti_super_shards"] >= expected["4070_ti_super_shards"]
    complete_5060 = audit["5060_shards"] >= expected["5060_shards"]
    complete_total = audit["shard_count"] >= expected["total_shards"]
    return {
        **audit,
        "expected": expected,
        "coverage_marker": str(marker_path),
        "coverage_marker_exists": coverage_marker_exists,
        "complete_4070_ti_super": complete_4070,
        "complete_5060": complete_5060,
        "complete_total": complete_total,
        "complete": coverage_marker_exists or complete_total or (complete_4070 and complete_5060),
    }
