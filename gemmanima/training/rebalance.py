from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SHARD_SIZE = 2000
DEFAULT_4070_SUBSET = Path(
    r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\teacher_targets\hiddenstage_multimodal_planner_anima_v2_teacher_subset_4070ti_super_70p.jsonl"
)
DEFAULT_REBALANCE_DIR = Path(
    r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\teacher_targets\rebalance"
)


@dataclass(frozen=True)
class RebalancePlan:
    completed_4070_shards: int
    remaining_rows: int
    rows_for_4070: int
    rows_for_5060: int
    subset_4070: Path
    subset_5060: Path

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "completed_4070_shards": self.completed_4070_shards,
            "remaining_rows": self.remaining_rows,
            "rows_for_4070": self.rows_for_4070,
            "rows_for_5060": self.rows_for_5060,
            "subset_4070": str(self.subset_4070),
            "subset_5060": str(self.subset_5060),
        }


def build_rebalance_subsets(
    *,
    completed_4070_shards: int,
    source_subset: str | Path = DEFAULT_4070_SUBSET,
    output_dir: str | Path = DEFAULT_REBALANCE_DIR,
    share_4070: float = 0.56,
) -> RebalancePlan:
    src = Path(source_subset)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = src.read_text(encoding="utf-8").splitlines()
    start = min(len(lines), completed_4070_shards * SHARD_SIZE)
    remaining = lines[start:]
    split = int(len(remaining) * share_4070)
    rows_4070 = remaining[:split]
    rows_5060 = remaining[split:]

    subset_4070 = out / "remaining_for_4070_ti_super.jsonl"
    subset_5060 = out / "remaining_for_5060.jsonl"
    subset_4070.write_text("\n".join(rows_4070) + ("\n" if rows_4070 else ""), encoding="utf-8")
    subset_5060.write_text("\n".join(rows_5060) + ("\n" if rows_5060 else ""), encoding="utf-8")

    return RebalancePlan(
        completed_4070_shards=completed_4070_shards,
        remaining_rows=len(remaining),
        rows_for_4070=len(rows_4070),
        rows_for_5060=len(rows_5060),
        subset_4070=subset_4070,
        subset_5060=subset_5060,
    )
