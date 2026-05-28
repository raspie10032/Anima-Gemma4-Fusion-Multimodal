from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PLANNER_OUT = Path(r"D:\Projects\training\out\hiddenstage_multimodal_planner_anima_v2")
DEFAULT_MANIFEST = Path(r"E:\anima_gemma_swap\dataset_manifests\hiddenstage_multimodal_planner_anima_v2.jsonl")
DEFAULT_TEST_MANIFEST = Path(
    r"E:\anima_gemma_swap\dataset_manifests\hiddenstage_multimodal_planner_anima_v2_test1k.jsonl"
)


@dataclass(frozen=True)
class ArtifactCheck:
    path: Path
    exists: bool
    size_bytes: int = 0

    @classmethod
    def from_path(cls, path: Path) -> "ArtifactCheck":
        return cls(path=path, exists=path.exists(), size_bytes=path.stat().st_size if path.exists() else 0)

    def to_json_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "exists": self.exists, "size_bytes": self.size_bytes}


@dataclass
class ManifestAudit:
    path: Path
    exists: bool
    rows: int = 0
    generate_rows: int = 0
    clarify_rows: int = 0
    rows_with_teacher_prompt: int = 0
    rows_with_target_crossattn: int = 0
    rows_with_image_embed: int = 0
    missing_image_embed: int = 0
    rating_counts: dict[str, int] = field(default_factory=dict)
    example_ids_missing_target: list[Any] = field(default_factory=list)

    @property
    def bridge_ready_rows(self) -> int:
        return self.rows_with_target_crossattn

    @property
    def needs_teacher_target_extraction(self) -> bool:
        return self.generate_rows > 0 and self.rows_with_target_crossattn < self.generate_rows

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "rows": self.rows,
            "generate_rows": self.generate_rows,
            "clarify_rows": self.clarify_rows,
            "rows_with_teacher_prompt": self.rows_with_teacher_prompt,
            "rows_with_target_crossattn": self.rows_with_target_crossattn,
            "rows_with_image_embed": self.rows_with_image_embed,
            "missing_image_embed": self.missing_image_embed,
            "rating_counts": self.rating_counts,
            "bridge_ready_rows": self.bridge_ready_rows,
            "needs_teacher_target_extraction": self.needs_teacher_target_extraction,
            "example_ids_missing_target": self.example_ids_missing_target,
        }


def audit_manifest(path: str | Path, *, limit: int = 0, check_image_embed_exists: bool = False) -> ManifestAudit:
    manifest_path = Path(path)
    audit = ManifestAudit(path=manifest_path, exists=manifest_path.exists())
    if not manifest_path.exists():
        return audit

    ratings: Counter[str] = Counter()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audit.rows += 1
            action = row.get("action")
            if action == "generate":
                audit.generate_rows += 1
            elif action == "clarify":
                audit.clarify_rows += 1

            if row.get("teacher_prompt"):
                audit.rows_with_teacher_prompt += 1
            if row.get("target_crossattn_emb"):
                audit.rows_with_target_crossattn += 1
            elif action == "generate" and len(audit.example_ids_missing_target) < 10:
                audit.example_ids_missing_target.append(row.get("id", row.get("idx")))

            image_embed = row.get("image_embed_pre")
            if image_embed:
                embed_path = Path(str(image_embed))
                if not check_image_embed_exists or embed_path.exists():
                    audit.rows_with_image_embed += 1
                else:
                    audit.missing_image_embed += 1

            rating = row.get("rating")
            if rating:
                ratings[str(rating)] += 1

            if limit and audit.rows >= limit:
                break
    audit.rating_counts = dict(sorted(ratings.items()))
    return audit


def build_training_readiness_report(
    *,
    planner_out: str | Path = DEFAULT_PLANNER_OUT,
    train_manifest: str | Path = DEFAULT_MANIFEST,
    eval_manifest: str | Path = DEFAULT_TEST_MANIFEST,
    manifest_limit: int = 0,
    check_image_embed_exists: bool = False,
) -> dict[str, Any]:
    planner_dir = Path(planner_out)
    artifacts = {
        "adapter_model": ArtifactCheck.from_path(planner_dir / "adapter_model.safetensors"),
        "adapter_config": ArtifactCheck.from_path(planner_dir / "adapter_config.json"),
        "vision_embedding": ArtifactCheck.from_path(planner_dir / "embed_vision.pt"),
        "eval_pass": ArtifactCheck.from_path(planner_dir / "EVAL_PASS.json"),
        "eval_checkpoints": ArtifactCheck.from_path(planner_dir / "eval_checkpoints.jsonl"),
    }
    train_audit = audit_manifest(
        train_manifest,
        limit=manifest_limit,
        check_image_embed_exists=check_image_embed_exists,
    )
    eval_audit = audit_manifest(
        eval_manifest,
        limit=manifest_limit,
        check_image_embed_exists=check_image_embed_exists,
    )
    planner_ready = all(
        artifacts[name].exists for name in ("adapter_model", "adapter_config", "vision_embedding", "eval_pass")
    )
    bridge_ready = train_audit.bridge_ready_rows > 0 and eval_audit.bridge_ready_rows > 0
    next_required = (
        "extract_teacher_crossattn_targets"
        if planner_ready and not bridge_ready
        else "train_hiddenstage_exit_bridge"
        if planner_ready and bridge_ready
        else "repair_planner_artifacts"
    )
    return {
        "planner_ready": planner_ready,
        "hiddenstage_bridge_dataset_ready": bridge_ready,
        "next_required_action": next_required,
        "planner_artifacts": {name: artifact.to_json_dict() for name, artifact in artifacts.items()},
        "train_manifest": train_audit.to_json_dict(),
        "eval_manifest": eval_audit.to_json_dict(),
        "training_targets": {
            "already_trained": ["gemma_multimodal_planner_lora"],
            "needs_training_now": ["hiddenstage_exit_bridge"] if bridge_ready else [],
            "blocked_until_targets_exist": ["hiddenstage_exit_bridge"] if not bridge_ready else [],
            "needs_code_or_extraction_first": ["teacher_crossattn_target_extraction"] if not bridge_ready else [],
        },
    }


def write_training_readiness_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
