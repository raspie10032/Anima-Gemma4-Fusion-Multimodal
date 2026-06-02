from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmanima.core.config import EngineConfig
from gemmanima.core.protocol import ConflictReport
from gemmanima.core.schemas import ConditioningBundle, ContextCapsule, GenerationPlan
from gemmanima.training.evaluation import audit_bridge_checkpoint


@dataclass(frozen=True)
class HiddenStageBridgeAudit:
    path: Path
    exists: bool
    readable: bool
    val_mse: float | None
    passed_mse_gate: bool
    kv_key_count: int

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "HiddenStageBridgeAudit":
        data = audit_bridge_checkpoint(path)
        return cls(
            path=Path(path),
            exists=bool(data["exists"]),
            readable=bool(data["readable"]),
            val_mse=data["val_mse"],
            passed_mse_gate=bool(data["passed_mse_gate"]),
            kv_key_count=int(data.get("kv_key_count", 0)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "readable": self.readable,
            "val_mse": self.val_mse,
            "passed_mse_gate": self.passed_mse_gate,
            "kv_key_count": self.kv_key_count,
        }


class HiddenStageExit:
    """Adapter boundary for Gemma hidden-state to Anima conditioning."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.bridge_path = self.config.models.hiddenstage_bridge

    def audit_bridge(self) -> HiddenStageBridgeAudit:
        return HiddenStageBridgeAudit.from_checkpoint(self.bridge_path)

    def encode(
        self,
        capsule: ContextCapsule,
        plan: GenerationPlan,
        *,
        conflict_report: ConflictReport | None = None,
    ) -> ConditioningBundle:
        audit = self.audit_bridge()
        protocol = capsule.protocol
        conflict = conflict_report or protocol.conflict
        bundle = ConditioningBundle(
            source="trained_hiddenstage_bridge" if audit.passed_mse_gate else "dry_run_hiddenstage_exit",
            shape=(1, 512, 1024),
            metadata={
                "capsule_id": capsule.capsule_id,
                "prompt_preview": plan.prompt[:240],
                "bridge": "Gemma hidden [B,S,1536] -> Anima crossattn_emb [B,512,1024]",
                "bridge_checkpoint": audit.to_json_dict(),
            },
            semantic_conditioning={
                "scene": protocol.scene.to_json_dict(),
                "instruction": protocol.instruction.to_json_dict(),
                "prompt_preview": plan.prompt[:240],
            },
            reference_conditioning=protocol.reference.to_json_dict(),
            style_conditioning=protocol.style.to_json_dict(),
            mood_conditioning=protocol.mood.to_json_dict(),
            negative_conditioning={"constraints": list(protocol.scene.negative_constraints)},
            strength_weights=protocol.conditioning.to_json_dict(),
            conflict_state=conflict.to_json_dict(),
            renderer_profile=plan.renderer_profile,
        )
        bundle.validate()
        return bundle
