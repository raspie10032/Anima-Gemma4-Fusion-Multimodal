from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROTECTED_BASELINE = "v5"
DEFAULT_BASELINE_CHECKPOINT = Path("runs/cache/text_preservation_blended_v5/bridge/text_preservation_blended_v5_bridge.pt")
DEFAULT_REPORT_ROOT = Path("reports/text_rendering_qwen_baseline")


def build_candidate_workflow_status(
    *,
    candidate_name: str,
    checkpoint: str | Path,
    fixed6_summary: str | Path | None = None,
    general_quality_report: str | Path | None = None,
    smoke_report: str | Path | None = None,
    protected_baseline: str = DEFAULT_PROTECTED_BASELINE,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    fixed6 = _read_optional_json(fixed6_summary)
    general = _read_optional_json(general_quality_report)
    smoke = _read_optional_json(smoke_report)
    fixed6_gate = _fixed6_gate(fixed6, fixed6_summary)
    general_gate = _general_quality_gate(general, general_quality_report)
    smoke_gate = _smoke_gate(smoke, smoke_report)
    decision = _decision_for(fixed6_gate=fixed6_gate, general_gate=general_gate)
    return {
        "stage": "candidate_workflow_status",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate": {
            "name": candidate_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
        },
        "safety": {
            "protected_baseline": protected_baseline,
            "baseline_checkpoint": str(Path(baseline_checkpoint)),
            "baseline_checkpoint_exists": Path(baseline_checkpoint).exists(),
            "v5_default_changed": False,
            "gpu_policy": "4070_only",
            "cuda_visible_devices": "0",
            "forbidden_gpu": "RTX 5060",
        },
        "workflow_position": {
            "current": decision["workflow_position"],
            "next": decision["next_step"],
            "promotion_allowed": decision["promotion_allowed"],
            "mode": decision["mode"],
        },
        "gates": {
            "smoke": smoke_gate,
            "fixed6": fixed6_gate,
            "general_quality": general_gate,
        },
        "commands": _candidate_commands(candidate_name=candidate_name, checkpoint=checkpoint_path),
        "next_actions": decision["next_actions"],
    }


def write_candidate_workflow_status(
    *,
    output: str | Path,
    candidate_name: str,
    checkpoint: str | Path,
    fixed6_summary: str | Path | None = None,
    general_quality_report: str | Path | None = None,
    smoke_report: str | Path | None = None,
    protected_baseline: str = DEFAULT_PROTECTED_BASELINE,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
) -> Path:
    payload = build_candidate_workflow_status(
        candidate_name=candidate_name,
        checkpoint=checkpoint,
        fixed6_summary=fixed6_summary,
        general_quality_report=general_quality_report,
        smoke_report=smoke_report,
        protected_baseline=protected_baseline,
        baseline_checkpoint=baseline_checkpoint,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_candidate_objective_manifest(
    *,
    candidate_name: str,
    donor_checkpoint: str | Path,
    fixed6_summary: str | Path,
    protected_baseline: str = DEFAULT_PROTECTED_BASELINE,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    target_sample_count: int = 10_000,
) -> dict[str, Any]:
    fixed6 = _read_required_json(fixed6_summary)
    regression_rows = list(fixed6.get("regressions") or [])
    if not regression_rows:
        regression_rows = [
            row for row in fixed6.get("cases", []) if float(row.get("delta_vs_v5") or 0.0) > 0.0
        ]
    replay_cases = [_replay_case(row) for row in regression_rows]
    total_replay_weight = sum(case["replay_weight"] for case in replay_cases)
    return {
        "stage": "candidate_objective_manifest",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate": candidate_name,
        "source_fixed6_summary": str(Path(fixed6_summary)),
        "objective": {
            "name": f"{candidate_name}_fixed6_protected_hybrid",
            "intent": "retain image-quality/style gains from donor checkpoint while repairing protected fixed6 text regressions",
            "mode": "hybrid_repair",
            "target_sample_count": target_sample_count,
        },
        "lineage": {
            "style_donor_checkpoint": str(Path(donor_checkpoint)),
            "style_donor_exists": Path(donor_checkpoint).exists(),
            "protected_baseline": protected_baseline,
            "baseline_checkpoint": str(Path(baseline_checkpoint)),
            "baseline_checkpoint_exists": Path(baseline_checkpoint).exists(),
        },
        "fixed6_replay": {
            "source_status": fixed6.get("status"),
            "regression_count": len(replay_cases),
            "total_replay_weight": total_replay_weight,
            "cases": replay_cases,
        },
        "loss_policy": {
            "hard_preconditions": [
                "fixed6_per_case_delta_vs_v5_must_be_lte_0",
                "v5_default_must_not_change_during_training",
                "gpu_work_must_use_cuda_visible_devices_0",
            ],
            "donor_use": "initialize_or_regularize_style_only; never promote donor directly",
            "protected_terms": [
                "fixed6_text_roi_replay",
                "v5_baseline_distillation",
                "general_scene_prompt_adherence",
            ],
        },
        "next_gate": {
            "first": "fixed6",
            "expand_to_heldout": False,
            "expand_to_general": False,
            "promotion_allowed": False,
            "reason": "fixed6 regressions must be repaired before broader gates",
        },
    }


def write_candidate_objective_manifest(
    *,
    output: str | Path,
    candidate_name: str,
    donor_checkpoint: str | Path,
    fixed6_summary: str | Path,
    protected_baseline: str = DEFAULT_PROTECTED_BASELINE,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    target_sample_count: int = 10_000,
) -> Path:
    payload = build_candidate_objective_manifest(
        candidate_name=candidate_name,
        donor_checkpoint=donor_checkpoint,
        fixed6_summary=fixed6_summary,
        protected_baseline=protected_baseline,
        baseline_checkpoint=baseline_checkpoint,
        target_sample_count=target_sample_count,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_candidate_promotion_bundle(
    *,
    workflow_status: str | Path,
    candidate_name: str | None = None,
    protected_baseline: str = DEFAULT_PROTECTED_BASELINE,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
) -> dict[str, Any]:
    status = _read_required_json(workflow_status)
    candidate = candidate_name or (status.get("candidate") or {}).get("name")
    fixed6 = (status.get("gates") or {}).get("fixed6") or {}
    general = (status.get("gates") or {}).get("general_quality") or {}
    smoke = (status.get("gates") or {}).get("smoke") or {}
    fixed6_passed = fixed6.get("status") not in {"missing", "rejected"} and int(fixed6.get("regression_count") or 0) == 0
    required_artifacts = [
        _artifact("workflow_status", workflow_status, True),
        _artifact("fixed6_summary", fixed6.get("source"), True),
        _artifact("smoke_report", smoke.get("source"), True),
        _artifact("general_quality_report", general.get("source"), False),
        _artifact("general_quality_contact_sheet", general.get("contact_sheet"), False),
    ]
    missing_required = [
        item["name"] for item in required_artifacts if item["required"] and not item["exists"]
    ]
    allowed = fixed6_passed and not missing_required
    return {
        "stage": "candidate_promotion_bundle",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate": candidate,
        "source_workflow_status": str(Path(workflow_status)),
        "decision": {
            "status": "eligible_for_manual_default_update" if allowed else "blocked",
            "default_update_allowed": allowed,
            "protected_baseline": protected_baseline,
            "reason": "all required gates passed" if allowed else _promotion_block_reason(fixed6, missing_required),
        },
        "required_artifacts": required_artifacts,
        "gate_summary": {
            "fixed6_status": fixed6.get("status"),
            "fixed6_regression_count": fixed6.get("regression_count"),
            "fixed6_mean_delta_vs_baseline": fixed6.get("mean_delta_vs_baseline"),
            "general_quality_status": general.get("status"),
            "smoke_status": smoke.get("status"),
        },
        "default_change_package": {
            "current_default": protected_baseline,
            "rollback_checkpoint": str(Path(baseline_checkpoint)),
            "rollback_checkpoint_exists": Path(baseline_checkpoint).exists(),
            "write_defaults": allowed,
            "requires_manual_review": True,
        },
    }


def write_candidate_promotion_bundle(
    *,
    output: str | Path,
    workflow_status: str | Path,
    candidate_name: str | None = None,
    protected_baseline: str = DEFAULT_PROTECTED_BASELINE,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
) -> Path:
    payload = build_candidate_promotion_bundle(
        workflow_status=workflow_status,
        candidate_name=candidate_name,
        protected_baseline=protected_baseline,
        baseline_checkpoint=baseline_checkpoint,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fixed6_gate(data: dict[str, Any] | None, source: str | Path | None) -> dict[str, Any]:
    if data is None:
        return {
            "status": "missing",
            "source": _path_or_none(source),
            "required_before_promotion": True,
        }
    regressions = list(data.get("regressions") or [])
    cases = list(data.get("cases") or [])
    worst = None
    if cases:
        worst = max(cases, key=lambda item: float(item.get("delta_vs_v5") or 0.0))
    return {
        "status": data.get("status") or ("rejected" if regressions else "unknown"),
        "source": _path_or_none(source),
        "protected_baseline": data.get("protected_baseline"),
        "promotion_allowed": bool(data.get("promotion_allowed")),
        "gate_rule": data.get("gate_rule"),
        "mean_baseline_mse": data.get("mean_v5_mse"),
        "mean_candidate_mse": data.get("mean_candidate_mse"),
        "mean_delta_vs_baseline": data.get("mean_delta_vs_v5"),
        "case_count": len(cases),
        "regression_count": int(data.get("regression_count") or len(regressions)),
        "worst_regression": worst,
    }


def _general_quality_gate(data: dict[str, Any] | None, source: str | Path | None) -> dict[str, Any]:
    if data is None:
        return {
            "status": "missing",
            "source": _path_or_none(source),
            "required_before_promotion": False,
        }
    cases = list(data.get("cases") or [])
    pair_mse_values = [
        float(case["pair_mse_not_quality_score"])
        for case in cases
        if case.get("pair_mse_not_quality_score") is not None
    ]
    contact_sheet = data.get("contact_sheet")
    return {
        "status": "review_ready" if cases else "empty",
        "source": _path_or_none(source),
        "count": data.get("count") or len(cases),
        "contact_sheet": contact_sheet,
        "contact_sheet_exists": Path(contact_sheet).exists() if contact_sheet else False,
        "pair_mse_mean_not_quality_score": (
            sum(pair_mse_values) / len(pair_mse_values) if pair_mse_values else None
        ),
        "purpose": data.get("purpose"),
    }


def _smoke_gate(data: dict[str, Any] | None, source: str | Path | None) -> dict[str, Any]:
    if data is None:
        return {
            "status": "missing",
            "source": _path_or_none(source),
            "required_before_promotion": True,
        }
    output = data.get("output") or data.get("output_path") or (data.get("real_render_smoke") or {}).get("output")
    return {
        "status": "completed" if output else "observed",
        "source": _path_or_none(source),
        "output": output,
        "output_exists": Path(output).exists() if output else None,
    }


def _decision_for(*, fixed6_gate: dict[str, Any], general_gate: dict[str, Any]) -> dict[str, Any]:
    fixed6_status = fixed6_gate["status"]
    if fixed6_status == "missing":
        return {
            "workflow_position": "adapter_registered_needs_fixed6",
            "next_step": "run_fixed6_gate",
            "promotion_allowed": False,
            "mode": "candidate_intake",
            "next_actions": [
                "run fixed6 Qwen-vs-Gemma comparison before any default change",
                "keep protected v5 as the runtime baseline",
            ],
        }
    if fixed6_status == "rejected" or fixed6_gate.get("regression_count", 0) > 0:
        return {
            "workflow_position": "concept_proof_style_donor",
            "next_step": "build_hybrid_or_repair_candidate",
            "promotion_allowed": False,
            "mode": "concept_proof_only",
            "next_actions": [
                "reuse this checkpoint as an image-quality/style donor, not as a direct default",
                "train a fixed6-protected hybrid candidate before heldout/general expansion",
                "preserve v5 as default until fixed6 has zero per-case regressions",
            ],
        }
    if general_gate["status"] == "missing":
        return {
            "workflow_position": "fixed6_passed_needs_general_quality",
            "next_step": "run_general_quality_review",
            "promotion_allowed": False,
            "mode": "candidate_gate",
            "next_actions": [
                "run general-scene comparison and manual contact-sheet review",
                "then run heldout checks before changing defaults",
            ],
        }
    return {
        "workflow_position": "needs_manual_promotion_review",
        "next_step": "manual_review_then_heldout_gate",
        "promotion_allowed": bool(fixed6_gate.get("promotion_allowed")),
        "mode": "candidate_gate",
        "next_actions": [
            "review general-quality contact sheet",
            "run heldout/general promotion checks before default change",
        ],
    }


def _candidate_commands(*, candidate_name: str, checkpoint: Path) -> dict[str, str]:
    checkpoint_arg = _quote_pwsh(str(checkpoint))
    return {
        "smoke": (
            "python -m gemmanima.cli real-render-command "
            f"--hiddenstage-bridge {checkpoint_arg} --json"
        ),
        "fixed6_plan": (
            "python -m gemmanima.cli text-rendering-qwen-baseline-plan "
            f"--max-cases 6 --student-checkpoint {checkpoint_arg} --student-name {candidate_name} --json"
        ),
        "general_scene_plan": (
            "python -m gemmanima.cli text-preservation-general-scene-eval-plan "
            f"--count 50 --student-checkpoint {checkpoint_arg} --student-name {candidate_name} --json"
        ),
    }


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8-sig"))


def _read_required_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8-sig"))


def _replay_case(row: dict[str, Any]) -> dict[str, Any]:
    delta = float(row.get("delta_vs_v5") or 0.0)
    weight = 1.0 + min(4.0, max(0.0, delta) * 20.0)
    return {
        "case_id": row.get("case_id"),
        "v5_mse": row.get("v5_mse"),
        "candidate_mse": row.get("candidate_mse"),
        "delta_vs_v5": delta,
        "replay_weight": round(weight, 4),
        "repair_priority": "hard" if delta >= 0.05 else "medium",
    }


def _artifact(name: str, path: str | Path | None, required: bool) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(Path(path)) if path else None,
        "required": required,
        "exists": Path(path).exists() if path else False,
    }


def _promotion_block_reason(fixed6: dict[str, Any], missing_required: list[str]) -> str:
    if missing_required:
        return f"missing_required_artifacts:{','.join(missing_required)}"
    if fixed6.get("status") == "missing":
        return "fixed6_missing"
    if fixed6.get("status") == "rejected" or int(fixed6.get("regression_count") or 0) > 0:
        return "fixed6_regression"
    return "manual_review_required"


def _path_or_none(path: str | Path | None) -> str | None:
    return str(Path(path)) if path is not None else None


def _quote_pwsh(value: str) -> str:
    return f"\"{value.replace('`', '``').replace('$', '`$').replace(chr(34), '`' + chr(34))}\""
