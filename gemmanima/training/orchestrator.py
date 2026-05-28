from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gemmanima.training.bridge_training import BridgeTrainingPlan
from gemmanima.training.gemma_cache import audit_cache_pairing, DEFAULT_GEMMA_DIR, DEFAULT_TARGET_DIR
from gemmanima.training.teacher_targets import audit_split_target_completion


DEFAULT_STATE_PATH = Path(r"D:\Projects\training\logs\hiddenstage_pipeline_state.json")


def pipeline_status(
    *,
    target_dir: str | Path = DEFAULT_TARGET_DIR,
    gemma_dir: str | Path = DEFAULT_GEMMA_DIR,
) -> dict[str, Any]:
    target = audit_split_target_completion(target_dir)
    pairing = audit_cache_pairing(target_dir=target_dir, gemma_dir=gemma_dir)
    bridge = BridgeTrainingPlan(target_dir=Path(target_dir), gemma_dir=Path(gemma_dir)).to_json_dict()
    if not target["complete"]:
        next_action = "wait_for_teacher_targets"
    elif not pairing["ready_for_bridge_training"]:
        next_action = "start_gemma_hidden_cache"
    elif not Path(bridge["output"]).exists():
        next_action = "start_hiddenstage_bridge_training"
    else:
        next_action = "evaluate_hiddenstage_bridge"
    return {
        "target_completion": target,
        "gemma_pairing": pairing,
        "bridge_training": bridge,
        "next_action": next_action,
    }


def write_pipeline_status(path: str | Path = DEFAULT_STATE_PATH, **kwargs: Any) -> Path:
    status = pipeline_status(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
