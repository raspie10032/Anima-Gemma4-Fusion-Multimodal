from __future__ import annotations

from pathlib import Path
from typing import Any

from gemmanima.core.model_paths import model_path

DEFAULT_BRIDGE_OUT = model_path("hiddenstage_bridge", "kv_proj_hiddenstage_planner_v2.pt")


def audit_bridge_checkpoint(path: str | Path = DEFAULT_BRIDGE_OUT) -> dict[str, Any]:
    ckpt = Path(path)
    result: dict[str, Any] = {
        "path": str(ckpt),
        "exists": ckpt.exists(),
        "size_bytes": ckpt.stat().st_size if ckpt.exists() else 0,
        "readable": False,
        "val_mse": None,
        "epoch": None,
        "batch_size": None,
        "kv_keys": [],
        "passed_mse_gate": False,
        "mse_gate": 0.004,
    }
    if not ckpt.exists():
        return result
    try:
        import torch

        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        result["readable"] = True
        result["val_mse"] = data.get("val_mse")
        result["epoch"] = data.get("epoch")
        result["batch_size"] = data.get("batch_size")
        kv = data.get("kv", {})
        result["kv_keys"] = sorted(kv.keys())[:20]
        result["kv_key_count"] = len(kv)
        result["passed_mse_gate"] = result["val_mse"] is not None and float(result["val_mse"]) <= result["mse_gate"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
