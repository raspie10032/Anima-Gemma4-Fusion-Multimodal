from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch


BuildAdapter = Callable[..., tuple[object, object]]
LoadKv = Callable[[object, object], None]


def attach_hiddenstage_adapter(
    model: Any,
    *,
    diffusion_model_path: str | Path,
    checkpoint: str | Path,
    build_adapter: BuildAdapter | None = None,
    load_kv: LoadKv | None = None,
    adapter_dtype: torch.dtype | None = None,
    device: str = "cuda",
) -> object:
    if build_adapter is None or load_kv is None:
        import swap_adapter

        build_adapter = build_adapter or swap_adapter.build
        load_kv = load_kv or swap_adapter.load_kv

    dtype = adapter_dtype if adapter_dtype is not None else model.model.get_dtype_inference()
    adapter, _ = build_adapter(str(diffusion_model_path), device=device, dtype=dtype)
    checkpoint_data = torch.load(checkpoint, weights_only=False)
    load_kv(adapter, checkpoint_data["kv"])
    model.model.diffusion_model.llm_adapter = adapter
    return adapter


def adapter_dtype_for_unet(model: Any, unet_dtype: str) -> torch.dtype:
    if unet_dtype.startswith("fp8"):
        return torch.bfloat16
    return model.model.get_dtype_inference()
