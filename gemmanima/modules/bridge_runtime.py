from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmanima.core.config import EngineConfig


@dataclass(frozen=True)
class BridgeForwardSummary:
    checkpoint: Path
    input_shape: tuple[int, ...]
    t5_ids_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    finite: bool
    val_mse: float | None
    epoch: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "input_shape": list(self.input_shape),
            "t5_ids_shape": list(self.t5_ids_shape),
            "output_shape": list(self.output_shape),
            "finite": self.finite,
            "val_mse": self.val_mse,
            "epoch": self.epoch,
        }


class TrainedBridgeRuntime:
    """Heavy runtime for the trained Gemma hidden -> Anima adapter bridge."""

    def __init__(
        self,
        *,
        config: EngineConfig | None = None,
        comfy_root: str | Path = r"E:\ComfyUI_sage\ComfyUI",
        swap_adapter_root: str | Path = r"E:\anima_gemma_swap\scripts\core",
        device: str = "cuda",
    ) -> None:
        self.config = config or EngineConfig()
        self.comfy_root = Path(comfy_root)
        self.swap_adapter_root = Path(swap_adapter_root)
        self.device = device
        self._adapter = None
        self._checkpoint: dict[str, Any] | None = None

    def load(self) -> None:
        if self._adapter is not None:
            return
        sys.path.insert(0, str(self.comfy_root))
        sys.path.insert(0, str(self.swap_adapter_root))
        import torch
        import swap_adapter

        checkpoint = torch.load(self.config.models.hiddenstage_bridge, map_location="cpu", weights_only=False)
        adapter, _ = swap_adapter.build(
            str(self.config.models.anima_diffusion_model),
            device=self.device,
            dtype=torch.float32,
        )
        swap_adapter.load_kv(adapter, checkpoint["kv"])
        adapter.eval()
        self._adapter = adapter
        self._checkpoint = checkpoint

    def forward(self, gemma_hidden, t5_ids):
        self.load()
        import torch

        assert self._adapter is not None
        with torch.no_grad():
            return self._adapter(gemma_hidden.to(self.device, torch.float32), t5_ids.to(self.device).long())

    def smoke(self, *, seq_len: int = 16, t5_len: int = 32) -> BridgeForwardSummary:
        self.load()
        import torch

        gemma_hidden = torch.randn(1, seq_len, 1536, device=self.device, dtype=torch.float32)
        t5_ids = torch.arange(t5_len, device=self.device, dtype=torch.long)[None]
        out = self.forward(gemma_hidden, t5_ids)
        assert self._checkpoint is not None
        return BridgeForwardSummary(
            checkpoint=self.config.models.hiddenstage_bridge,
            input_shape=tuple(gemma_hidden.shape),
            t5_ids_shape=tuple(t5_ids.shape),
            output_shape=tuple(out.shape),
            finite=bool(torch.isfinite(out).all().item()),
            val_mse=self._checkpoint.get("val_mse"),
            epoch=self._checkpoint.get("epoch"),
        )
