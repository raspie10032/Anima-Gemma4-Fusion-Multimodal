from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmanima.training.gemma_cache import DEFAULT_GEMMA_DIR, DEFAULT_TARGET_DIR, audit_cache_pairing


DEFAULT_BRIDGE_OUT = Path(r"E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt")
DEFAULT_TRAIN_SCRIPT = Path(r"E:\anima_gemma_swap\scripts\core\08_train_stream_batched.py")
DEFAULT_PYTHON = Path(r"E:\ComfyUI_sage\python_embeded\python.exe")


@dataclass(frozen=True)
class BridgeTrainingPlan:
    target_dir: Path = DEFAULT_TARGET_DIR
    gemma_dir: Path = DEFAULT_GEMMA_DIR
    output: Path = DEFAULT_BRIDGE_OUT
    epochs: int = 2
    batch_size: int = 2
    accum: int = 2
    val: int = 2000
    lr: float = 5e-4
    prefetch_gb: float = 48.0
    save_every_shards: int = 25
    gpu_index: int = 0

    def command(self, *, python_exe: str | Path = DEFAULT_PYTHON, script: str | Path = DEFAULT_TRAIN_SCRIPT) -> str:
        return (
            f"$env:CUDA_VISIBLE_DEVICES='{self.gpu_index}'; "
            f"\"{Path(python_exe)}\" \"{Path(script)}\" "
            f"--targets \"{self.target_dir}\" "
            f"--gemma \"{self.gemma_dir}\" "
            f"--out \"{self.output}\" "
            f"--epochs {self.epochs} "
            f"--lr {self.lr} "
            f"--batch-size {self.batch_size} "
            f"--accum {self.accum} "
            f"--val {self.val} "
            f"--prefetch-gb {self.prefetch_gb} "
            f"--save-every-shards {self.save_every_shards}"
        )

    def to_json_dict(self) -> dict[str, Any]:
        pairing = audit_cache_pairing(target_dir=self.target_dir, gemma_dir=self.gemma_dir)
        return {
            "target_dir": str(self.target_dir),
            "gemma_dir": str(self.gemma_dir),
            "output": str(self.output),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accum": self.accum,
            "val": self.val,
            "lr": self.lr,
            "prefetch_gb": self.prefetch_gb,
            "save_every_shards": self.save_every_shards,
            "gpu_index": self.gpu_index,
            "gpu_name": "RTX 4070 Ti SUPER",
            "cache_pairing": pairing,
            "ready": pairing["ready_for_bridge_training"],
            "command": self.command(),
        }
