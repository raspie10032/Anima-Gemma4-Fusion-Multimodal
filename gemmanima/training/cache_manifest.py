from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gemmanima.core.protocol import PROTOCOL_VERSION
from gemmanima.core.validation import SchemaValidationError, validate_cache_build_manifest_payload


@dataclass
class CacheBuildManifest:
    stage: str
    cache_kind: str
    sample_count: int
    source_manifest: Path
    output_dir: Path
    success_count: int = 0
    failure_count: int = 0
    shape: tuple[int, ...] = ()
    dtype: str = ""
    device: str = ""
    dataset_id: str = ""
    lineage: str = "internal_experimental"
    public_release_allowed: bool = False
    protocol_version: str = PROTOCOL_VERSION

    @property
    def success_rate(self) -> float:
        if self.sample_count <= 0:
            return 0.0
        return round(self.success_count / self.sample_count, 6)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "stage": self.stage,
            "cache_kind": self.cache_kind,
            "sample_count": self.sample_count,
            "source_manifest": str(self.source_manifest),
            "output_dir": str(self.output_dir),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "lineage": {
                "lineage": self.lineage,
                "public_release_allowed": self.public_release_allowed,
                "dataset_id": self.dataset_id,
            },
        }

    def validate(self) -> None:
        if self.success_count + self.failure_count > self.sample_count:
            raise SchemaValidationError("success_count plus failure_count cannot exceed sample_count")
        validate_cache_build_manifest_payload(self.to_json_dict())


def write_cache_build_manifest(manifest: CacheBuildManifest, output_path: str | Path) -> Path:
    manifest.validate()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_cache_manifest_write_command(
    *,
    cache_kind: str,
    sample_count: int,
    source_manifest: str | Path,
    output_dir: str | Path,
    manifest_out: str | Path,
    stage: str = "poc1_1k_smoke",
    success_count: int = 0,
    failure_count: int = 0,
    shape: tuple[int, ...] = (),
    dtype: str = "",
    device: str = "",
) -> str:
    shape_text = ",".join(str(part) for part in shape)
    return (
        "python -m gemmanima.cli write-cache-manifest "
        f"--stage {stage} "
        f"--cache-kind {cache_kind} "
        f"--sample-count {sample_count} "
        f"--source-manifest \"{Path(source_manifest)}\" "
        f"--output-dir \"{Path(output_dir)}\" "
        f"--success-count {success_count} "
        f"--failure-count {failure_count} "
        f"--shape {shape_text} "
        f"--dtype {dtype} "
        f"--device {device} "
        f"--manifest-out \"{Path(manifest_out)}\""
    )
