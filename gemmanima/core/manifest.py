from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from gemmanima.core.schemas import ContextCapsule, GenerationPlan, JobStatus, Mode, RenderResult


@dataclass
class Manifest:
    job_id: str
    session_id: str
    image_id: str | None
    mode: Mode
    status: JobStatus
    user_request: str
    context_capsule: ContextCapsule | None = None
    plan: GenerationPlan | None = None
    models: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    renderer: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        session_id: str,
        mode: Mode,
        status: JobStatus,
        user_request: str,
        context_capsule: ContextCapsule | None = None,
        plan: GenerationPlan | None = None,
        render_result: RenderResult | None = None,
        models: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
        renderer: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: str | None = None,
    ) -> "Manifest":
        return cls(
            job_id=uuid4().hex,
            session_id=session_id,
            image_id=render_result.image_id if render_result else None,
            mode=mode,
            status=status,
            user_request=user_request,
            context_capsule=context_capsule,
            plan=plan,
            models=models or {},
            hardware=hardware or {},
            renderer=renderer or {},
            output={"path": str(render_result.output_path), "seed": render_result.seed} if render_result else {},
            warnings=list(warnings or ()),
            error=error,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["status"] = self.status.value
        return data


class ManifestStore:
    def __init__(self, root: str | Path = "runs/manifests") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, manifest: Manifest) -> Path:
        path = self.root / f"{manifest.created_at[:10]}_{manifest.job_id}.json"
        path.write_text(
            json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def read_json(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def latest(self) -> Path | None:
        manifests = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return manifests[0] if manifests else None
