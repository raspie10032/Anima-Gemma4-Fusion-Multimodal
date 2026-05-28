from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from gemmanima.core.config import EngineConfig
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.training.real_render import build_real_render_command


Runner = Callable[..., subprocess.CompletedProcess[str]]


class ExternalAnimaRendererAdapter(AnimaRendererAdapter):
    """Calls the existing Anima HiddenStage chat renderer as a backend adapter."""

    dry_run = False

    def __init__(
        self,
        output_root: str | Path = "runs/images",
        *,
        config: EngineConfig | None = None,
        runner: Runner = subprocess.run,
        unet_dtype: str = "fp8_e4m3fn_fast",
    ) -> None:
        super().__init__(output_root)
        self.config = config or EngineConfig()
        self.runner = runner
        self.unet_dtype = unet_dtype

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        conditioning.validate()
        plan.validate()
        seed = plan.seed if plan.seed is not None else self._stable_seed(plan.prompt)
        image_id = uuid4().hex
        output_path = self.output_root / f"{image_id}.png"
        size = min(plan.width, plan.height)
        command = build_real_render_command(
            config=self.config,
            request=plan.prompt,
            output=output_path,
            seed=seed,
            size=size,
            steps=plan.steps,
            cfg=plan.cfg,
            unet_dtype=self.unet_dtype,
        )
        if not command.dependencies["ready"]:
            raise RuntimeError(f"real Anima renderer dependencies are not ready: {command.dependencies}")
        completed = self.runner(command.argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"real Anima renderer failed: {detail}")
        if not output_path.exists():
            raise RuntimeError(f"real Anima renderer did not create output: {output_path}")
        warnings = ()
        if plan.width != plan.height:
            warnings = (f"legacy renderer accepts square size only; used {size}",)
        return RenderResult(image_id=image_id, output_path=output_path, seed=seed, warnings=warnings)
