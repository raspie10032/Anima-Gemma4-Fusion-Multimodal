from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult


class AnimaRendererAdapter:
    dry_run = True

    def __init__(self, output_root: str | Path = "runs/images") -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        conditioning.validate()
        plan.validate()
        seed = plan.seed if plan.seed is not None else self._stable_seed(plan.prompt)
        image_id = uuid4().hex
        output_path = self.output_root / f"{image_id}.dryrun.txt"
        output_path.write_text(
            "\n".join(
                [
                    "GemmAnima dry-run render",
                    f"prompt: {plan.prompt}",
                    f"negative_prompt: {plan.negative_prompt}",
                    f"size: {plan.width}x{plan.height}",
                    f"steps: {plan.steps}",
                    f"cfg: {plan.cfg}",
                    f"seed: {seed}",
                    f"conditioning_shape: {conditioning.shape}",
                    f"conditioning_source: {conditioning.source}",
                    f"conditioning_metadata: {conditioning.metadata}",
                ]
            ),
            encoding="utf-8",
        )
        return RenderResult(image_id=image_id, output_path=output_path, seed=seed)

    @staticmethod
    def _stable_seed(prompt: str) -> int:
        return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
