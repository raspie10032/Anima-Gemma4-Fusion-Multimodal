from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from gemmanima.core.config import EngineConfig
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.training.real_render import DEFAULT_EMBEDDED_PYTHON


Runner = Callable[..., subprocess.CompletedProcess[str]]
WORKER_RESULT_PREFIX = "GEMMANIMA_WORKER_RESULT "


class LocalWorkerAnimaRendererAdapter(AnimaRendererAdapter):
    """Runs the native Anima/Comfy path in a child process so the chat server survives native crashes."""

    dry_run = False

    def __init__(
        self,
        output_root: str | Path = "runs/images",
        *,
        config: EngineConfig | None = None,
        runner: Runner = subprocess.run,
        python_executable: str | Path | None = None,
        unet_dtype: str = "fp8_e4m3fn_fast",
        tiled_vae: bool = True,
        comfy_args: tuple[str, ...] = (),
        timeout_seconds: int = 1800,
    ) -> None:
        super().__init__(output_root)
        self.config = config or EngineConfig()
        self.runner = runner
        self.python_executable = str(python_executable or _default_worker_python())
        self.unet_dtype = unet_dtype
        self.tiled_vae = tiled_vae
        self.comfy_args = comfy_args
        self.timeout_seconds = timeout_seconds

    def generate(self, plan: GenerationPlan, conditioning: ConditioningBundle) -> RenderResult:
        conditioning.validate()
        plan.validate()
        seed = plan.seed if plan.seed is not None else self._stable_seed(plan.prompt)
        worker_plan = GenerationPlan.from_dict({**plan.to_json_dict(), "seed": seed})
        image_id = uuid4().hex
        output_path = self.output_root / f"{image_id}.png"
        payload = {
            "plan": worker_plan.to_json_dict(),
            "conditioning": conditioning.to_json_dict(),
            "output_root": str(self.output_root),
            "image_id": image_id,
            "output_path": str(output_path),
            "config": _config_to_json_dict(self.config),
            "unet_dtype": self.unet_dtype,
            "tiled_vae": self.tiled_vae,
            "comfy_args": list(self.comfy_args),
        }
        repo_root = Path(__file__).resolve().parents[2]
        command = [
            self.python_executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(repo_root)!r}); "
                "from gemmanima.rendering.worker_render import main; "
                "raise SystemExit(main(hard_exit_on_success=True))"
            ),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        completed = self.runner(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=self.timeout_seconds,
            env=env,
        )
        if completed.returncode != 0:
            if output_path.exists():
                return RenderResult(
                    image_id=image_id,
                    output_path=output_path,
                    seed=seed,
                    warnings=(
                        f"worker recovered image after native exit code {completed.returncode}; output file is valid",
                    ),
                )
            detail = _combined_output(completed)
            raise RuntimeError(f"local Anima worker failed with exit code {completed.returncode}: {detail}")
        data = _parse_worker_result(completed.stdout)
        output_path = Path(str(data["output_path"]))
        if not output_path.exists():
            raise RuntimeError(f"local Anima worker did not create output: {output_path}")
        return RenderResult(
            image_id=str(data["image_id"]),
            output_path=output_path,
            seed=int(data["seed"]),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
        )


def _config_to_json_dict(config: EngineConfig) -> dict[str, object]:
    return {
        "anima_diffusion_model": str(config.models.anima_diffusion_model),
        "anima_vae": str(config.models.anima_vae),
        "hiddenstage_bridge": str(config.models.hiddenstage_bridge),
    }


def _default_worker_python() -> Path | str:
    if DEFAULT_EMBEDDED_PYTHON.exists():
        return DEFAULT_EMBEDDED_PYTHON
    return sys.executable


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    return stderr or stdout or "no worker output"


def _parse_worker_result(stdout: str) -> dict[str, object]:
    for raw_line in reversed((stdout or "").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(WORKER_RESULT_PREFIX):
            return json.loads(line[len(WORKER_RESULT_PREFIX) :])
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"local Anima worker did not return JSON: {(stdout or '').strip()}")
