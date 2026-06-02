from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gemmanima.core.config import EngineConfig


DEFAULT_EMBEDDED_PYTHON = Path(r"E:\ComfyUI_sage\python_embeded\python.exe")
DEFAULT_CHAT_RENDER_SCRIPT = Path(r"E:\anima_gemma_swap\scripts\core\18_hiddenstage_chat_generate.py")
DEFAULT_REAL_RENDER_OUTPUT = Path(
    r"C:\Users\seine\Documents\Anima-Gemma4-Fusion-Multimodal\runs\images\nahida_hiddenstage_bridge_real_smoke.png"
)
DEFAULT_REAL_RENDER_REQUEST = (
    "Draw Nahida from Genshin Impact as a bright forest anime illustration, "
    "gentle expression, detailed green-and-white outfit, soft sunlight."
)
DEFAULT_REAL_RENDER_SEED = 19375672098


@dataclass(frozen=True)
class RealRenderCommand:
    command: str
    argv: tuple[str, ...]
    gpu: str
    embedded_python: Path
    script: Path
    adapter: Path
    output: Path
    request: str
    seed: int
    size: int
    steps: int
    cfg: float
    unet_dtype: str
    dependencies: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "argv": list(self.argv),
            "gpu": self.gpu,
            "embedded_python": str(self.embedded_python),
            "script": str(self.script),
            "adapter": str(self.adapter),
            "output": str(self.output),
            "request": self.request,
            "seed": self.seed,
            "size": self.size,
            "steps": self.steps,
            "cfg": self.cfg,
            "unet_dtype": self.unet_dtype,
            "dependencies": self.dependencies,
        }


def build_real_render_command(
    *,
    config: EngineConfig | None = None,
    request: str = DEFAULT_REAL_RENDER_REQUEST,
    output: str | Path = DEFAULT_REAL_RENDER_OUTPUT,
    seed: int = DEFAULT_REAL_RENDER_SEED,
    size: int = 512,
    steps: int = 12,
    cfg: float = 4.5,
    unet_dtype: str = "fp8_e4m3fn_fast",
) -> RealRenderCommand:
    resolved_config = config or EngineConfig()
    embedded_python = DEFAULT_EMBEDDED_PYTHON
    script = DEFAULT_CHAT_RENDER_SCRIPT
    output_path = Path(output)
    adapter = resolved_config.models.hiddenstage_bridge
    argv = (
        str(embedded_python),
        str(script),
        "--request",
        request,
        "--adapter",
        str(adapter),
        "--out",
        str(output_path),
        "--seed",
        str(seed),
        "--size",
        str(size),
        "--steps",
        str(steps),
        "--cfg",
        str(cfg),
        "--unet-dtype",
        unet_dtype,
    )
    command = " ".join(_quote_arg(item) for item in argv)
    return RealRenderCommand(
        command=command,
        argv=argv,
        gpu=resolved_config.hardware.primary_gpu,
        embedded_python=embedded_python,
        script=script,
        adapter=adapter,
        output=output_path,
        request=request,
        seed=seed,
        size=size,
        steps=steps,
        cfg=cfg,
        unet_dtype=unet_dtype,
        dependencies=audit_real_render_dependencies(config=resolved_config),
    )


def audit_real_render_dependencies(config: EngineConfig | None = None) -> dict[str, object]:
    resolved_config = config or EngineConfig()
    checks = {
        "embedded_python": DEFAULT_EMBEDDED_PYTHON.exists(),
        "script": DEFAULT_CHAT_RENDER_SCRIPT.exists(),
        "hiddenstage_bridge": resolved_config.models.hiddenstage_bridge.exists(),
        "anima_diffusion_model": resolved_config.models.anima_diffusion_model.exists(),
        "anima_vae": resolved_config.models.anima_vae.exists(),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
    }


def _quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value) or '"' in value:
        escaped = value.replace('"', r'\"')
        return f'"{escaped}"'
    return value
