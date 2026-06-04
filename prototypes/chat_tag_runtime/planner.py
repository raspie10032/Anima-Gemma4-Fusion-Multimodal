from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from generator import compile_tipo_planner_prompt
from model_prototype import DEFAULT_CHAT_MODEL, DEFAULT_COMPLETION_CLI, DEFAULT_PLANNER_LORA_GGUF, DEFAULT_PLANNER_MODEL

Runner = Callable[..., Any]
Timer = Callable[[], float]


@dataclass(frozen=True)
class TipoPlannerConfig:
    cli: Path = DEFAULT_COMPLETION_CLI
    model: Path = DEFAULT_CHAT_MODEL
    lora: Path | None = DEFAULT_PLANNER_LORA_GGUF
    merged_model_default: Path | None = DEFAULT_PLANNER_MODEL
    visible_devices: str = "0"
    device: str = "CUDA0"
    max_new_tokens: int = 96
    temperature: float = 0.7
    timeout_seconds: int = 180
    seed: int = 42
    work_dir: Path = Path("runs/prototypes/chat_tag_runtime/planner")


def run_tipo_planner(
    *,
    message: str,
    config: TipoPlannerConfig | None = None,
    runner: Runner = subprocess.Popen,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    cfg = config or TipoPlannerConfig()
    model = cfg.model
    lora = cfg.lora if cfg.lora and cfg.lora.is_file() else None
    using_default_model = False
    if lora is None and cfg.merged_model_default and cfg.merged_model_default.is_file():
        model = cfg.merged_model_default
        using_default_model = True

    missing = _missing({"cli": cfg.cli, "model": model})
    if missing:
        return {"status": "failed", "mode": "planner", "error": "; ".join(missing), "tags": []}

    prompt = compile_tipo_planner_prompt(message=message)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}"
    prompt_file = cfg.work_dir / f"planner_{stamp}.prompt.txt"
    stdout_file = cfg.work_dir / f"planner_{stamp}.stdout.txt"
    stderr_file = cfg.work_dir / f"planner_{stamp}.stderr.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    cmd = [
        str(cfg.cli),
        "-m",
        str(model),
        "--no-warmup",
        "-no-cnv",
        "--no-display-prompt",
        "-dev",
        cfg.device,
        "-ngl",
        "auto",
        "-n",
        str(cfg.max_new_tokens),
        "--temp",
        str(cfg.temperature),
        "--seed",
        str(cfg.seed),
        "-f",
        str(prompt_file),
    ]
    if lora is not None:
        cmd[3:3] = ["--lora", str(lora)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    start = timer()
    with stdout_file.open("w", encoding="utf-8", errors="ignore") as stdout, stderr_file.open("w", encoding="utf-8", errors="ignore") as stderr:
        proc = runner(cmd, stdout=stdout, stderr=stderr, env=env)
        deadline = timer() + cfg.timeout_seconds
        while proc.poll() is None and timer() < deadline:
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
            timed_out = True
        else:
            timed_out = False
    if timed_out:
        return {"status": "failed", "mode": "planner", "error": f"timeout after {cfg.timeout_seconds}s", "tags": []}
    raw = stdout_file.read_text(encoding="utf-8", errors="ignore") if stdout_file.is_file() else ""
    stderr_text = stderr_file.read_text(encoding="utf-8", errors="ignore") if stderr_file.is_file() else ""
    tags = parse_planner_tags(raw)
    return_code = proc.returncode if proc.returncode is not None else -1
    return {
        "mode": "planner",
        "status": "completed" if return_code == 0 else "failed",
        "prompt": prompt,
        "tags": tags,
        "tag_text": ", ".join(tags),
        "raw": raw,
        "stdout_file": str(stdout_file),
        "stderr_file": str(stderr_file),
        "stderr_tail": stderr_text[-2000:],
        "seconds": round(timer() - start, 3),
        "model": str(model),
        "lora": str(lora) if lora else "",
        "merged_model_default": str(cfg.merged_model_default) if cfg.merged_model_default else "",
        "using_default_model": using_default_model,
        "device": cfg.device,
        "error": "" if return_code == 0 else ((stderr_text or raw)[-1000:]),
    }


def parse_planner_tags(raw: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip().strip("`").strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("loading model", "build", "model", "modalities", "available commands", "/exit", "/regen", "/clear", "/read", "/glob", ">", "[ prompt", "exiting")):
            continue
        if "partial tags:" in lower:
            tail = stripped.split("Partial tags:", 1)[-1].strip()
            if tail:
                lines.append(tail)
            continue
        lines.append(stripped)

    seen = set()
    tags: list[str] = []
    text = ", ".join(lines).replace("[end of text]", "").replace("\n", ",")
    for chunk in text.split(","):
        tag = " ".join(chunk.strip().strip(".").split())
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


def merge_user_prompt_and_planner_tags(*, user_prompt: str, planner_tags: list[str], style: str) -> str:
    parts = [" ".join(user_prompt.strip().split())]
    if planner_tags:
        parts.append(", ".join(planner_tags))
    if style.strip():
        parts.append(" ".join(style.strip().split()))
    return ", ".join(part for part in parts if part)


def _missing(assets: dict[str, Path]) -> list[str]:
    return [f"missing {name}: {path}" for name, path in assets.items() if not path.is_file()]
