from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageChops, ImageDraw


DEFAULT_EMBEDDED_PYTHON = Path(r"E:\ComfyUI_sage\python_embeded\python.exe")
DEFAULT_EVAL_GENERATE_SCRIPT = Path(r"E:\anima_gemma_swap\scripts\core\11_eval_generate.py")
DEFAULT_TEXT_DELTA_ADAPTER = Path(r"E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt")


def default_output_root() -> Path:
    return Path("reports") / "bridge_quality_diagnosis" / "latest"


@dataclass(frozen=True)
class BridgeQualityDiagnosisConfig:
    prompt: str
    output_root: Path = default_output_root()
    adapter: Path = DEFAULT_TEXT_DELTA_ADAPTER
    embedded_python: Path = DEFAULT_EMBEDDED_PYTHON
    eval_generate_script: Path = DEFAULT_EVAL_GENERATE_SCRIPT
    gpu_index: int = 0
    seed: int = 424242
    size: int = 1024
    steps: int = 30
    cfg: float = 4.5
    sampler: str = "euler_ancestral"
    scheduler: str = "sgm_uniform"
    unet_dtype: str = "fp8_e4m3fn_fast"

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("GEMMA_EMBED_ON_GPU", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        return env


def build_eval_generate_command(
    config: BridgeQualityDiagnosisConfig,
    *,
    mode: Literal["qwen", "gemma"],
    name: str,
    prompt_file: Path,
) -> list[str]:
    command = [
        str(config.embedded_python),
        str(config.eval_generate_script),
        "--mode",
        mode,
        "--name",
        name,
        "--prompts",
        str(prompt_file),
        "--limit",
        "1",
        "--seed",
        str(config.seed),
        "--size",
        str(config.size),
        "--steps",
        str(config.steps),
        "--cfg",
        str(config.cfg),
        "--sampler",
        config.sampler,
        "--scheduler",
        config.scheduler,
        "--out-root",
        str(config.output_root / "images"),
        "--unet-dtype",
        config.unet_dtype,
    ]
    if mode == "gemma":
        command.extend(["--adapter", str(config.adapter)])
    return command


def write_prompt_file(config: BridgeQualityDiagnosisConfig) -> Path:
    prompt_file = config.output_root / "prompt.jsonl"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "eval_idx": 0,
        "idx": 0,
        "id": "bridge_quality_manual_prompt",
        "src": "manual",
        "text": config.prompt,
    }
    prompt_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return prompt_file


def run_diagnosis(config: BridgeQualityDiagnosisConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = config.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = write_prompt_file(config)
    results = {}
    for mode, name in (("qwen", "qwen_baseline"), ("gemma", "gemma_bridge")):
        command = build_eval_generate_command(config, mode=mode, name=name, prompt_file=prompt_file)
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=config.env())
        (log_dir / f"{name}.out.log").write_text(completed.stdout or "", encoding="utf-8", errors="replace")
        (log_dir / f"{name}.err.log").write_text(completed.stderr or "", encoding="utf-8", errors="replace")
        image_path = config.output_root / "images" / name / "000.png"
        results[name] = {
            "command": command,
            "returncode": completed.returncode,
            "image": str(image_path),
            "exists": image_path.exists(),
        }
        if completed.returncode != 0:
            break
    comparison = {}
    qwen_image = Path(results.get("qwen_baseline", {}).get("image", ""))
    gemma_image = Path(results.get("gemma_bridge", {}).get("image", ""))
    if qwen_image.exists() and gemma_image.exists():
        comparison = compare_images(qwen_image, gemma_image)
        contact = config.output_root / "contact_sheet.png"
        write_contact_sheet(qwen_image, gemma_image, contact)
        comparison["contact_sheet"] = str(contact)
    report = {
        "stage": "bridge_quality_diagnosis",
        "prompt": config.prompt,
        "adapter": str(config.adapter),
        "gpu_index": config.gpu_index,
        "render_settings": {
            "seed": config.seed,
            "size": config.size,
            "steps": config.steps,
            "cfg": config.cfg,
            "sampler": config.sampler,
            "scheduler": config.scheduler,
            "unet_dtype": config.unet_dtype,
        },
        "results": results,
        "comparison": comparison,
    }
    report_path = config.output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def compare_images(left: Path, right: Path) -> dict[str, Any]:
    with Image.open(left).convert("RGB") as left_img, Image.open(right).convert("RGB") as right_img:
        if left_img.size != right_img.size:
            right_img = right_img.resize(left_img.size)
        diff = ImageChops.difference(left_img, right_img)
        values = list(diff.getdata())
        mse = sum(sum(channel * channel for channel in pixel) for pixel in values) / max(1, len(values) * 3)
        return {
            "mse": mse / (255.0 * 255.0),
            "left_size": list(left_img.size),
            "right_size": list(right_img.size),
        }


def write_contact_sheet(qwen_image: Path, gemma_image: Path, output: Path) -> None:
    with Image.open(qwen_image).convert("RGB") as qwen, Image.open(gemma_image).convert("RGB") as gemma:
        label_h = 48
        width = qwen.width + gemma.width
        height = max(qwen.height, gemma.height) + label_h
        sheet = Image.new("RGB", (width, height), "white")
        sheet.paste(qwen, (0, label_h))
        sheet.paste(gemma, (qwen.width, label_h))
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 14), "Qwen/Anima TE baseline", fill=(0, 0, 0))
        draw.text((qwen.width + 12, 14), "Gemma bridge", fill=(0, 0, 0))
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output)
