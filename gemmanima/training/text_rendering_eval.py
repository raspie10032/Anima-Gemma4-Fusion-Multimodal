from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEXT_EVAL_IMAGE_DIR = "runs/images/text_rendering_eval"
TEXT_EVAL_REPORT_DIR = "reports/text_rendering_eval"
TEXT_EVAL_SEED_BASE = 440000
DEFAULT_EMBEDDED_PYTHON = r"E:\ComfyUI_sage\python_embeded\python.exe"
DEFAULT_RENDER_SCRIPT = r"E:\anima_gemma_swap\scripts\core\18_hiddenstage_chat_generate.py"
DEFAULT_EVAL_GENERATE_SCRIPT = r"E:\anima_gemma_swap\scripts\core\11_eval_generate.py"
DEFAULT_TEACHER_CHECKPOINT = r"E:\anima_gemma_swap\kv_proj_hiddenstage_planner_v2.pt"
DEFAULT_STUDENT_CHECKPOINT = r"runs\cache\poc1_10k\bridge\poc1_10k_bridge.pt"
QWEN_BASELINE_IMAGE_DIR = "runs/images/text_rendering_qwen_baseline"
QWEN_BASELINE_REPORT_DIR = "reports/text_rendering_qwen_baseline"
QWEN_BASELINE_PROMPT_FILE = f"{QWEN_BASELINE_REPORT_DIR}/prompts.jsonl"


def build_text_rendering_eval_pack() -> dict[str, Any]:
    prompts = [
        _item(
            1,
            "sign",
            "LUNA GATE",
            "Object-only text rendering test: draw a close-up moonlit street sign that clearly reads LUNA GATE. No people, no characters, no faces, no bodies; keep the sign centered and large enough to inspect the letters.",
        ),
        _item(
            2,
            "book_cover",
            "STAR ATLAS",
            "Object-only text rendering test: draw a fantasy book cover lying flat on a desk with the title STAR ATLAS printed clearly in serif letters. No people, no characters, no hands, no faces; keep the cover and title centered.",
        ),
        _item(
            3,
            "magic_circle",
            "AETHER",
            "Object-only text rendering test: draw a glowing magic circle carved into stone with the word AETHER repeated around the ring in legible glyph-like lettering. No people, no characters, no faces, no bodies; show only the circle and stone floor.",
        ),
        _item(
            4,
            "UI_panel",
            "HP 42",
            "Object-only text rendering test: draw a standalone anime game UI panel on a neutral background with the label HP 42 clearly readable. No people, no characters, no faces, no bodies; keep the UI panel centered.",
        ),
        _item(
            5,
            "handwritten_note",
            "MEET AT DAWN",
            "Object-only text rendering test: draw a handwritten note pinned to a wooden door that reads MEET AT DAWN, readable but naturally handwritten. No people, no characters, no hands, no faces; keep the note centered.",
        ),
        _item(
            6,
            "label",
            "TEA",
            "Object-only text rendering test: draw a small glass jar with a simple paper label that reads TEA, cozy kitchen lighting, readable label. No people, no characters, no hands, no faces; keep the jar and label centered.",
        ),
    ]
    return {
        "stage": "text_rendering_preservation",
        "metrics": [
            "legibility_score",
            "exact_text_match",
            "character_order_preserved",
            "placement_accuracy",
            "artifact_rate",
            "style_consistency",
        ],
        "prompts": prompts,
        "reporting": _reporting_contract(prompts),
    }


def build_text_rendering_eval_status() -> dict[str, Any]:
    pack = build_text_rendering_eval_pack()
    cases: list[dict[str, Any]] = []
    ready_cases = 0

    for item in pack["prompts"]:
        artifact_status = {
            name: _artifact_status(path)
            for name, path in item["artifacts"].items()
        }
        case_ready = all(artifact["exists"] for artifact in artifact_status.values())
        if case_ready:
            ready_cases += 1
        cases.append(
            {
                "id": item["id"],
                "category": item["category"],
                "target_text": item["target_text"],
                "seed": item["seed"],
                "status": "ready" if case_ready else "pending",
                "artifacts": artifact_status,
                "metrics": _read_observed_metrics(item["artifacts"]["compare_report"]),
            }
        )

    total_cases = len(cases)
    pending_cases = total_cases - ready_cases
    return {
        "stage": pack["stage"],
        "ready": total_cases > 0 and pending_cases == 0,
        "metrics_policy": "observed_artifacts_only",
        "summary": {
            "total_cases": total_cases,
            "ready_cases": ready_cases,
            "pending_cases": pending_cases,
        },
        "cases": cases,
    }


def build_text_rendering_eval_execution_plan(
    pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_pack = pack if pack is not None else build_text_rendering_eval_pack()
    metrics = list(source_pack["reporting"]["comparison_metrics"])
    cases = [_execution_plan_case(item, metrics) for item in source_pack["prompts"]]
    return {
        "stage": source_pack["stage"],
        "mode": "dry_run",
        "executes_gpu_commands": False,
        "artifact_policy": "declare_paths_only",
        "total_cases": len(cases),
        "cases": cases,
    }


def build_text_rendering_eval_run_plan(
    pack: dict[str, Any] | None = None,
    *,
    max_cases: int | None = None,
    gpu_index: int = 0,
    teacher_checkpoint: str | Path = DEFAULT_TEACHER_CHECKPOINT,
    student_checkpoint: str | Path = DEFAULT_STUDENT_CHECKPOINT,
    size: int = 512,
    steps: int = 12,
    cfg: float = 4.5,
    unet_dtype: str = "fp8_e4m3fn_fast",
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("text rendering eval run plan is 4070-only; use CUDA device 0")
    source_pack = pack if pack is not None else build_text_rendering_eval_pack()
    prompts = list(source_pack["prompts"])
    if max_cases is not None:
        prompts = prompts[:max(0, max_cases)]
    cases = [
        _run_plan_case(
            item,
            gpu_index=gpu_index,
            teacher_checkpoint=Path(teacher_checkpoint),
            student_checkpoint=Path(student_checkpoint),
            size=size,
            steps=steps,
            cfg=cfg,
            unet_dtype=unet_dtype,
        )
        for item in prompts
    ]
    return {
        "stage": source_pack["stage"],
        "mode": "executable",
        "executes_gpu_commands": True,
        "artifact_policy": "generate_then_compare_observed_outputs",
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER" if gpu_index == 0 else f"CUDA device {gpu_index}",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "setup_commands": [
            f"New-Item -ItemType Directory -Force -Path \"{TEXT_EVAL_IMAGE_DIR}\"",
            f"New-Item -ItemType Directory -Force -Path \"{TEXT_EVAL_REPORT_DIR}\"",
        ],
        "teacher_checkpoint": str(Path(teacher_checkpoint)),
        "student_checkpoint": str(Path(student_checkpoint)),
        "render_settings": {
            "size": size,
            "steps": steps,
            "cfg": cfg,
            "unet_dtype": unet_dtype,
        },
        "total_cases": len(cases),
        "cases": cases,
    }


def build_text_rendering_qwen_prompt_records(
    pack: dict[str, Any] | None = None,
    *,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    source_pack = pack if pack is not None else build_text_rendering_eval_pack()
    prompts = list(source_pack["prompts"])
    if max_cases is not None:
        prompts = prompts[:max(0, max_cases)]
    return [
        {
            "text": item["prompt"],
            "src": item["id"],
            "id": item["id"],
            "idx": index,
            "eval_idx": index,
        }
        for index, item in enumerate(prompts)
    ]


def write_text_rendering_qwen_prompt_file(
    output: str | Path = QWEN_BASELINE_PROMPT_FILE,
    *,
    pack: dict[str, Any] | None = None,
    max_cases: int | None = None,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_text_rendering_qwen_prompt_records(pack, max_cases=max_cases)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def build_text_rendering_qwen_baseline_plan(
    pack: dict[str, Any] | None = None,
    *,
    max_cases: int | None = None,
    gpu_index: int = 0,
    prompt_file: str | Path = QWEN_BASELINE_PROMPT_FILE,
    out_root: str | Path = QWEN_BASELINE_IMAGE_DIR,
    report_root: str | Path = QWEN_BASELINE_REPORT_DIR,
    student_checkpoint: str | Path = DEFAULT_STUDENT_CHECKPOINT,
    student_name: str = "gemma_poc1_10k",
    seed_base: int = 440001,
    size: int = 512,
    steps: int = 20,
    cfg: float = 4.5,
    sampler: str = "euler",
    scheduler: str = "normal",
    unet_dtype: str = "default",
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("Qwen baseline text rendering plan is 4070-only; use CUDA device 0")
    source_pack = pack if pack is not None else build_text_rendering_eval_pack()
    prompts = list(source_pack["prompts"])
    if max_cases is not None:
        prompts = prompts[:max(0, max_cases)]
    prompt_path = Path(prompt_file)
    out_root_path = Path(out_root)
    report_root_path = Path(report_root)
    qwen_dir = out_root_path / "qwen"
    gemma_dir = out_root_path / student_name
    cases = [
        _qwen_baseline_case(
            item,
            index=index,
            qwen_dir=qwen_dir,
            gemma_dir=gemma_dir,
            report_root=report_root_path,
            student_checkpoint=Path(student_checkpoint),
            student_name=student_name,
            seed=seed_base + index,
        )
        for index, item in enumerate(prompts)
    ]
    return {
        "stage": "text_rendering_qwen_baseline",
        "mode": "executable",
        "executes_gpu_commands": True,
        "teacher_mode": "qwen",
        "student_mode": "gemma",
        "student_name": student_name,
        "gpu_policy": {
            "cuda_visible_devices": str(gpu_index),
            "gpu_name": "RTX 4070 Ti SUPER",
            "reserved_gpu": "RTX 5060 / CUDA device 1",
        },
        "prompt_file": str(prompt_path).replace("\\", "/"),
        "prompt_write_command": (
            "python -m gemmanima.cli text-rendering-qwen-baseline-prompts "
            f"--output {_quote_pwsh(str(prompt_path))} "
            f"--max-cases {len(prompts)} --json"
        ),
        "out_root": str(out_root_path).replace("\\", "/"),
        "report_root": str(report_root_path).replace("\\", "/"),
        "seed_base": seed_base,
        "render_settings": {
            "size": size,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "unet_dtype": unet_dtype,
        },
        "qwen_command": _eval_generate_command(
            gpu_index=gpu_index,
            mode="qwen",
            name="qwen",
            prompt_file=prompt_path,
            out_root=out_root_path,
            seed=seed_base,
            size=size,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            unet_dtype=unet_dtype,
            limit=len(prompts),
        ),
        "gemma_command": _eval_generate_command(
            gpu_index=gpu_index,
            mode="gemma",
            name=student_name,
            prompt_file=prompt_path,
            out_root=out_root_path,
            seed=seed_base,
            size=size,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            unet_dtype=unet_dtype,
            limit=len(prompts),
            adapter=Path(student_checkpoint),
        ),
        "total_cases": len(cases),
        "cases": cases,
    }


def _item(index: int, category: str, target_text: str, prompt: str) -> dict[str, Any]:
    case_id = f"text_eval_{index:03d}_{_slug(category)}_{_slug(target_text)}"
    return {
        "id": case_id,
        "category": category,
        "target_text": target_text,
        "prompt": prompt,
        "seed": TEXT_EVAL_SEED_BASE + index,
        "artifacts": {
            "teacher_image": f"{TEXT_EVAL_IMAGE_DIR}/{case_id}_teacher.png",
            "student_image": f"{TEXT_EVAL_IMAGE_DIR}/{case_id}_student.png",
            "compare_report": f"{TEXT_EVAL_REPORT_DIR}/{case_id}_compare.json",
        },
    }


def _qwen_baseline_case(
    item: dict[str, Any],
    *,
    index: int,
    qwen_dir: Path,
    gemma_dir: Path,
    report_root: Path,
    student_checkpoint: Path,
    student_name: str,
    seed: int,
) -> dict[str, Any]:
    qwen_raw = qwen_dir / f"{index:03d}.png"
    gemma_raw = gemma_dir / f"{index:03d}.png"
    qwen_named = qwen_dir / f"{item['id']}.png"
    gemma_named = gemma_dir / f"{item['id']}.png"
    compare_report = report_root / f"{item['id']}_qwen_vs_{student_name}_compare.json"
    return {
        "id": item["id"],
        "category": item["category"],
        "target_text": item["target_text"],
        "seed": seed,
        "prompt": item["prompt"],
        "qwen_image": str(qwen_named).replace("\\", "/"),
        "gemma_image": str(gemma_named).replace("\\", "/"),
        "qwen_raw_image": str(qwen_raw).replace("\\", "/"),
        "gemma_raw_image": str(gemma_raw).replace("\\", "/"),
        "rename_commands": [
            f"Copy-Item -Force -LiteralPath {_quote_pwsh(str(qwen_raw))} -Destination {_quote_pwsh(str(qwen_named))}",
            f"Copy-Item -Force -LiteralPath {_quote_pwsh(str(gemma_raw))} -Destination {_quote_pwsh(str(gemma_named))}",
        ],
        "compare_report": str(compare_report).replace("\\", "/"),
        "compare_command": _compare_command(
            prompt=item["prompt"],
            seed=seed,
            teacher_image=str(qwen_named),
            student_image=str(gemma_named),
            student_checkpoint=student_checkpoint,
            output=str(compare_report),
        ),
    }


def _eval_generate_command(
    *,
    gpu_index: int,
    mode: str,
    name: str,
    prompt_file: Path,
    out_root: Path,
    seed: int,
    size: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    unet_dtype: str,
    limit: int,
    adapter: Path | None = None,
) -> str:
    embed_prefix = "" if mode == "qwen" else "$env:GEMMA_EMBED_ON_GPU='1'; "
    command = (
        f"$env:CUDA_VISIBLE_DEVICES='{gpu_index}'; "
        f"{embed_prefix}"
        f"& {_quote_pwsh(DEFAULT_EMBEDDED_PYTHON)} {_quote_pwsh(DEFAULT_EVAL_GENERATE_SCRIPT)} "
        f"--mode {mode} --prompts {_quote_pwsh(str(prompt_file))} --name {name} "
        f"--out-root {_quote_pwsh(str(out_root))} --limit {limit} --seed {seed} "
        f"--size {size} --steps {steps} --cfg {cfg} --sampler {sampler} --scheduler {scheduler} "
        f"--unet-dtype {unet_dtype}"
    )
    if adapter is not None:
        command += f" --adapter {_quote_pwsh(str(adapter))}"
    return command


def _run_plan_case(
    item: dict[str, Any],
    *,
    gpu_index: int,
    teacher_checkpoint: Path,
    student_checkpoint: Path,
    size: int,
    steps: int,
    cfg: float,
    unet_dtype: str,
) -> dict[str, Any]:
    teacher_image = item["artifacts"]["teacher_image"]
    student_image = item["artifacts"]["student_image"]
    compare_report = item["artifacts"]["compare_report"]
    prompt = item["prompt"]
    seed = int(item["seed"])
    return {
        "id": item["id"],
        "category": item["category"],
        "target_text": item["target_text"],
        "seed": seed,
        "teacher": {
            "prompt": prompt,
            "adapter": str(teacher_checkpoint),
            "output_path": teacher_image,
            "command": _render_command(
                prompt=prompt,
                adapter=teacher_checkpoint,
                output=teacher_image,
                seed=seed,
                gpu_index=gpu_index,
                size=size,
                steps=steps,
                cfg=cfg,
                unet_dtype=unet_dtype,
            ),
        },
        "student": {
            "prompt": prompt,
            "adapter": str(student_checkpoint),
            "output_path": student_image,
            "command": _render_command(
                prompt=prompt,
                adapter=student_checkpoint,
                output=student_image,
                seed=seed,
                gpu_index=gpu_index,
                size=size,
                steps=steps,
                cfg=cfg,
                unet_dtype=unet_dtype,
            ),
        },
        "comparison": {
            "teacher_image": teacher_image,
            "student_image": student_image,
            "compare_report": compare_report,
            "target_text": item["target_text"],
            "command": _compare_command(
                prompt=prompt,
                seed=seed,
                teacher_image=teacher_image,
                student_image=student_image,
                student_checkpoint=student_checkpoint,
                output=compare_report,
            ),
        },
    }


def _render_command(
    *,
    prompt: str,
    adapter: Path,
    output: str,
    seed: int,
    gpu_index: int,
    size: int,
    steps: int,
    cfg: float,
    unet_dtype: str,
) -> str:
    return (
        f"$env:CUDA_VISIBLE_DEVICES='{gpu_index}'; "
        f"& {_quote_pwsh(DEFAULT_EMBEDDED_PYTHON)} {_quote_pwsh(DEFAULT_RENDER_SCRIPT)} "
        f"--request {_quote_pwsh(prompt)} "
        f"--adapter {_quote_pwsh(str(Path(adapter)))} "
        f"--out {_quote_pwsh(str(Path(output)))} "
        f"--seed {seed} --size {size} --steps {steps} --cfg {cfg} --unet-dtype {unet_dtype}"
    )


def _compare_command(
    *,
    prompt: str,
    seed: int,
    teacher_image: str,
    student_image: str,
    student_checkpoint: Path,
    output: str,
) -> str:
    return (
        "python -m gemmanima.cli write-compare-report "
        f"--prompt {_quote_pwsh(prompt)} "
        f"--seed {seed} "
        f"--teacher-image {_quote_pwsh(str(Path(teacher_image)))} "
        f"--student-image {_quote_pwsh(str(Path(student_image)))} "
        f"--student-checkpoint {_quote_pwsh(str(student_checkpoint))} "
        f"--output {_quote_pwsh(str(Path(output)))} --json"
    )


def _quote_pwsh(value: str) -> str:
    return f"\"{_escape_pwsh_double_quoted(value)}\""


def _escape_pwsh_double_quoted(value: str) -> str:
    return value.replace("`", "``").replace("$", "`$").replace('"', '`"')


def _execution_plan_case(item: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    teacher_image = item["artifacts"]["teacher_image"]
    student_image = item["artifacts"]["student_image"]
    compare_report = item["artifacts"]["compare_report"]
    return {
        "id": item["id"],
        "category": item["category"],
        "target_text": item["target_text"],
        "seed": item["seed"],
        "teacher": {
            "prompt": item["prompt"],
            "seed": item["seed"],
            "output_path": teacher_image,
        },
        "student": {
            "prompt": item["prompt"],
            "seed": item["seed"],
            "output_path": student_image,
        },
        "comparison": {
            "teacher_image": teacher_image,
            "student_image": student_image,
            "compare_report": compare_report,
            "target_text": item["target_text"],
            "metrics": list(metrics),
        },
    }


def _artifact_status(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "exists": Path(path).exists(),
    }


def _read_observed_metrics(compare_report: str) -> dict[str, Any]:
    path = Path(compare_report)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {}
    for key in ("conditioning", "image_metrics", "ocr_metrics"):
        if key == "conditioning" and _has_only_null_values(payload.get(key)):
            continue
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def _has_only_null_values(value: Any) -> bool:
    if not isinstance(value, dict):
        return value is None
    return all(item is None for item in value.values())


def _reporting_contract(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "0.1",
        "image_dir": TEXT_EVAL_IMAGE_DIR,
        "report_dir": TEXT_EVAL_REPORT_DIR,
        "comparison_metrics": ["mse", "mae", "psnr_db", "ocr_exact_text_match"],
        "cases": [
            {
                "id": item["id"],
                "target_text": item["target_text"],
                "seed": item["seed"],
                "teacher_image": item["artifacts"]["teacher_image"],
                "student_image": item["artifacts"]["student_image"],
                "compare_report": item["artifacts"]["compare_report"],
            }
            for item in prompts
        ],
    }


def _slug(value: str) -> str:
    chars: list[str] = []
    previous_separator = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_separator = False
        elif chars and not previous_separator:
            chars.append("_")
            previous_separator = True
    return "".join(chars).strip("_")
