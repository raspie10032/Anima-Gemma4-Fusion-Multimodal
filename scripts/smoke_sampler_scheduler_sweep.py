from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemmanima.core.generation_presets import sampler_options, scheduler_options
from gemmanima.core.schemas import ConditioningBundle, GenerationPlan
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter


@dataclass(frozen=True)
class SweepCase:
    kind: str
    name: str
    sampler: str
    scheduler: str


def build_cases(mode: str) -> list[SweepCase]:
    if mode == "quick":
        return [
            SweepCase("sampler", "euler", "euler", "sgm_uniform"),
            SweepCase("scheduler", "normal", "euler_ancestral", "normal"),
        ]
    if mode == "matrix":
        return [
            SweepCase("matrix", f"{sampler}__{scheduler}", sampler, scheduler)
            for sampler in sampler_options()
            for scheduler in scheduler_options()
        ]
    cases = [
        SweepCase("sampler", sampler, sampler, "sgm_uniform")
        for sampler in sampler_options()
    ]
    cases.extend(
        SweepCase("scheduler", scheduler, "euler_ancestral", scheduler)
        for scheduler in scheduler_options()
    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one smoke image per ComfyUI sampler/scheduler option.")
    parser.add_argument("--mode", choices=("quick", "axis", "matrix"), default="axis")
    parser.add_argument("--outdir", default="runs/sampler_scheduler_sweep")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=440001)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-sampler", default="")
    parser.add_argument("--only-scheduler", default="")
    parser.add_argument("--prompt", default="1girl, solo, green dress, quiet forest garden, anime illustration")
    parser.add_argument("--unet-dtype", default="fp8_e4m3fn_fast")
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ.setdefault("GEMMA_EMBED_ON_GPU", "1")

    output_root = Path(args.outdir)
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    renderer = InProcessAnimaRendererAdapter(images_root, unet_dtype=args.unet_dtype)
    conditioning = ConditioningBundle(
        source="sampler_scheduler_smoke",
        metadata={"hiddenstage_source_text": args.prompt},
    )
    cases = build_cases(args.mode)
    if args.only_sampler:
        cases = [case for case in cases if case.sampler == args.only_sampler]
    if args.only_scheduler:
        cases = [case for case in cases if case.scheduler == args.only_scheduler]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no sweep cases selected")

    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        plan = GenerationPlan(
            prompt=args.prompt,
            width=args.size,
            height=args.size,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed + index,
            sampler=case.sampler,
            scheduler=case.scheduler,
            renderer_profile="sampler_scheduler_smoke",
        )
        row = {
            **asdict(case),
            "index": index,
            "status": "failed",
            "output_path": "",
            "seconds": 0.0,
            "error": "",
        }
        try:
            result = renderer.generate(plan, conditioning)
            row.update(
                {
                    "status": "completed",
                    "output_path": str(result.output_path),
                    "seconds": round(time.perf_counter() - case_started, 3),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "seconds": round(time.perf_counter() - case_started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(row)
        print(
            f"[{index:03d}/{len(cases):03d}] {case.kind}:{case.name} "
            f"{case.sampler}/{case.scheduler} -> {row['status']} ({row['seconds']}s)",
            flush=True,
        )

    csv_path = output_root / "sampler_scheduler_sweep.csv"
    json_path = output_root / "sampler_scheduler_sweep.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": args.mode,
        "total": len(rows),
        "completed": sum(1 for row in rows if row["status"] == "completed"),
        "failed": sum(1 for row in rows if row["status"] != "completed"),
        "seconds": round(time.perf_counter() - started, 3),
        "csv": str(csv_path),
        "images": str(images_root),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
