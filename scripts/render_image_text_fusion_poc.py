from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ANIMA_CORE = Path(r"E:\anima_gemma_swap\scripts\core")
if str(ANIMA_CORE) not in sys.path:
    sys.path.insert(0, str(ANIMA_CORE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _env  # noqa: E402,F401
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from comfy.text_encoders.anima import T5XXLTokenizer  # noqa: E402

from gemmanima.modules.image_state_anima_renderer import ImageStateAnimaRendererAdapter  # noqa: E402
from gemmanima.rendering.anima_adapter import adapter_dtype_for_unet, attach_hiddenstage_adapter  # noqa: E402
from gemmanima.rendering.anima_sampler import AnimaSamplerRuntime, build_comfy_sampler, load_anima_model_vae  # noqa: E402
from gemmanima.rendering.gemma_hidden import GemmaHiddenConfig, GemmaHiddenProvider, GemmaTextRuntime  # noqa: E402
from gemmanima.rendering.image_state_engine import (  # noqa: E402
    ConditioningBlend,
    FusionMode,
    ImageStateConditioningConfig,
    ImageStateConditioningEngine,
)
from gemmanima.rendering.t5_tokenizer import T5TokenizerProvider  # noqa: E402


DEFAULT_SUBSET = Path(r"reports\image_state_conditioning_v2_full\subset_full.jsonl")
DEFAULT_IMAGE_CHECKPOINT = Path(
    r"runs\cache\image_state_conditioning_v2_full\bridge\image_state_conditioning_v2_full_image_translator.pt"
)
DEFAULT_TEXT_CHECKPOINT = Path(r"E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt")


def read_subset_record(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["idx"]) == index:
                return row
    raise SystemExit(f"idx {index} not found in {path}")


def read_subset_records(path: Path, indices: list[int]) -> dict[int, dict]:
    wanted = set(indices)
    found = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            if idx in wanted:
                found[idx] = row
                if len(found) == len(wanted):
                    break
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"indices not found in {path}: {missing}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Render image/text fusion PoC samples through the GEMMANIMA runtime engine.")
    parser.add_argument("--subset", default=str(DEFAULT_SUBSET))
    parser.add_argument("--idx", type=int, default=93274)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--modes", nargs="+", choices=["hidden_fusion", "conditioning_fusion"], default=["hidden_fusion", "conditioning_fusion"])
    parser.add_argument("--image-checkpoint", default=str(DEFAULT_IMAGE_CHECKPOINT))
    parser.add_argument("--text-checkpoint", default=str(DEFAULT_TEXT_CHECKPOINT))
    parser.add_argument("--out-dir", default=r"runs\images\image_text_fusion_poc")
    parser.add_argument("--report", default=r"reports\image_state_conditioning_v2_full\image_text_fusion_poc_report.json")
    parser.add_argument("--seed", type=int, default=940001)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--unet-dtype", choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="fp8_e4m3fn_fast")
    parser.add_argument("--text-weight", type=float, default=0.55)
    parser.add_argument("--image-weight", type=float, default=0.45)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("GEMMA_EMBED_ON_GPU", "1")

    indices = args.indices if args.indices is not None else [args.idx]
    seeds = args.seeds if args.seeds is not None else [args.seed]
    records = read_subset_records(Path(args.subset), indices)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diffusion_model = folder_paths.get_full_path("diffusion_models", _env.ANIMA_DM)
    model, vae = load_anima_model_vae(diffusion_model_path=diffusion_model, vae_path=folder_paths.get_full_path("vae", _env.ANIMA_VAE), unet_dtype=args.unet_dtype)
    adapter_dtype = adapter_dtype_for_unet(model, args.unet_dtype)
    text_adapter = attach_hiddenstage_adapter(
        model,
        diffusion_model_path=diffusion_model,
        checkpoint=args.text_checkpoint,
        adapter_dtype=adapter_dtype,
    )
    hidden_provider = GemmaHiddenProvider(GemmaTextRuntime(GemmaHiddenConfig(device="cuda", dtype="bfloat16", embed_on_gpu=True)))
    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(
            checkpoint=Path(args.image_checkpoint),
            device="cuda",
            dtype="bfloat16",
            blend=ConditioningBlend(text_weight=args.text_weight, image_weight=args.image_weight),
        ),
        t5_provider=T5TokenizerProvider(T5XXLTokenizer()),
        text_adapter=text_adapter,
    )
    renderer = ImageStateAnimaRendererAdapter(
        output_root=out_dir,
        engine=engine,
        sampler_runtime=AnimaSamplerRuntime(model=model, vae=vae, sampler=build_comfy_sampler()),
        checkpoint=args.image_checkpoint,
        unet_dtype=args.unet_dtype,
    )

    from gemmanima.core.schemas import GenerationPlan

    outputs = []
    for idx in indices:
        record = records[idx]
        prompt = args.prompt or record.get("visible_prompt") or record.get("teacher_prompt") or record["text"]
        text_state = hidden_provider.encode_image_intent(str(record.get("teacher_prompt") or prompt), prompt)
        for seed in seeds:
            for mode in args.modes:
                result = renderer.generate_from_record(
                    GenerationPlan(prompt=prompt, width=args.size, height=args.size, steps=args.steps, cfg=args.cfg, seed=seed),
                    record,
                    mode=mode,  # type: ignore[arg-type]
                    text_state=text_state,
                )
                outputs.append(
                    {
                        "idx": idx,
                        "mode": mode,
                        "output": str(result.output_path),
                        "seed": result.seed,
                        "warnings": list(result.warnings),
                    }
                )

    payload = {
        "stage": "image_text_fusion_poc",
        "subset": args.subset,
        "indices": indices,
        "seeds": seeds,
        "prompt_override": args.prompt,
        "image_checkpoint": args.image_checkpoint,
        "text_checkpoint": args.text_checkpoint,
        "size": args.size,
        "steps": args.steps,
        "cfg": args.cfg,
        "blend": {"text_weight": args.text_weight, "image_weight": args.image_weight},
        "outputs": outputs,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in outputs:
            print(f"{item['mode']}: {item['output']}")
        print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
