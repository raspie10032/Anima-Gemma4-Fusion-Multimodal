from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


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

from gemmanima.rendering.image_state_translator import ImageStateToConditioningTranslator, load_image_state_translator  # noqa: E402


NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, text"


def read_subset_record(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["idx"]) == index:
                return row
    raise SystemExit(f"idx {index} not found in {path}")


def t5_ids_weights(tokenizer: T5XXLTokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    token_weight_pairs = tokenizer.tokenize_with_weights(text, return_word_ids=False)[0]
    ids = torch.tensor([pair[0] for pair in token_weight_pairs], dtype=torch.int32)
    weights = torch.tensor([pair[1] for pair in token_weight_pairs], dtype=torch.float32)
    return ids, weights


def load_translator(path: Path, device: str, dtype: torch.dtype) -> ImageStateToConditioningTranslator:
    return load_image_state_translator(path, device=device, dtype=dtype)


def save_img(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().to("cpu", torch.float32)
    while image.dim() > 3:
        image = image[0]
    array = ((image.clamp(0, 1) * 255 + 0.5).to(torch.uint8)).contiguous().numpy()
    Image.fromarray(array, "RGB").save(path)


def make_latent(vae, size: int) -> dict[str, torch.Tensor]:
    gray = torch.full((1, size, size, 3), 0.5, dtype=torch.float32)
    return {"samples": vae.encode(gray)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render with the image-state to Anima conditioning translator.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--seed", type=int, default=930001)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="normal")
    parser.add_argument("--unet-dtype", choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    record = read_subset_record(Path(args.subset), args.idx)
    prompt = args.prompt or record.get("visible_prompt") or record["text"]
    image_state = torch.load(record["image_embed_pre"], map_location="cpu", weights_only=False).float().unsqueeze(0)

    model_options = {}
    if args.unet_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif args.unet_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif args.unet_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    diffusion_model = folder_paths.get_full_path("diffusion_models", _env.ANIMA_DM)
    model = comfy.sd.load_diffusion_model(diffusion_model, model_options=model_options)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", _env.ANIMA_VAE)))
    adapter_dtype = torch.bfloat16 if args.unet_dtype.startswith("fp8") else model.model.get_dtype_inference()
    translator = load_translator(Path(args.checkpoint), "cuda", adapter_dtype)
    model.model.diffusion_model.llm_adapter = translator

    tokenizer = T5XXLTokenizer()
    pos_ids, pos_weights = t5_ids_weights(tokenizer, prompt)
    neg_ids, neg_weights = t5_ids_weights(tokenizer, args.negative)
    positive = [[
        image_state,
        {
            "t5xxl_ids": pos_ids,
            "t5xxl_weights": pos_weights,
            "visible_prompt": prompt,
            "image_embed_pre": record["image_embed_pre"],
            "source_idx": record["idx"],
        },
    ]]
    negative_state = torch.zeros((1, 1, image_state.shape[-1]), dtype=torch.float32)
    negative = [[negative_state, {"t5xxl_ids": neg_ids, "t5xxl_weights": neg_weights}]]

    latent = make_latent(vae, args.size)
    sampled = nodes.KSampler().sample(
        model,
        args.seed,
        args.steps,
        args.cfg,
        args.sampler,
        args.scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
    )[0]
    out = Path(args.out)
    save_img(vae.decode(sampled["samples"]), out)
    payload = {
        "output": str(out),
        "checkpoint": args.checkpoint,
        "subset": args.subset,
        "idx": args.idx,
        "source_image": record.get("image"),
        "image_embed_pre": record.get("image_embed_pre"),
        "prompt": prompt,
        "seed": args.seed,
        "size": args.size,
        "steps": args.steps,
        "cfg": args.cfg,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
