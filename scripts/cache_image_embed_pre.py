from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


PROC_REPO = "unsloth/gemma-4-e2b-it"
TRAINING_ROOT = Path(r"D:\Projects\training")


def load_training_config():
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINING_ROOT))
    from config import BASE_MODEL, load_secrets  # type: ignore

    return BASE_MODEL, load_secrets


def load_rows(manifest: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("image") and row.get("image_embed_pre"):
                row["_manifest_index"] = row_index
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def select_device(torch, requested: str) -> str:
    if requested != "cuda":
        return requested
    if not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is false")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    name = torch.cuda.get_device_name(0)
    print(f"[cache-image-embed] CUDA_VISIBLE_DEVICES={visible!r} visible cuda:0={name}")
    return "cuda:0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache Gemma vision pre-projector image states for planner manifest rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--cache-manifest", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    base_model, load_secrets = load_training_config()
    manifest = Path(args.manifest)
    rows = load_rows(manifest, args.limit)
    if not rows:
        raise SystemExit(f"no rows with image and image_embed_pre found: {manifest}")
    cache_manifest = Path(args.cache_manifest) if args.cache_manifest else Path(rows[0]["image_embed_pre"]).parent / "cache_manifest.jsonl"
    cache_manifest.parent.mkdir(parents=True, exist_ok=True)

    sharded_rows = [
        row for row in rows
        if int(row.get("_manifest_index", 0)) % args.num_shards == args.shard_index
    ]
    todo = list(sharded_rows)
    skipped_existing = 0
    print(
        f"[cache-image-embed] rows={len(rows)} shard={args.shard_index}/{args.num_shards} "
        f"sharded_rows={len(sharded_rows)} todo={len(todo)}"
    )
    if not todo:
        return 0

    device = select_device(torch, args.device)
    token = load_secrets()["hf_token"]
    processor = AutoProcessor.from_pretrained(PROC_REPO, token=token)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0} if device.startswith("cuda") else None,
        token=token,
    )
    model.eval()

    done = 0
    failures = 0
    t0 = time.time()
    with cache_manifest.open("a", encoding="utf-8") as mani:
        for row in todo:
            image_path = Path(row["image"])
            dst = Path(row["image_embed_pre"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if args.resume and dst.exists():
                    skipped_existing += 1
                    if skipped_existing % max(1, args.log_every * 10) == 0:
                        print(
                            f"[cache-image-embed] runtime_skip_existing={skipped_existing}/{len(todo)} "
                            f"done={done} failures={failures}",
                            flush=True,
                        )
                    continue
                image = Image.open(image_path).convert("RGB")
                enc = processor.image_processor(images=image, return_tensors="pt")
                px = enc["pixel_values"].to(model.device, torch.bfloat16)
                ipi = enc["image_position_ids"].to(model.device)
                with torch.no_grad():
                    feats = model.get_image_features(px, ipi, return_dict=True).last_hidden_state
                tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
                torch.save(feats.to(torch.float16).cpu(), tmp)
                os.replace(tmp, dst)
                mani.write(
                    json.dumps(
                        {
                            "id": row.get("id"),
                            "image": str(image_path),
                            "embed": str(dst),
                            "kind": "pre_proj",
                            "n_tok": int(feats.shape[0]),
                            "hidden": int(feats.shape[-1]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                done += 1
            except Exception as exc:  # keep the long cache run resumable
                failures += 1
                print(f"[cache-image-embed] failed id={row.get('id')} image={image_path}: {exc}", flush=True)
            if done and done % args.log_every == 0:
                mani.flush()
                elapsed = time.time() - t0
                rate = done / max(1e-6, elapsed)
                eta = (len(todo) - done - failures - skipped_existing) / max(1e-6, rate) / 3600
                print(
                    f"[cache-image-embed] {done}/{len(todo)} done skipped_existing={skipped_existing} failures={failures} "
                    f"rate={rate:.2f} img/s eta={eta:.2f}h",
                    flush=True,
                )
    elapsed = time.time() - t0
    print(
        f"[cache-image-embed] complete done={done} skipped_existing={skipped_existing} "
        f"failures={failures} elapsed={elapsed:.1f}s"
    )
    return 0 if done or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
