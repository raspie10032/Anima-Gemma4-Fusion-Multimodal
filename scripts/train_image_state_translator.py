from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemmanima.rendering.image_state_translator import ImageStateToConditioningTranslator


class TensorFileCache:
    def __init__(self, max_bytes: int = 0) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self._items: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "items": 0,
            "bytes": 0,
            "max_bytes": self.max_bytes,
        }

    @staticmethod
    def tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.nelement() * tensor.element_size())

    def load(self, path: str | Path) -> torch.Tensor:
        key = str(path)
        if self.max_bytes <= 0:
            self.stats["misses"] += 1
            return torch.load(path, map_location="cpu", weights_only=False).float()
        if key in self._items:
            tensor, size = self._items.pop(key)
            self._items[key] = (tensor, size)
            self.stats["hits"] += 1
            self._sync_stats()
            return tensor

        try:
            tensor = torch.load(path, map_location="cpu", weights_only=False).float()
        except Exception as exc:
            raise RuntimeError(f"failed to load image_embed_pre cache: {path}") from exc
        size = self.tensor_bytes(tensor)
        self.stats["misses"] += 1
        if size <= self.max_bytes:
            while self._items and self.current_bytes + size > self.max_bytes:
                _, (_, evicted_size) = self._items.popitem(last=False)
                self.current_bytes -= evicted_size
                self.stats["evictions"] += 1
            self._items[key] = (tensor, size)
            self.current_bytes += size
        self._sync_stats()
        return tensor

    def _sync_stats(self) -> None:
        self.stats["items"] = len(self._items)
        self.stats["bytes"] = self.current_bytes


def load_subset(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                rows[int(record["idx"])] = record
    return rows


def target_shards(target_dir: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(target_dir / "*.pt")))]


def load_examples_from_shard(subset_rows: dict[int, dict[str, Any]], shard: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in torch.load(shard, map_location="cpu", weights_only=False):
        idx = int(row["idx"])
        source = subset_rows.get(idx)
        if not source:
            continue
        image_embed = source.get("image_embed_pre")
        if not image_embed or not Path(image_embed).exists():
            continue
        examples.append(
            {
                "idx": idx,
                "image_embed_pre": image_embed,
                "t5_ids": row["t5_ids"],
                "target": row["target"],
            }
        )
    return examples


def load_examples(subset_rows: dict[int, dict[str, Any]], target_dir: Path, limit_examples: int = 0) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for shard in target_shards(target_dir):
        examples.extend(load_examples_from_shard(subset_rows, shard))
        if limit_examples and len(examples) >= limit_examples:
            return examples[:limit_examples]
    return examples


def load_guard_replay_indices(path: str | Path | None) -> set[int]:
    if not path:
        return set()
    replay_path = Path(path)
    if not replay_path.exists():
        raise FileNotFoundError(f"guard replay file not found: {replay_path}")
    indices = set()
    with replay_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            indices.add(int(json.loads(line)["idx"]))
    return indices


def expand_guard_replay_examples(
    examples: list[dict[str, Any]],
    *,
    guard_indices: set[int],
    replay_weight: int,
) -> list[dict[str, Any]]:
    if not guard_indices or replay_weight <= 1:
        return examples
    replay = []
    for example in examples:
        if int(example["idx"]) in guard_indices:
            for _ in range(replay_weight - 1):
                replay_item = dict(example)
                replay_item["guard_replay"] = True
                replay.append(replay_item)
    return [*examples, *replay]


def collate(
    examples: list[dict[str, Any]],
    device: torch.device,
    image_cache: TensorFileCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_states = [
        image_cache.load(ex["image_embed_pre"])
        if image_cache
        else torch.load(ex["image_embed_pre"], map_location="cpu", weights_only=False).float()
        for ex in examples
    ]
    t5_ids = [ex["t5_ids"].long() for ex in examples]
    targets = [ex["target"].float() for ex in examples]
    image_max = max(int(t.shape[0]) for t in image_states)
    target_max = max(int(t.shape[0]) for t in targets)
    image_dim = int(image_states[0].shape[-1])
    target_dim = int(targets[0].shape[-1])
    bsz = len(examples)
    image_batch = torch.zeros((bsz, image_max, image_dim), dtype=torch.float32)
    target_batch = torch.zeros((bsz, target_max, target_dim), dtype=torch.float32)
    t5_batch = torch.zeros((bsz, target_max), dtype=torch.long)
    image_mask = torch.zeros((bsz, image_max), dtype=torch.bool)
    target_mask = torch.zeros((bsz, target_max), dtype=torch.bool)
    for i, (image, ids, target) in enumerate(zip(image_states, t5_ids, targets)):
        image_batch[i, : image.shape[0]] = image
        target_batch[i, : target.shape[0]] = target
        t5_batch[i, : ids.shape[0]] = ids
        image_mask[i, : image.shape[0]] = True
        target_mask[i, : target.shape[0]] = True
    return (
        image_batch.to(device),
        t5_batch.to(device),
        target_batch.to(device),
        image_mask.to(device),
        target_mask.to(device),
    )


def batched(items: list[dict[str, Any]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def batch_loss(
    model: nn.Module,
    batch: list[dict[str, Any]],
    device: torch.device,
    image_cache: TensorFileCache | None = None,
) -> torch.Tensor:
    image, t5, target, image_mask, target_mask = collate(batch, device, image_cache=image_cache)
    pred = model(image, t5, image_mask=image_mask, target_mask=target_mask)
    diff = (pred - target).pow(2) * target_mask.unsqueeze(-1)
    denom = target_mask.sum().clamp_min(1) * pred.shape[-1]
    return diff.sum() / denom


@torch.no_grad()
def evaluate(
    model: nn.Module,
    examples: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    image_cache: TensorFileCache | None = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in batched(examples, batch_size):
        loss = batch_loss(model, batch, device, image_cache=image_cache)
        total += float(loss.item()) * len(batch)
        count += len(batch)
    return total / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train image-state to Anima conditioning translator.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-translator-anchor", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val", type=int, default=128)
    parser.add_argument("--limit-examples", type=int, default=0)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=41040)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-cache-gb", type=float, default=0.0)
    parser.add_argument("--guard-replay", default=None)
    parser.add_argument("--guard-replay-weight", type=int, default=1)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epoch-offset", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    subset_rows = load_subset(Path(args.subset))
    shards = target_shards(Path(args.targets))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        raise SystemExit("no target shards found")
    if args.limit_examples:
        examples = load_examples(subset_rows, Path(args.targets), args.limit_examples)
    else:
        examples = load_examples(subset_rows, Path(args.targets), args.val)
    if not examples:
        raise SystemExit("no image-state training examples found")
    random.shuffle(examples)
    val = examples[: min(args.val, len(examples))]
    image_cache_bytes = int(max(0.0, args.image_cache_gb) * (1024**3))
    image_cache = TensorFileCache(image_cache_bytes) if image_cache_bytes > 0 else None
    guard_indices = load_guard_replay_indices(args.guard_replay)

    model = ImageStateToConditioningTranslator().to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    init_val = evaluate(model, val, args.batch_size, device, image_cache=image_cache)
    print(f"target_shards={len(shards)} val={len(val)} init_val_mse={init_val:.6f} image_cache_gb={args.image_cache_gb:.1f}")
    step = 0
    final_train = math.nan
    total_train_examples = 0
    history: list[dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        display_epoch = args.epoch_offset + epoch + 1
        model.train()
        random.shuffle(shards)
        total = 0.0
        count = 0
        opt.zero_grad(set_to_none=True)
        remaining_limit = args.limit_examples
        for shard_index, shard in enumerate(shards, start=1):
            shard_examples = load_examples_from_shard(subset_rows, shard)
            shard_examples = expand_guard_replay_examples(
                shard_examples,
                guard_indices=guard_indices,
                replay_weight=args.guard_replay_weight,
            )
            if remaining_limit:
                if remaining_limit <= 0:
                    break
                shard_examples = shard_examples[:remaining_limit]
                remaining_limit -= len(shard_examples)
            random.shuffle(shard_examples)
            for batch in batched(shard_examples, args.batch_size):
                loss = batch_loss(model, batch, device, image_cache=image_cache)
                (loss / args.accum).backward()
                total += float(loss.item()) * len(batch)
                count += len(batch)
                step += 1
                if step % args.accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            if shard_index % 10 == 0:
                cache_text = ""
                if image_cache:
                    cache_text = (
                        " image_cache="
                        f"{image_cache.stats['bytes'] / (1024**3):.1f}GB"
                        f" hits={image_cache.stats['hits']}"
                        f" misses={image_cache.stats['misses']}"
                        f" evictions={image_cache.stats['evictions']}"
                    )
                print(f"epoch {display_epoch} shard {shard_index}/{len(shards)} examples={count} train_mse={total / max(1, count):.6f}{cache_text}")
        if step % args.accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
        final_train = total / max(1, count)
        total_train_examples = count
        val_mse = evaluate(model, val, args.batch_size, device, image_cache=image_cache)
        epoch_record = {
            "epoch": display_epoch,
            "train_mse": final_train,
            "val_mse": val_mse,
            "examples": {"train": total_train_examples, "val": len(val)},
        }
        history.append(epoch_record)
        if args.save_each_epoch:
            epoch_out = out.with_name(f"{out.stem}_epoch{display_epoch}{out.suffix}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "image_dim": 768,
                        "width": 1024,
                        "target_dim": 1024,
                        "vocab_size": 32128,
                        "source": "image_embed_pre_to_anima_conditioning",
                        "text_translator_anchor": args.text_translator_anchor,
                        "init_checkpoint": args.init_checkpoint,
                    "guard_replay": args.guard_replay,
                    "guard_replay_weight": args.guard_replay_weight,
                    "epoch_offset": args.epoch_offset,
                },
                    "train_mse": final_train,
                    "val_mse": val_mse,
                    "epoch": display_epoch,
                    "history": list(history),
                    "examples": {"train": total_train_examples, "val": len(val)},
                    "image_cache": image_cache.stats if image_cache else None,
                },
                epoch_out,
            )
            print("epoch checkpoint:", epoch_out)
        print(f"epoch {display_epoch}/{args.epoch_offset + args.epochs} train_mse={final_train:.6f} val_mse={val_mse:.6f}")

    final_val = evaluate(model, val, args.batch_size, device, image_cache=image_cache)
    payload = {
        "model": model.state_dict(),
        "config": {
            "image_dim": 768,
            "width": 1024,
            "target_dim": 1024,
            "vocab_size": 32128,
            "source": "image_embed_pre_to_anima_conditioning",
            "text_translator_anchor": args.text_translator_anchor,
            "init_checkpoint": args.init_checkpoint,
            "guard_replay": args.guard_replay,
            "guard_replay_weight": args.guard_replay_weight,
            "epoch_offset": args.epoch_offset,
        },
        "train_mse": final_train,
        "val_mse": final_val,
        "epoch": args.epoch_offset + args.epochs,
        "history": history,
        "examples": {"train": total_train_examples, "val": len(val)},
        "guard_replay": {
            "path": args.guard_replay,
            "indices": sorted(guard_indices),
            "weight": args.guard_replay_weight,
        },
        "image_cache": image_cache.stats if image_cache else None,
    }
    torch.save(payload, out)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "stage": "image_state_conditioning_train",
                    "checkpoint": str(out),
                    "text_translator_anchor": args.text_translator_anchor,
                    "train_mse": final_train,
                    "val_mse": final_val,
                    "epochs": args.epochs,
                    "epoch_offset": args.epoch_offset,
                    "final_epoch": args.epoch_offset + args.epochs,
                    "history": history,
                    "examples": {"train": total_train_examples, "val": len(val)},
                    "guard_replay": {
                        "path": args.guard_replay,
                        "indices": sorted(guard_indices),
                        "weight": args.guard_replay_weight,
                    },
                    "image_cache": image_cache.stats if image_cache else None,
                    "device": str(device),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("report:", report)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
