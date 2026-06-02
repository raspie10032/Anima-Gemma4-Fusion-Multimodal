from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ImageStateToConditioningTranslator(nn.Module):
    def __init__(
        self,
        *,
        image_dim: int = 768,
        width: int = 1024,
        target_dim: int = 1024,
        vocab_size: int = 32128,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.t5_embed = nn.Embedding(vocab_size, width)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, target_dim),
        )

    def forward(
        self,
        image_state: torch.Tensor,
        t5_ids: torch.Tensor,
        *,
        image_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.image_proj(image_state)
        query = self.t5_embed(t5_ids.clamp_min(0).clamp_max(self.t5_embed.num_embeddings - 1))
        key_padding_mask = None
        if image_mask is not None:
            key_padding_mask = ~image_mask
        attn, _ = self.cross_attn(query, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        if target_mask is not None:
            attn = attn * target_mask.unsqueeze(-1)
        out = self.out(attn)
        if target_mask is not None:
            out = out * target_mask.unsqueeze(-1)
        return out


def load_image_state_translator(
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    image_dim: int = 768,
    width: int = 1024,
    target_dim: int = 1024,
    vocab_size: int = 32128,
    heads: int = 8,
    dropout: float = 0.0,
) -> ImageStateToConditioningTranslator:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"image-state translator checkpoint not found: {path}")
    checkpoint_data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint_data:
        raise KeyError(f"image-state translator checkpoint missing 'model': {path}")
    model = ImageStateToConditioningTranslator(
        image_dim=image_dim,
        width=width,
        target_dim=target_dim,
        vocab_size=vocab_size,
        heads=heads,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint_data["model"], strict=True)
    model.eval().to(device=device, dtype=dtype)
    return model
