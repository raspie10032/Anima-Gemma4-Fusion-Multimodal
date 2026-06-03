from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from gemmanima.core.model_paths import default_model_root

DEFAULT_GEMMA_PREFIX = "model.language_model."


class GemmaHiddenRuntime(Protocol):
    def encode(self, text: str) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class GemmaHiddenConfig:
    gemma_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("GEMMANIMA_GEMMA_HF_DIR", str(default_model_root() / "gemma_core_hf"))
        )
    )
    prefix: str = DEFAULT_GEMMA_PREFIX
    dtype: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_GEMMA_HIDDEN_DTYPE", "bfloat16"))
    device: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_GEMMA_HIDDEN_DEVICE", "cuda"))
    embed_on_gpu: bool = field(default_factory=lambda: _env_bool("GEMMA_EMBED_ON_GPU", True))


class GemmaHiddenProvider:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    def encode_image_intent(self, source_text: str, span_text: str) -> torch.Tensor:
        if hasattr(self.runtime, "encode_image_intent"):
            hidden = self.runtime.encode_image_intent(source_text, span_text)
        elif hasattr(self.runtime, "encode"):
            hidden = self.runtime.encode(span_text)
        else:
            raise TypeError("Gemma runtime must provide encode_image_intent(...) or encode(...)")
        return normalize_gemma_hidden(hidden)


def normalize_gemma_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim != 3:
        raise ValueError(f"Gemma hidden must be [S,1536] or [1,S,1536], got {tuple(hidden.shape)}")
    if hidden.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported in v1, got {hidden.shape[0]}")
    if hidden.shape[-1] != 1536:
        raise ValueError(f"Gemma hidden dim must be 1536, got {hidden.shape[-1]}")
    return hidden.to(torch.float32)


class CpuEmbeddingProxy(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Module, device: str) -> None:
        super().__init__()
        self.embedding = embedding
        self.device = device

    def forward(self, x, **kwargs):
        return self.embedding(x.to("cpu"), **kwargs).to(self.device)


class GemmaTextRuntime:
    """Repo-native port of the legacy GemmaText hidden-state loader."""

    def __init__(self, config: GemmaHiddenConfig | None = None) -> None:
        self.config = config or GemmaHiddenConfig()
        self.device = self.config.device
        self.input_device = self.device if self.config.embed_on_gpu else "cpu"
        self.dtype = _torch_dtype(self.config.dtype)
        self.model, self.tokenizer = self._load()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        ids = [2] + self.tokenizer.encode(text, add_special_tokens=False).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.input_device)
        out = self.model(x=input_ids, input_ids=input_ids, dtype=self.dtype)
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        return hidden[0]

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> list[torch.Tensor]:
        encoded = [[2] + item.ids for item in self.tokenizer.encode_batch(texts, add_special_tokens=False)]
        lengths = [len(ids) for ids in encoded]
        max_len = max(lengths)
        padded = [ids + [0] * (max_len - len(ids)) for ids in encoded]
        mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.input_device)
        attention_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        out = self.model(x=input_ids, input_ids=input_ids, attention_mask=attention_mask, dtype=self.dtype)
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        return [hidden[i, : lengths[i]] for i in range(len(texts))]

    def _load(self):
        from safetensors import safe_open
        from tokenizers import Tokenizer

        import comfy.ops
        from comfy.text_encoders.gemma4 import Gemma4Transformer, Gemma4_E2B_Config

        ops = comfy.ops.disable_weight_init
        model = Gemma4Transformer(Gemma4_E2B_Config(), device="cpu", dtype=self.dtype, ops=ops)
        state_dict = {}
        with safe_open(str(self.config.gemma_dir / "model.safetensors"), "pt") as handle:
            for key in handle.keys():
                if key.startswith(self.config.prefix):
                    state_dict[key[len(self.config.prefix) :]] = handle.get_tensor(key)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        for name, child in model.named_children():
            if self.config.embed_on_gpu or name not in ("embed_tokens", "embed_tokens_per_layer"):
                child.to(self.device)
        for buffer_name, buffer in list(model.named_buffers(recurse=False)):
            setattr(model, buffer_name, buffer.to(self.device))
        if not self.config.embed_on_gpu:
            model.embed_tokens = CpuEmbeddingProxy(model.embed_tokens, self.device)
            model.embed_tokens_per_layer = CpuEmbeddingProxy(model.embed_tokens_per_layer, self.device)
        tokenizer = Tokenizer.from_file(str(self.config.gemma_dir / "tokenizer.json"))
        return model, tokenizer


def gemma_hidden_environment(config: GemmaHiddenConfig | None = None) -> dict[str, object]:
    resolved = config or GemmaHiddenConfig()
    return {
        "gemma_dir": str(resolved.gemma_dir),
        "model_safetensors": (resolved.gemma_dir / "model.safetensors").exists(),
        "tokenizer_json": (resolved.gemma_dir / "tokenizer.json").exists(),
        "device": resolved.device,
        "embed_on_gpu": resolved.embed_on_gpu,
        "env_gemma_embed_on_gpu": os.environ.get("GEMMA_EMBED_ON_GPU"),
    }


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported Gemma dtype: {name}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}
