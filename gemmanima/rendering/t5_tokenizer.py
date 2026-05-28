from __future__ import annotations

from typing import Protocol

import torch


class WeightedTokenizer(Protocol):
    def tokenize_with_weights(self, prompt: str, return_word_ids: bool = False):
        ...


class T5TokenizerProvider:
    def __init__(self, tokenizer: WeightedTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode_ids_weights(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        token_weight_pairs = self.tokenizer.tokenize_with_weights(prompt, return_word_ids=False)[0]
        ids = torch.tensor([pair[0] for pair in token_weight_pairs], dtype=torch.int32)
        weights = torch.tensor([pair[1] for pair in token_weight_pairs], dtype=torch.float32)
        return ids, weights


def build_t5_tokenizer_provider() -> T5TokenizerProvider:
    from comfy.text_encoders.anima import T5XXLTokenizer

    return T5TokenizerProvider(T5XXLTokenizer())


def t5_tokenizer_environment(*, load_tokenizer: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider_module": True,
        "tokenizer_load_attempted": load_tokenizer,
    }
    if not load_tokenizer:
        return payload
    try:
        provider = build_t5_tokenizer_provider()
    except Exception as exc:
        payload.update(
            {
                "tokenizer_loaded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        payload.update(
            {
                "tokenizer_loaded": True,
                "tokenizer_class": type(provider.tokenizer).__name__,
            }
        )
    return payload
