from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import torch
from torch import nn

from gemmanima.modules.in_process_anima_renderer import NEGATIVE_PROMPT
from gemmanima.rendering.image_state_translator import load_image_state_translator
from gemmanima.rendering.t5_tokenizer import T5TokenizerProvider, build_t5_tokenizer_provider


FusionMode = Literal["image_only", "hidden_fusion", "conditioning_fusion"]
TranslatorLoader = Callable[[Path, str, torch.dtype], nn.Module]


class ImageStateConditioningError(RuntimeError):
    pass


class HiddenStateFuser(Protocol):
    def __call__(self, text_state: torch.Tensor, image_state: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class ConditioningBlend:
    text_weight: float = 0.55
    image_weight: float = 0.45


@dataclass(frozen=True)
class ImageStateConditioningConfig:
    checkpoint: Path
    device: str = "cuda"
    dtype: str = "bfloat16"
    image_dim: int = 768
    target_dim: int = 1024
    default_mode: FusionMode = "image_only"
    blend: ConditioningBlend = field(default_factory=ConditioningBlend)
    conditioning_tokens: int = 512


@dataclass(frozen=True)
class ConditioningPayload:
    positive: list[list[object]]
    negative: list[list[object]]
    diagnostics: dict[str, Any]


class TokenConcatFuser:
    def __call__(self, text_state: torch.Tensor, image_state: torch.Tensor) -> torch.Tensor:
        text = _ensure_batched(text_state)
        image = _ensure_batched(image_state).to(device=text.device, dtype=text.dtype)
        if text.shape[-1] > image.shape[-1]:
            pad = torch.zeros(
                (image.shape[0], image.shape[1], text.shape[-1] - image.shape[-1]),
                dtype=image.dtype,
                device=image.device,
            )
            image = torch.cat([image, pad], dim=-1)
        elif text.shape[-1] < image.shape[-1]:
            image = image[..., : text.shape[-1]]
        return torch.cat([text, image], dim=1)


class ImageStateConditioningEngine:
    def __init__(
        self,
        config: ImageStateConditioningConfig,
        *,
        t5_provider: T5TokenizerProvider | None = None,
        translator_loader: TranslatorLoader | None = None,
        text_adapter: nn.Module | None = None,
        hidden_fuser: HiddenStateFuser | None = None,
    ) -> None:
        self.config = config
        self.t5_provider = t5_provider
        self.translator_loader = translator_loader or _default_translator_loader
        self.text_adapter = text_adapter
        self.hidden_fuser = hidden_fuser or TokenConcatFuser()
        self._translator: nn.Module | None = None

    def attach_to_model(self, model: Any, *, mode: FusionMode = "image_only") -> nn.Module:
        if mode == "conditioning_fusion":
            raise ImageStateConditioningError("conditioning_fusion uses precomputed cross-attn and does not attach an llm_adapter")
        if mode == "hidden_fusion":
            if self.text_adapter is None:
                raise ImageStateConditioningError("hidden_fusion requires a text_adapter to be attached to Anima")
            adapter = self.text_adapter
        else:
            adapter = self._ensure_translator()
        model.model.diffusion_model.llm_adapter = adapter
        return adapter

    def condition_from_tensor(
        self,
        image_state: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str = NEGATIVE_PROMPT,
        metadata: dict[str, Any] | None = None,
    ) -> ConditioningPayload:
        return self.condition_from_states(
            image_state=image_state,
            prompt=prompt,
            negative_prompt=negative_prompt,
            metadata=metadata,
            mode="image_only",
        )

    def condition_from_states(
        self,
        *,
        image_state: torch.Tensor,
        prompt: str,
        negative_prompt: str = NEGATIVE_PROMPT,
        text_state: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
        mode: FusionMode | None = None,
    ) -> ConditioningPayload:
        selected_mode = mode or self.config.default_mode
        provider = self._ensure_t5_provider()
        image = _validate_image_state(image_state, self.config.image_dim)
        pos_ids, pos_weights = provider.encode_ids_weights(prompt)
        neg_ids, neg_weights = provider.encode_ids_weights(negative_prompt)
        meta = dict(metadata or {})
        meta.update(
            {
                "fusion_mode": selected_mode,
                "prompt": prompt,
                "conditioning_engine": "image_state_conditioning",
            }
        )

        if selected_mode == "image_only":
            positive = [[image, {**meta, "t5xxl_ids": pos_ids, "t5xxl_weights": pos_weights}]]
            negative = [[_zero_image_state(image), {"fusion_mode": selected_mode, "t5xxl_ids": neg_ids, "t5xxl_weights": neg_weights}]]
        elif selected_mode == "hidden_fusion":
            if text_state is None:
                raise ImageStateConditioningError("hidden_fusion requires text_state")
            fused = _ensure_batched(self.hidden_fuser(text_state, image))
            positive = [[fused, {**meta, "conditioning_stage": "pre_adapter", "t5xxl_ids": pos_ids, "t5xxl_weights": pos_weights}]]
            negative = [[torch.zeros((1, 1, fused.shape[-1]), dtype=fused.dtype), {"fusion_mode": selected_mode, "t5xxl_ids": neg_ids, "t5xxl_weights": neg_weights}]]
        elif selected_mode == "conditioning_fusion":
            if text_state is None:
                raise ImageStateConditioningError("conditioning_fusion requires text_state")
            if self.text_adapter is None:
                raise ImageStateConditioningError("conditioning_fusion requires a text_adapter")
            fused_conditioning = self._blend_conditioning_outputs(text_state, image, pos_ids, pos_weights)
            positive = [[fused_conditioning, {**meta, "conditioning_stage": "post_adapter"}]]
            negative = [[torch.zeros((1, max(1, int(neg_ids.numel())), fused_conditioning.shape[-1]), dtype=fused_conditioning.dtype), {"fusion_mode": selected_mode, "conditioning_stage": "post_adapter"}]]
        else:
            raise ImageStateConditioningError(f"unsupported fusion mode: {selected_mode}")

        diagnostics = {
            "mode": selected_mode,
            "checkpoint": str(self.config.checkpoint),
            "image_state": _tensor_stats(image),
            "positive_tokens": int(pos_ids.numel()),
            "negative_tokens": int(neg_ids.numel()),
            "metadata_keys": sorted(meta),
        }
        if text_state is not None:
            diagnostics["text_state"] = _tensor_stats(_ensure_batched(text_state.float()))
        if selected_mode == "hidden_fusion":
            diagnostics["hidden_fusion_policy"] = "concat_text_tokens_with_image_tokens_after_pad_or_truncate_to_text_dim"
        if selected_mode == "conditioning_fusion":
            diagnostics["blend"] = {
                "text_weight": self.config.blend.text_weight,
                "image_weight": self.config.blend.image_weight,
            }
            diagnostics["fused_conditioning"] = _tensor_stats(positive[0][0])
        return ConditioningPayload(positive=positive, negative=negative, diagnostics=diagnostics)

    def condition_from_record(
        self,
        record: dict[str, Any],
        *,
        prompt: str | None = None,
        negative_prompt: str = NEGATIVE_PROMPT,
        mode: FusionMode | None = None,
        text_state: torch.Tensor | None = None,
    ) -> ConditioningPayload:
        image_path = record.get("image_embed_pre")
        if not image_path:
            raise ImageStateConditioningError("record missing image_embed_pre")
        image = torch.load(image_path, map_location="cpu", weights_only=False).float()
        selected_prompt = prompt or record.get("visible_prompt") or record.get("teacher_prompt") or record.get("text")
        if not selected_prompt:
            raise ImageStateConditioningError("record missing prompt text")
        return self.condition_from_states(
            image_state=image,
            text_state=text_state,
            prompt=str(selected_prompt),
            negative_prompt=negative_prompt,
            metadata={
                "source_idx": record.get("idx"),
                "source_id": record.get("source_id"),
                "source_image": record.get("image"),
                "image_embed_pre": str(image_path),
            },
            mode=mode,
        )

    def _ensure_t5_provider(self) -> T5TokenizerProvider:
        if self.t5_provider is None:
            self.t5_provider = build_t5_tokenizer_provider()
        return self.t5_provider

    def _ensure_translator(self) -> nn.Module:
        if self._translator is None:
            self._translator = self.translator_loader(self.config.checkpoint, self.config.device, _dtype_from_name(self.config.dtype))
        return self._translator

    def _blend_conditioning_outputs(
        self,
        text_state: torch.Tensor,
        image_state: torch.Tensor,
        t5_ids: torch.Tensor,
        t5_weights: torch.Tensor,
    ) -> torch.Tensor:
        ids = t5_ids.to(device=self.config.device, dtype=torch.long).unsqueeze(0)
        weights = t5_weights.to(device=self.config.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        text_dtype = _module_dtype(self.text_adapter)
        image_dtype = _module_dtype(self._ensure_translator())
        text = _ensure_batched(text_state).to(device=self.config.device, dtype=text_dtype)
        image = _ensure_batched(image_state).to(device=self.config.device, dtype=image_dtype)
        text_conditioning = self.text_adapter(text, ids) * weights.to(dtype=text_dtype)
        image_conditioning = self._ensure_translator()(image, ids) * weights.to(dtype=image_dtype)
        text_conditioning, image_conditioning = _align_conditioning_pair(text_conditioning, image_conditioning)
        blended = self.config.blend.text_weight * text_conditioning + self.config.blend.image_weight * image_conditioning
        blended = _pad_tokens(blended, self.config.conditioning_tokens)
        return blended.detach().to("cpu", torch.float32)


def image_state_engine_status(
    *,
    checkpoint: str | Path,
    subset: str | Path | None = None,
    train_report: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    subset_path = Path(subset) if subset is not None else None
    report_path = Path(train_report) if train_report is not None else None
    status: dict[str, Any] = {
        "engine": "image_state_conditioning",
        "checkpoint": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "subset": str(subset_path) if subset_path else None,
        "subset_exists": subset_path.exists() if subset_path else None,
        "train_report_path": str(report_path) if report_path else None,
        "train_report_exists": report_path.exists() if report_path else None,
        "supported_modes": ["image_only", "hidden_fusion", "conditioning_fusion"],
        "default_protected": True,
    }
    if report_path and report_path.exists():
        status["train_report"] = _summarize_train_report(report_path)
    status["ready"] = bool(status["checkpoint_exists"] and (subset_path is None or status["subset_exists"]))
    return status


def _default_translator_loader(path: Path, device: str, dtype: torch.dtype) -> nn.Module:
    return load_image_state_translator(path, device=device, dtype=dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ImageStateConditioningError(f"unsupported dtype: {name}")


def _ensure_batched(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ImageStateConditioningError(f"expected a 2D or 3D state tensor, got shape {list(tensor.shape)}")


def _validate_image_state(tensor: torch.Tensor, image_dim: int) -> torch.Tensor:
    state = _ensure_batched(tensor.float())
    if state.shape[-1] != image_dim:
        raise ImageStateConditioningError(f"expected image state last dim {image_dim}, got {state.shape[-1]}")
    if not torch.isfinite(state).all():
        raise ImageStateConditioningError("image state contains non-finite values")
    if state.shape[1] <= 0:
        raise ImageStateConditioningError("image state has no tokens")
    return state


def _zero_image_state(image: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, 1, image.shape[-1]), dtype=image.dtype)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to("cpu", torch.float32)
    return {
        "shape": list(cpu.shape),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
        "absmax": float(cpu.abs().max().item()),
        "finite": bool(torch.isfinite(cpu).all().item()),
    }


def _align_conditioning_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != 3 or b.dim() != 3:
        raise ImageStateConditioningError(f"conditioning adapters must return [B, T, C], got {list(a.shape)} and {list(b.shape)}")
    if a.shape[0] != b.shape[0] or a.shape[-1] != b.shape[-1]:
        raise ImageStateConditioningError(f"conditioning outputs are incompatible: {list(a.shape)} vs {list(b.shape)}")
    max_tokens = max(a.shape[1], b.shape[1])
    return _pad_tokens(a, max_tokens), _pad_tokens(b, max_tokens)


def _pad_tokens(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    if tensor.shape[1] == tokens:
        return tensor
    pad = torch.zeros((tensor.shape[0], tokens - tensor.shape[1], tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1)


def _summarize_train_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("history") or []
    val_items = [float(item["val_mse"]) for item in history if "val_mse" in item]
    return {
        "epochs": len(history),
        "best_val_mse": min(val_items) if val_items else None,
        "final_checkpoint": data.get("final_checkpoint") or data.get("checkpoint"),
    }


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32
