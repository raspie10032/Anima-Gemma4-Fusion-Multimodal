from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass(frozen=True)
class SamplerRequest:
    positive: list[list[object]]
    negative: list[list[object]]
    output_path: Path
    seed: int
    size: int
    steps: int
    cfg: float
    sampler: str = "euler_ancestral"
    scheduler: str = "sgm_uniform"
    denoise: float = 1.0


class AnimaSamplerRuntime:
    def __init__(self, *, model: Any, vae: Any, sampler: Any) -> None:
        self.model = model
        self.vae = vae
        self.sampler = sampler

    def sample_to_file(self, request: SamplerRequest) -> Path:
        latent = make_gray_latent(self.vae, request.size)
        sampled = self.sampler.sample(
            self.model,
            request.seed,
            request.steps,
            request.cfg,
            request.sampler,
            request.scheduler,
            request.positive,
            request.negative,
            latent,
            denoise=request.denoise,
        )[0]
        decoded = self.vae.decode(sampled["samples"])
        save_image_tensor(decoded, request.output_path)
        return request.output_path


def make_gray_latent(vae: Any, size: int) -> dict[str, torch.Tensor]:
    gray = torch.full((1, size, size, 3), 0.5, dtype=torch.float32)
    return {"samples": vae.encode(gray)}


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().to("cpu", torch.float32)
    while image.dim() > 3:
        image = image[0]
    array = ((image.clamp(0, 1) * 255 + 0.5).to(torch.uint8)).contiguous().numpy()
    Image.fromarray(array, "RGB").save(output_path)


def model_options_for_unet_dtype(unet_dtype: str) -> dict[str, object]:
    if unet_dtype == "default":
        return {}
    if unet_dtype == "fp8_e4m3fn":
        return {"dtype": torch.float8_e4m3fn}
    if unet_dtype == "fp8_e4m3fn_fast":
        return {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True}
    if unet_dtype == "fp8_e5m2":
        return {"dtype": torch.float8_e5m2}
    raise ValueError(f"unsupported unet dtype: {unet_dtype}")


def load_anima_model_vae(
    *,
    diffusion_model_path: str | Path | None = None,
    vae_path: str | Path | None = None,
    unet_dtype: str = "default",
) -> tuple[object, object]:
    import comfy.sd
    import comfy.utils
    import folder_paths

    dm_path = Path(diffusion_model_path) if diffusion_model_path else folder_paths.get_full_path(
        "diffusion_models", "anima-base-v1.0.safetensors"
    )
    resolved_vae_path = Path(vae_path) if vae_path else folder_paths.get_full_path("vae", "qwen_image_vae.safetensors")
    model = comfy.sd.load_diffusion_model(str(dm_path), model_options=model_options_for_unet_dtype(unet_dtype))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(resolved_vae_path)))
    return model, vae


def build_comfy_sampler() -> object:
    import nodes

    return nodes.KSampler()


def anima_sampler_environment() -> dict[str, object]:
    return {
        "sampler_module": True,
        "supports_unet_dtypes": ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
        "default_sampler": "euler_ancestral",
        "default_scheduler": "sgm_uniform",
    }
