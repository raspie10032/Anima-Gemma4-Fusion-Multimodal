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
    steps: int
    cfg: float
    width: int | None = None
    height: int | None = None
    size: int | None = None
    sampler: str = "euler_ancestral"
    scheduler: str = "sgm_uniform"
    denoise: float = 1.0
    tiled_vae: bool = True


class AnimaSamplerRuntime:
    def __init__(self, *, model: Any, vae: Any, sampler: Any) -> None:
        self.model = model
        self.vae = vae
        self.sampler = sampler

    def sample_to_file(self, request: SamplerRequest) -> Path:
        width = request.width or request.size
        height = request.height or request.size
        if width is None or height is None:
            raise ValueError("sampler request requires width/height or size")
        latent = make_empty_latent(width, height)
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
        decoded = decode_samples(self.vae, sampled["samples"], tiled_vae=request.tiled_vae)
        save_image_tensor(decoded, request.output_path)
        return request.output_path


def make_empty_latent(width: int, height: int | None = None, *, batch_size: int = 1) -> dict[str, torch.Tensor]:
    resolved_height = height if height is not None else width
    latent_height = max(1, resolved_height // 8)
    latent_width = max(1, width // 8)
    device: torch.device | str = "cpu"
    dtype = torch.float32
    try:
        import comfy.model_management

        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()
    except Exception:
        pass
    latent = torch.zeros([batch_size, 4, latent_height, latent_width], device=device, dtype=dtype)
    return {"samples": latent, "downscale_ratio_spacial": 8}


def decode_samples(vae: Any, samples: torch.Tensor, *, tiled_vae: bool = True) -> torch.Tensor:
    with torch.inference_mode():
        if tiled_vae and hasattr(vae, "decode_tiled"):
            return vae.decode_tiled(samples, **vae_tile_kwargs(vae, samples))
        return vae.decode(samples)


def vae_tile_kwargs(
    vae: Any,
    samples: torch.Tensor,
    *,
    target_tile_pixels: int = 512,
    minimum_tile_pixels: int = 256,
    overlap_pixels: int = 64,
) -> dict[str, int]:
    scale = _vae_spatial_scale(vae)
    latent_width = int(samples.shape[-1])
    latent_height = int(samples.shape[-2])
    tile_x = _pixel_multiple_tile_to_latent(
        latent_width,
        scale=scale,
        target_tile_pixels=target_tile_pixels,
        minimum_tile_pixels=minimum_tile_pixels,
    )
    tile_y = _pixel_multiple_tile_to_latent(
        latent_height,
        scale=scale,
        target_tile_pixels=target_tile_pixels,
        minimum_tile_pixels=minimum_tile_pixels,
    )
    overlap = min(max(1, min(tile_x, tile_y) // 4), max(1, overlap_pixels // scale))
    overlap = max(1, min(overlap, tile_x - 1, tile_y - 1))
    return {"tile_x": tile_x, "tile_y": tile_y, "overlap": overlap}


def _vae_spatial_scale(vae: Any) -> int:
    scale = getattr(vae, "spacial_compression_decode", None)
    if callable(scale):
        try:
            return max(1, int(scale()))
        except Exception:
            pass
    return 8


def _pixel_multiple_tile_to_latent(
    latent_dim: int,
    *,
    scale: int,
    target_tile_pixels: int,
    minimum_tile_pixels: int,
) -> int:
    full_pixels = max(scale, latent_dim * scale)
    desired_pixels = min(full_pixels, target_tile_pixels)
    if full_pixels >= minimum_tile_pixels:
        desired_pixels = max(minimum_tile_pixels, desired_pixels)
    desired_pixels = max(64, (desired_pixels // 64) * 64)
    return max(1, min(latent_dim, desired_pixels // scale))


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
