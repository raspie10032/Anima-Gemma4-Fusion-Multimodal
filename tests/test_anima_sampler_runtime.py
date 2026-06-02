from pathlib import Path

import torch

from gemmanima.rendering.anima_sampler import (
    AnimaSamplerRuntime,
    SamplerRequest,
    decode_samples,
    make_empty_latent,
    save_image_tensor,
    vae_tile_kwargs,
)


class FakeVAE:
    def __init__(self) -> None:
        self.encoded = []
        self.decoded = []

    def encode(self, tensor):
        self.encoded.append(tensor.shape)
        return torch.zeros(1, 4, 8, 8)

    def decode(self, samples):
        self.decoded.append(samples.shape)
        return torch.ones(1, 8, 8, 3)

    def spacial_compression_decode(self):
        return 8


class FakeTiledVAE(FakeVAE):
    def __init__(self) -> None:
        super().__init__()
        self.tiled_decoded = []

    def decode_tiled(self, samples, **kwargs):
        self.tiled_decoded.append((samples.shape, kwargs))
        return torch.ones(1, 8, 8, 3)


class FakeSampler:
    def __init__(self) -> None:
        self.calls = []

    def sample(self, model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=1.0):
        self.calls.append((seed, steps, cfg, sampler, scheduler, denoise, latent["samples"].shape))
        return ({"samples": torch.ones(1, 4, 8, 8)},)


def test_make_empty_latent_uses_rectangular_latent_tensor() -> None:
    latent = make_empty_latent(64, 96)

    assert latent["samples"].shape == (1, 4, 12, 8)
    assert latent["downscale_ratio_spacial"] == 8


def test_save_image_tensor_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "image.png"

    save_image_tensor(torch.ones(1, 8, 8, 3), path)

    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG")


def test_decode_samples_prefers_tiled_vae() -> None:
    vae = FakeTiledVAE()

    decoded = decode_samples(vae, torch.ones(1, 4, 8, 8), tiled_vae=True)

    assert decoded.shape == (1, 8, 8, 3)
    assert vae.decoded == []
    assert vae.tiled_decoded == [
        (torch.Size([1, 4, 8, 8]), {"tile_x": 8, "tile_y": 8, "overlap": 2})
    ]


def test_vae_tile_kwargs_uses_64_pixel_multiple_tiles() -> None:
    kwargs = vae_tile_kwargs(FakeVAE(), torch.ones(1, 4, 128, 128))

    assert kwargs == {"tile_x": 64, "tile_y": 64, "overlap": 8}


def test_vae_tile_kwargs_handles_rectangular_resolution() -> None:
    kwargs = vae_tile_kwargs(FakeVAE(), torch.ones(1, 4, 152, 104))

    assert kwargs == {"tile_x": 64, "tile_y": 64, "overlap": 8}


def test_anima_sampler_runtime_samples_decodes_and_saves(tmp_path: Path) -> None:
    vae = FakeVAE()
    sampler = FakeSampler()
    runtime = AnimaSamplerRuntime(model=object(), vae=vae, sampler=sampler)
    output = tmp_path / "render.png"

    result = runtime.sample_to_file(
        SamplerRequest(
            positive=[["pos"]],
            negative=[["neg"]],
            output_path=output,
            seed=123,
            width=64,
            height=96,
            steps=12,
            cfg=4.5,
            sampler="euler_ancestral",
            scheduler="sgm_uniform",
        )
    )

    assert result == output
    assert output.exists()
    assert vae.encoded == []
    assert sampler.calls == [(123, 12, 4.5, "euler_ancestral", "sgm_uniform", 1.0, torch.Size([1, 4, 12, 8]))]
    assert vae.decoded == [torch.Size([1, 4, 8, 8])]
