from pathlib import Path

import torch

from gemmanima.rendering.anima_sampler import (
    AnimaSamplerRuntime,
    SamplerRequest,
    make_gray_latent,
    save_image_tensor,
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


class FakeSampler:
    def __init__(self) -> None:
        self.calls = []

    def sample(self, model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=1.0):
        self.calls.append((seed, steps, cfg, sampler, scheduler, denoise, latent["samples"].shape))
        return ({"samples": torch.ones(1, 4, 8, 8)},)


def test_make_gray_latent_uses_square_rgb_tensor() -> None:
    vae = FakeVAE()

    latent = make_gray_latent(vae, 64)

    assert vae.encoded == [torch.Size([1, 64, 64, 3])]
    assert latent["samples"].shape == (1, 4, 8, 8)


def test_save_image_tensor_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "image.png"

    save_image_tensor(torch.ones(1, 8, 8, 3), path)

    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG")


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
            size=64,
            steps=12,
            cfg=4.5,
            sampler="euler_ancestral",
            scheduler="sgm_uniform",
        )
    )

    assert result == output
    assert output.exists()
    assert sampler.calls == [(123, 12, 4.5, "euler_ancestral", "sgm_uniform", 1.0, torch.Size([1, 4, 8, 8]))]
    assert vae.decoded == [torch.Size([1, 4, 8, 8])]
