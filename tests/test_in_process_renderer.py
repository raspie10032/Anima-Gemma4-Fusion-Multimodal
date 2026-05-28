from pathlib import Path

import torch

from gemmanima.core.schemas import ConditioningBundle, GenerationPlan, RenderResult
from gemmanima.modules.in_process_anima_renderer import InProcessAnimaRendererAdapter


class FakeHiddenProvider:
    def encode_image_intent(self, source_text, span_text):
        return torch.ones(1, 2, 1536)


class FakeT5Provider:
    def encode_ids_weights(self, prompt):
        return torch.tensor([1, 2], dtype=torch.int32), torch.tensor([1.0, 1.0], dtype=torch.float32)


class FakeSamplerRuntime:
    def __init__(self):
        self.requests = []

    def sample_to_file(self, request):
        self.requests.append(request)
        request.output_path.write_bytes(b"png")
        return request.output_path


def test_in_process_renderer_builds_conditioning_and_samples(tmp_path: Path) -> None:
    sampler_runtime = FakeSamplerRuntime()
    renderer = InProcessAnimaRendererAdapter(
        output_root=tmp_path,
        hidden_provider=FakeHiddenProvider(),
        t5_provider=FakeT5Provider(),
        sampler_runtime=sampler_runtime,
    )

    result = renderer.generate(
        GenerationPlan(prompt="bright forest", width=512, height=512, steps=12, cfg=4.5, seed=123),
        ConditioningBundle(
            source="trained_hiddenstage_bridge",
            metadata={
                "hiddenstage_source_text": "assistant: <image_intent>\nbright forest\n</image_intent>",
            },
        ),
    )

    assert isinstance(result, RenderResult)
    assert result.output_path.exists()
    request = sampler_runtime.requests[0]
    assert request.seed == 123
    assert request.positive[0][0].shape == (1, 2, 1536)
    assert request.positive[0][1]["t5xxl_ids"].tolist() == [1, 2]


def test_in_process_renderer_attaches_adapter_when_building_runtime(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class FakeModel:
        pass

    class FakeVAE:
        pass

    def fake_load_model_vae(diffusion_model_path, vae_path, unet_dtype):
        calls.append(("load", diffusion_model_path, vae_path, unet_dtype))
        return FakeModel(), FakeVAE()

    def fake_attach(model, **kwargs):
        calls.append(("attach", model, kwargs["checkpoint"]))

    def fake_sampler():
        return object()

    def fake_bootstrap():
        calls.append(("bootstrap",))

    monkeypatch.setattr("gemmanima.modules.in_process_anima_renderer.bootstrap_comfy", fake_bootstrap)
    monkeypatch.setattr("gemmanima.modules.in_process_anima_renderer.load_anima_model_vae", fake_load_model_vae)
    monkeypatch.setattr("gemmanima.modules.in_process_anima_renderer.attach_hiddenstage_adapter", fake_attach)
    monkeypatch.setattr("gemmanima.modules.in_process_anima_renderer.build_comfy_sampler", fake_sampler)
    renderer = InProcessAnimaRendererAdapter(
        output_root=tmp_path,
        hidden_provider=FakeHiddenProvider(),
        t5_provider=FakeT5Provider(),
    )

    renderer._ensure_runtime()

    assert calls[0] == ("bootstrap",)
    assert calls[1][0] == "load"
    assert str(calls[1][1]).endswith("anima-base-v1.0.safetensors")
    assert str(calls[1][2]).endswith("qwen_image_vae.safetensors")
    assert calls[1][3] == "fp8_e4m3fn_fast"
    assert calls[2][0] == "attach"
    assert str(calls[2][2]).endswith("kv_proj_hiddenstage_planner_v2.pt")
