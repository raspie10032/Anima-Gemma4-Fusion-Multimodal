import json
from pathlib import Path

import torch

from gemmanima.core.schemas import ConditioningBundle, GenerationPlan
from gemmanima.modules.image_state_anima_renderer import ImageStateAnimaRendererAdapter
from gemmanima.rendering.image_state_engine import ConditioningPayload


class FakeEngine:
    def __init__(self) -> None:
        self.attached = []
        self.calls = []

    def attach_to_model(self, model):
        self.attached.append(model)
        return "image-adapter"

    def condition_from_record(self, record, *, prompt, negative_prompt, mode, text_state=None):
        self.calls.append((record["idx"], prompt, negative_prompt, mode, text_state))
        return ConditioningPayload(
            positive=[["positive", {"mode": mode}]],
            negative=[["negative", {"mode": mode}]],
            diagnostics={"mode": mode, "record_idx": record["idx"]},
        )


class FakeSamplerRuntime:
    def __init__(self) -> None:
        self.requests = []

    def sample_to_file(self, request):
        self.requests.append(request)
        request.output_path.write_bytes(b"png")
        return request.output_path


def test_image_state_renderer_generates_from_record_without_gpu_runtime(tmp_path: Path) -> None:
    engine = FakeEngine()
    sampler = FakeSamplerRuntime()
    renderer = ImageStateAnimaRendererAdapter(output_root=tmp_path, engine=engine, sampler_runtime=sampler)

    result = renderer.generate_from_record(
        GenerationPlan(prompt="fusion prompt", width=512, height=512, steps=8, cfg=3.0, seed=77),
        {"idx": 12, "image_embed_pre": "x.pt", "text": "record prompt"},
        mode="image_only",
    )

    assert result.output_path.exists()
    assert sampler.requests[0].positive == [["positive", {"mode": "image_only"}]]
    assert sampler.requests[0].negative == [["negative", {"mode": "image_only"}]]
    assert sampler.requests[0].seed == 77
    assert engine.attached == []
    assert engine.calls[0][0:4] == (12, "fusion prompt", "", "image_only")


def test_image_state_renderer_uses_conditioning_metadata_record_and_text_state(tmp_path: Path) -> None:
    record_path = tmp_path / "record.json"
    text_state_path = tmp_path / "text_state.pt"
    record_path.write_text(json.dumps({"idx": 3, "image_embed_pre": "image.pt", "text": "record text"}), encoding="utf-8")
    torch.save(torch.ones(2, 4), text_state_path)
    engine = FakeEngine()
    sampler = FakeSamplerRuntime()
    renderer = ImageStateAnimaRendererAdapter(output_root=tmp_path, engine=engine, sampler_runtime=sampler)

    renderer.generate(
        GenerationPlan(prompt="blend prompt", width=512, height=512, steps=8, cfg=3.0, seed=1),
        ConditioningBundle(
            source="image_state_conditioning",
            metadata={
                "image_state_record": str(record_path),
                "fusion_mode": "conditioning_fusion",
                "text_state": str(text_state_path),
            },
        ),
    )

    assert engine.calls[0][3] == "conditioning_fusion"
    assert torch.equal(engine.calls[0][4], torch.ones(2, 4))
