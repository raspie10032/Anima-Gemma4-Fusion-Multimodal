from pathlib import Path

import pytest
import torch

from gemmanima.rendering.image_state_engine import (
    ConditioningBlend,
    ImageStateConditioningConfig,
    ImageStateConditioningEngine,
    ImageStateConditioningError,
    image_state_engine_status,
)
from gemmanima.rendering.image_state_translator import (
    ImageStateToConditioningTranslator,
    load_image_state_translator,
)


class FakeT5Provider:
    def encode_ids_weights(self, prompt):
        base = 10 if "bad" not in prompt else 90
        return torch.tensor([base, base + 1], dtype=torch.int32), torch.tensor([1.0, 0.5], dtype=torch.float32)


class FakeAdapter(torch.nn.Module):
    def __init__(self, value: float, width: int = 6) -> None:
        super().__init__()
        self.value = value
        self.width = width

    def forward(self, state, ids):
        return torch.full((state.shape[0], ids.shape[1], self.width), self.value, dtype=torch.float32)


def test_translator_forward_matches_anima_conditioning_shape() -> None:
    model = ImageStateToConditioningTranslator(image_dim=4, width=8, target_dim=6, vocab_size=32, heads=2)

    result = model(
        torch.ones(2, 3, 4),
        torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        image_mask=torch.ones(2, 3, dtype=torch.bool),
        target_mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
    )

    assert result.shape == (2, 4, 6)
    assert torch.all(result[:, 3:] == 0)


def test_load_image_state_translator_reads_training_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    model = ImageStateToConditioningTranslator(image_dim=4, width=8, target_dim=6, vocab_size=32, heads=2)
    torch.save({"model": model.state_dict(), "meta": {"epoch": 1}}, checkpoint)

    loaded = load_image_state_translator(
        checkpoint,
        device="cpu",
        dtype=torch.float32,
        image_dim=4,
        width=8,
        target_dim=6,
        vocab_size=32,
        heads=2,
    )

    assert isinstance(loaded, ImageStateToConditioningTranslator)
    assert loaded.training is False


def test_engine_builds_sampler_conditioning_and_diagnostics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(checkpoint=checkpoint, device="cpu", image_dim=4),
        t5_provider=FakeT5Provider(),
    )

    payload = engine.condition_from_tensor(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.25, 0.0, -0.25]]),
        prompt="masterpiece, test image state",
        metadata={"source_idx": 7},
    )

    assert payload.positive[0][0].shape == (1, 2, 4)
    assert payload.positive[0][1]["t5xxl_ids"].tolist() == [10, 11]
    assert payload.positive[0][1]["source_idx"] == 7
    assert payload.negative[0][0].shape == (1, 1, 4)
    assert payload.negative[0][1]["t5xxl_ids"].tolist() == [90, 91]
    assert payload.diagnostics["image_state"]["shape"] == [1, 2, 4]
    assert payload.diagnostics["positive_tokens"] == 2


def test_engine_supports_hidden_fusion_before_adapter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(checkpoint=checkpoint, device="cpu", image_dim=4),
        t5_provider=FakeT5Provider(),
    )

    payload = engine.condition_from_states(
        text_state=torch.ones(3, 4),
        image_state=torch.zeros(2, 4),
        prompt="hidden fusion",
        mode="hidden_fusion",
    )

    assert payload.positive[0][0].shape == (1, 5, 4)
    assert payload.positive[0][1]["conditioning_stage"] == "pre_adapter"
    assert payload.positive[0][1]["t5xxl_ids"].tolist() == [10, 11]


def test_engine_supports_conditioning_fusion_after_translation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(
            checkpoint=checkpoint,
            device="cpu",
            dtype="float32",
            image_dim=4,
            blend=ConditioningBlend(text_weight=0.25, image_weight=0.75),
        ),
        t5_provider=FakeT5Provider(),
        text_adapter=FakeAdapter(2.0),
        translator_loader=lambda path, device, dtype: FakeAdapter(10.0),
    )

    payload = engine.condition_from_states(
        text_state=torch.ones(3, 4),
        image_state=torch.zeros(2, 4),
        prompt="conditioning fusion",
        mode="conditioning_fusion",
    )

    assert payload.positive[0][0].shape == (1, 512, 6)
    assert torch.allclose(payload.positive[0][0][:, :1], torch.full((1, 1, 6), 8.0))
    assert torch.allclose(payload.positive[0][0][:, 1:2], torch.full((1, 1, 6), 4.0))
    assert torch.all(payload.positive[0][0][:, 2:] == 0)
    assert payload.positive[0][1]["conditioning_stage"] == "post_adapter"
    assert "t5xxl_ids" not in payload.positive[0][1]
    assert payload.diagnostics["blend"] == {"text_weight": 0.25, "image_weight": 0.75}


def test_engine_rejects_unstable_or_wrong_dim_image_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(checkpoint=checkpoint, device="cpu", image_dim=4),
        t5_provider=FakeT5Provider(),
    )

    with pytest.raises(ImageStateConditioningError, match="expected image state last dim 4"):
        engine.condition_from_tensor(torch.ones(2, 5), prompt="bad dim")

    with pytest.raises(ImageStateConditioningError, match="non-finite"):
        engine.condition_from_tensor(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), prompt="nan")


def test_engine_attaches_loaded_translator_to_anima_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    attached = object()
    calls = []

    def load_translator(path, device, dtype):
        calls.append((path, device, dtype))
        return attached

    class Diffusion:
        llm_adapter = None

    class ModelRoot:
        diffusion_model = Diffusion()

    class Model:
        model = ModelRoot()

    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(checkpoint=checkpoint, device="cpu", dtype="float32"),
        t5_provider=FakeT5Provider(),
        translator_loader=load_translator,
    )

    assert engine.attach_to_model(Model(), mode="image_only") is attached
    assert Model.model.diffusion_model.llm_adapter is attached
    assert calls == [(checkpoint, "cpu", torch.float32)]


def test_engine_attaches_text_adapter_for_hidden_fusion(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    text_adapter = object()

    class Diffusion:
        llm_adapter = None

    class ModelRoot:
        diffusion_model = Diffusion()

    class Model:
        model = ModelRoot()

    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(checkpoint=checkpoint, device="cpu", dtype="float32"),
        t5_provider=FakeT5Provider(),
        text_adapter=text_adapter,
    )

    assert engine.attach_to_model(Model(), mode="hidden_fusion") is text_adapter
    assert Model.model.diffusion_model.llm_adapter is text_adapter


def test_hidden_fusion_aligns_image_tokens_to_text_dim(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    checkpoint.write_bytes(b"placeholder")
    engine = ImageStateConditioningEngine(
        ImageStateConditioningConfig(checkpoint=checkpoint, device="cpu", image_dim=4),
        t5_provider=FakeT5Provider(),
    )

    payload = engine.condition_from_states(
        text_state=torch.ones(2, 6),
        image_state=torch.ones(3, 4),
        prompt="aligned hidden fusion",
        mode="hidden_fusion",
    )

    assert payload.positive[0][0].shape == (1, 5, 6)
    assert torch.all(payload.positive[0][0][0, 2:, 4:] == 0)


def test_image_state_engine_status_reports_operational_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "image_translator.pt"
    subset = tmp_path / "subset.jsonl"
    report = tmp_path / "train_report.json"
    checkpoint.write_bytes(b"checkpoint")
    subset.write_text("{}\n", encoding="utf-8")
    report.write_text('{"history": [{"epoch": 1, "val_mse": 0.1}]}', encoding="utf-8")

    status = image_state_engine_status(checkpoint=checkpoint, subset=subset, train_report=report)

    assert status["ready"] is True
    assert status["checkpoint_exists"] is True
    assert status["subset_exists"] is True
    assert status["train_report"]["best_val_mse"] == 0.1
