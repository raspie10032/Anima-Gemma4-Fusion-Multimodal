import torch

from gemmanima.rendering.anima_adapter import attach_hiddenstage_adapter


class DiffusionModel:
    llm_adapter = None


class InnerModel:
    def __init__(self) -> None:
        self.diffusion_model = DiffusionModel()

    def get_dtype_inference(self):
        return torch.float16


class FakeModel:
    def __init__(self) -> None:
        self.model = InnerModel()


def test_attach_hiddenstage_adapter_loads_kv_and_sets_model_adapter(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"kv": {"a": torch.ones(1)}}, checkpoint)
    loaded = {}

    def fake_build(dm_path, device, dtype):
        loaded["build"] = (dm_path, device, dtype)
        return "adapter", None

    def fake_load_kv(adapter, kv):
        loaded["load_kv"] = (adapter, kv)

    model = FakeModel()
    attach_hiddenstage_adapter(
        model,
        diffusion_model_path="anima.safetensors",
        checkpoint=checkpoint,
        build_adapter=fake_build,
        load_kv=fake_load_kv,
        adapter_dtype=torch.bfloat16,
    )

    assert loaded["build"] == ("anima.safetensors", "cuda", torch.bfloat16)
    assert loaded["load_kv"][0] == "adapter"
    assert model.model.diffusion_model.llm_adapter == "adapter"
