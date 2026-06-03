from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.config import EngineConfig, ModelConfig
from gemmanima.modules.hiddenstage_exit import HiddenStageExit


def _write_bridge_checkpoint(path):
    import torch

    torch.save({"val_mse": 0.001, "kv": {f"k{i}": i for i in range(12)}}, path)


def test_hiddenstage_exit_audits_trained_bridge_checkpoint(tmp_path) -> None:
    bridge = tmp_path / "kv_proj_hiddenstage_planner_v2.pt"
    _write_bridge_checkpoint(bridge)

    audit = HiddenStageExit(config=EngineConfig(models=ModelConfig(hiddenstage_bridge=bridge))).audit_bridge()

    assert audit.exists is True
    assert audit.readable is True
    assert audit.passed_mse_gate is True
    assert audit.kv_key_count == 12


def test_conductor_manifest_records_trained_bridge(tmp_path) -> None:
    bridge = tmp_path / "kv_proj_hiddenstage_planner_v2.pt"
    _write_bridge_checkpoint(bridge)
    config = EngineConfig(models=ModelConfig(hiddenstage_bridge=bridge))
    conductor = GemmAnimaConductor(
        session_id="bridge-integration-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
        config=config,
    )

    response = conductor.handle_user_message("draw a luminous forest shrine")

    assert response.manifest_path and response.manifest_path.exists()
    text = response.manifest_path.read_text(encoding="utf-8")
    assert "hiddenstage_bridge" in text
    assert "trained_hiddenstage_bridge" in text
