from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.modules.hiddenstage_exit import HiddenStageExit


def test_hiddenstage_exit_audits_trained_bridge_checkpoint() -> None:
    audit = HiddenStageExit().audit_bridge()

    assert audit.exists is True
    assert audit.readable is True
    assert audit.passed_mse_gate is True
    assert audit.kv_key_count == 12


def test_conductor_manifest_records_trained_bridge(tmp_path) -> None:
    conductor = GemmAnimaConductor(
        session_id="bridge-integration-test",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
    )

    response = conductor.handle_user_message("draw a luminous forest shrine")

    assert response.manifest_path and response.manifest_path.exists()
    text = response.manifest_path.read_text(encoding="utf-8")
    assert "hiddenstage_bridge" in text
    assert "trained_hiddenstage_bridge" in text
