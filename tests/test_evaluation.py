from pathlib import Path

from gemmanima.training.evaluation import audit_bridge_checkpoint


def test_audit_bridge_checkpoint_missing(tmp_path: Path) -> None:
    result = audit_bridge_checkpoint(tmp_path / "missing.pt")

    assert result["exists"] is False
    assert result["readable"] is False
    assert result["passed_mse_gate"] is False
