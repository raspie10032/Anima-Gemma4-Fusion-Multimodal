import json

from gemmanima.cli import main


def test_real_render_health_reports_dependency_readiness(capsys) -> None:
    assert main(["real-render-health", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is True
    assert payload["checks"]["embedded_python"] is True
    assert payload["checks"]["hiddenstage_bridge"] is True
    assert "anima_text_encoder" not in payload["checks"]
