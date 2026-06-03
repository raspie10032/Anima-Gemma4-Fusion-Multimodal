import json

from gemmanima.cli import main


def test_real_render_health_reports_dependency_readiness(capsys) -> None:
    assert main(["real-render-health", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert isinstance(payload["ready"], bool)
    assert "embedded_python" in payload["checks"]
    assert "hiddenstage_bridge" in payload["checks"]
    assert "anima_text_encoder" not in payload["checks"]
