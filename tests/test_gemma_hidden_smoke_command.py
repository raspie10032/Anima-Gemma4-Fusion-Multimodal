import json

from gemmanima.cli import main


def test_gemma_hidden_smoke_command_points_at_embedded_python(capsys) -> None:
    assert main(["gemma-hidden-smoke-command", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["gpu"] == "RTX 4070 Ti SUPER"
    assert payload["command"].startswith("python ")
    assert "smoke_gemma_hidden_provider.py" in payload["command"]
