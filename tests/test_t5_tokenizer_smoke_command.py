import json

from gemmanima.cli import main


def test_t5_tokenizer_smoke_command_points_at_embedded_python(capsys) -> None:
    assert main(["t5-tokenizer-smoke-command", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["gpu"] == "RTX 4070 Ti SUPER"
    assert payload["command"].startswith("python ")
    assert "smoke_t5_tokenizer_provider.py" in payload["command"]
