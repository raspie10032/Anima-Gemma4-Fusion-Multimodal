import json

from gemmanima.cli import main


def test_gui_command_points_to_local_server(capsys) -> None:
    assert main(["gui-command", "--port", "9876", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["url"] == "http://127.0.0.1:9876"
    assert "gemmanima.server" in payload["command"]
    assert payload["gpu"] == "RTX 4070 Ti SUPER"
