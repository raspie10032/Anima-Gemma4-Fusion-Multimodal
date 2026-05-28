import json

from gemmanima.cli import main


def test_renderer_backends_cli_reports_external_and_in_process(capsys) -> None:
    assert main(["renderer-backends", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["external_script"]["execution"] == "subprocess"
    assert payload["in_process"]["execution"] == "in_process"


def test_renderer_backends_reports_in_process_dependency_status(capsys) -> None:
    assert main(["renderer-backends", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["in_process"]["dependency_ready"] is True
