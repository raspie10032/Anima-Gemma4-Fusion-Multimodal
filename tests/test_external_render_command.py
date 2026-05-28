import json

from gemmanima.cli import main


def test_external_render_command_marks_legacy_external_script(capsys) -> None:
    assert main(["external-render-command", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["backend"] == "external_script"
    assert "18_hiddenstage_chat_generate.py" in payload["command"]
    assert "kv_proj_hiddenstage_planner_v2.pt" in payload["command"]


def test_real_render_command_remains_compat_alias_for_external_script(capsys) -> None:
    assert main(["real-render-command", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["backend"] == "external_script"
    assert payload["deprecated_alias"] == "real-render-command"
