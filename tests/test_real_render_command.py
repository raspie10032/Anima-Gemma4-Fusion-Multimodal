import json

from gemmanima.cli import main


def test_real_render_command_uses_trained_bridge_and_4070_ti_super(capsys) -> None:
    assert main(["real-render-command", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["gpu"] == "RTX 4070 Ti SUPER"
    assert "18_hiddenstage_chat_generate.py" in payload["command"]
    assert "kv_proj_hiddenstage_planner_v2.pt" in payload["command"]
    assert "nahida_hiddenstage_bridge_real_smoke.png" in payload["command"]
    assert payload["seed"] == 19375672098
    assert "--seed" in payload["argv"]
    assert payload["dependencies"]["ready"] is True
