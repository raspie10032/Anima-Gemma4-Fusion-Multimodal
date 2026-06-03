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
    assert isinstance(payload["dependencies"]["ready"], bool)
    assert "hiddenstage_bridge" in payload["dependencies"]["checks"]


def test_real_render_command_accepts_hiddenstage_bridge_override(tmp_path, capsys) -> None:
    bridge = tmp_path / "kv_proj_text_delta_300k_from_epoch1_a0p35.pt"
    bridge.write_bytes(b"checkpoint")

    assert main(["real-render-command", "--hiddenstage-bridge", str(bridge), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["adapter"] == str(bridge)
    assert str(bridge) in payload["command"]
    assert payload["dependencies"]["checks"]["hiddenstage_bridge"] is True
