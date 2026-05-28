from gemmanima.cli import main


def test_bridge_smoke_command(capsys) -> None:
    assert main(["bridge-smoke-command", "--json"]) == 0
    out = capsys.readouterr().out
    assert "smoke_hiddenstage_bridge_forward.py" in out
    assert "kv_proj_hiddenstage_planner_v2.pt" in out
