import json

from gemmanima.cli import main


def test_in_process_render_smoke_command_uses_4070_ti_super(capsys) -> None:
    assert main(["in-process-render-smoke-command", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["gpu"] == "RTX 4070 Ti SUPER"
    assert payload["cuda_visible_devices"] == "0"
    assert payload["gemma_embed_on_gpu"] == "1"
    assert "smoke_in_process_render.py" in payload["command"]
    assert "$env:CUDA_VISIBLE_DEVICES='0'" in payload["command"]
    assert "--steps 8" in payload["command"]
    assert "--size 512" in payload["command"]
    assert "--json" in payload["command"]
