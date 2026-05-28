import json

from gemmanima.cli import main


def test_cli_json_image_request(tmp_path, capsys) -> None:
    code = main(
        [
            "\uc6d0\uc2e0\uc758 \ub098\ud788\ub2e4\ub97c \uc232 \ubc30\uacbd\uc73c\ub85c \uadf8\ub824\uc918",
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--image-root",
            str(tmp_path / "images"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "generate_image"
    assert payload["status"] == "completed"
    assert payload["manifest_path"]
    assert payload["output_path"]
