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
    assert payload["clarification_required"] is False
    assert payload["conflict"] is None


def test_cli_tag_image_uses_tipo_runtime(tmp_path, capsys, monkeypatch) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"x")

    def fake_tag_image(**kwargs):
        assert kwargs["image_path"] == image
        assert kwargs["prompt"] == "tag it"
        return {
            "status": "completed",
            "tags": "1girl, solo",
            "raw": "1girl, solo",
            "stderr_tail": "",
            "seconds": 0.1,
            "model": "vision.gguf",
            "mmproj": "vision.mmproj",
            "device": "CUDA0",
            "command": ["llama-mtmd-cli.exe"],
        }

    monkeypatch.setattr("gemmanima.cli.run_tipo_vision_tag", fake_tag_image)

    code = main(["tag-image", str(image), "--prompt", "tag it", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "tag_image"
    assert payload["status"] == "completed"
    assert payload["tags"] == "1girl, solo"


def test_cli_model_download_plan_reports_original_sources(capsys) -> None:
    code = main(["model-download-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    sources = {asset["name"]: asset["source"] for asset in payload["assets"]}
    assert sources["gemma_core.shared_base_gguf"]["origin"] == "original_model_page"
    assert sources["gemma_core.shared_base_gguf"]["repo_id"] == "mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF"
    assert sources["hiddenstage_bridge.bridge_checkpoint"]["origin"] == "gemmanima_adapter_bundle"
