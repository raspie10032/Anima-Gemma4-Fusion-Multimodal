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


def test_cli_tag_image_cleans_tag_output(tmp_path, capsys, monkeypatch) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"x")
    raw_tags = (
        "1girl, solo, <start_of_turn>user, You are a helpful assistant, "
        "Hello<end_of_turn>, <start_of_turn>model, Hi there<end_of_turn>, "
        + ", ".join(f"tag_{index}" for index in range(1, 40))
    )

    def fake_tag_image(**kwargs):
        return {
            "status": "completed",
            "tags": raw_tags,
            "raw": raw_tags,
            "stderr_tail": "",
        }

    monkeypatch.setattr("gemmanima.cli.run_tipo_vision_tag", fake_tag_image)

    code = main(["tag-image", str(image), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    tags = [tag.strip() for tag in payload["tags"].split(",")]
    assert len(tags) == 24
    assert tags[:4] == ["1girl", "solo", "tag_1", "tag_2"]
    assert not any("<start_of_turn>" in tag or "<end_of_turn>" in tag for tag in tags)


def test_cli_model_download_plan_reports_original_sources(capsys) -> None:
    code = main(["model-download-plan", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    sources = {asset["name"]: asset["source"] for asset in payload["assets"]}
    assert sources["gemma_core.shared_base_gguf"]["origin"] == "original_model_page"
    assert sources["gemma_core.shared_base_gguf"]["repo_id"] == "mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF"
    assert sources["hiddenstage_bridge.bridge_checkpoint"]["origin"] == "gemmanima_adapter_bundle"


def test_cli_dependency_audit_reports_no_auto_install_policy(capsys) -> None:
    code = main(["dependency-audit", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["auto_install_policy"] == "disabled"
    assert payload["network_policy"] == "model assets only; no Python package installation at app launch"
    dependency_names = {item["name"] for item in payload["dependencies"]}
    assert "Embedded Headroom-style context compressor" in dependency_names
    assert "llama-cpp-python" in dependency_names
