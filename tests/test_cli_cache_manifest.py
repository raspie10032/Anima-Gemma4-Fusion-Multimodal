import json

from gemmanima.cli import main


def test_cli_write_cache_manifest_writes_validated_file(tmp_path, capsys) -> None:
    out = tmp_path / "CACHE_BUILD_MANIFEST.json"
    code = main(
        [
            "write-cache-manifest",
            "--stage",
            "poc1_1k_smoke",
            "--cache-kind",
            "gemma_text_state",
            "--sample-count",
            "2",
            "--source-manifest",
            str(tmp_path / "subset.jsonl"),
            "--output-dir",
            str(tmp_path / "cache"),
            "--success-count",
            "2",
            "--failure-count",
            "0",
            "--shape",
            "1,16,1536",
            "--dtype",
            "float32",
            "--device",
            "cuda:0",
            "--manifest-out",
            str(out),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_path"] == str(out)
    assert json.loads(out.read_text(encoding="utf-8"))["cache_kind"] == "gemma_text_state"


def test_prepare_commands_expose_cache_manifest_commands(tmp_path, capsys) -> None:
    assert main(["prepare-gemma-cache", "--json"]) == 0
    gemma_payload = json.loads(capsys.readouterr().out)
    assert "cache_manifest_command" in gemma_payload["plans"][0]

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": 1, "action": "generate", "teacher_prompt": "cat"}), encoding="utf-8")
    assert main(["prepare-teacher-targets", "--manifest", str(manifest), "--output-subset", str(tmp_path / "subset.jsonl"), "--json"]) == 0
    teacher_payload = json.loads(capsys.readouterr().out)
    assert "cache_manifest_command" in teacher_payload["export"]
