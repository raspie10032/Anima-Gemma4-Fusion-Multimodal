from pathlib import Path

from gemmanima.training.bridge_quality_diagnosis import (
    BridgeQualityDiagnosisConfig,
    build_eval_generate_command,
    default_output_root,
)


def test_bridge_quality_diagnosis_defaults_to_4070_only() -> None:
    config = BridgeQualityDiagnosisConfig(
        prompt="draw a tiny blue forest spirit",
        output_root=Path("reports/bridge_quality_diagnosis/test"),
    )

    assert config.gpu_index == 0
    assert config.env()["CUDA_VISIBLE_DEVICES"] == "0"
    assert config.env()["PYTHONUTF8"] == "1"


def test_build_eval_generate_command_uses_same_render_settings_for_qwen_and_gemma(tmp_path: Path) -> None:
    config = BridgeQualityDiagnosisConfig(
        prompt="draw a tiny blue forest spirit",
        output_root=tmp_path,
        size=1024,
        steps=30,
        cfg=1.0,
        sampler="euler_ancestral",
        scheduler="sgm_uniform",
        seed=424242,
        unet_dtype="fp8_e4m3fn_fast",
        adapter=Path(r"E:\anima_gemma_swap\final_adapters\kv_proj_text_delta_300k_from_epoch1_a0p35.pt"),
    )
    prompt_file = tmp_path / "prompt.jsonl"

    qwen = build_eval_generate_command(config, mode="qwen", name="qwen_baseline", prompt_file=prompt_file)
    gemma = build_eval_generate_command(config, mode="gemma", name="gemma_bridge", prompt_file=prompt_file)

    for command in (qwen, gemma):
        text = " ".join(str(part) for part in command)
        assert "--size 1024" in text
        assert "--steps 30" in text
        assert "--cfg 1.0" in text
        assert "--sampler euler_ancestral" in text
        assert "--scheduler sgm_uniform" in text
        assert "--seed 424242" in text
        assert "--unet-dtype fp8_e4m3fn_fast" in text
        assert str(prompt_file) in text
    assert "--mode qwen" in " ".join(qwen)
    assert "--mode gemma" in " ".join(gemma)
    assert "--adapter" in gemma


def test_default_output_root_is_report_scoped() -> None:
    root = default_output_root()

    assert root.parts[-2:] == ("bridge_quality_diagnosis", "latest")
