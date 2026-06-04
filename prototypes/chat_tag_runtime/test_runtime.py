from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from generator import build_generation_job, build_generation_job_with_planner_tags, compile_image_prompt, compile_tipo_planner_prompt, create_builtin_generation_job
from model_prototype import default_model_set, model_health
from planner import TipoPlannerConfig, merge_user_prompt_and_planner_tags, parse_planner_tags, run_tipo_planner
from runtime import ChatConfig, TagConfig, chat, create_image_job, route_request, tag_image


def test_route_sends_image_payload_to_tag(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    image.write_bytes(b"x")

    result = route_request({"task": "tag", "image_path": str(image), "message": "tag this"})

    assert result["mode"] == "tag"
    assert result["status"] in {"completed", "failed"}


def test_create_image_job_uses_builtin_generator_contract() -> None:
    result = create_image_job(
        message="red dress character",
        style="watercolor portrait",
    )

    assert result["mode"] == "image"
    assert result["status"] == "generator_required"
    assert result["generator"] == "anima-gemmanima-image-generator"
    assert result["external_backend"] is False
    assert result["prompt"] == "red dress character, watercolor portrait, high detail, coherent composition"
    assert result["planner_prompt"] == "rating: safe, red dress character, detailed, anime style, Partial tags:"
    assert result["planner_contract"] == "TIPO Partial tags continuation"
    assert result["source_message"] == "red dress character"


def test_builtin_generation_job_has_stable_output_slot() -> None:
    result = create_builtin_generation_job(message="1girl, blue eyes", style="anime key visual")

    assert result["job_id"]
    assert result["output_path"].endswith(f"{result['job_id']}.png")
    assert "runs/prototypes/chat_tag_runtime/images" in result["output_path"]


def test_builtin_generator_job_contract_rejects_external_default() -> None:
    job = build_generation_job(message="1girl, blue eyes")

    assert job.output_path.as_posix().endswith(".png")
    assert job.prompt
    assert job.planner_prompt.endswith("Partial tags:")


def test_compile_image_prompt_preserves_tag_lists() -> None:
    prompt = compile_image_prompt(message="1girl, blue eyes", style="anime key visual")

    assert prompt == "1girl, blue eyes, anime key visual"


def test_compile_tipo_planner_prompt_uses_partial_tags_contract() -> None:
    prompt = compile_tipo_planner_prompt(message="1girl, blue eyes")

    assert prompt == "rating: safe, 1girl, blue eyes, Partial tags:"


def test_parse_planner_tags_only_parses_and_dedupes_without_bias_filtering() -> None:
    tags = parse_planner_tags("1girl, blue eyes, black hair, blue eyes, smile [end of text]")

    assert tags == ["1girl", "blue eyes", "black hair", "smile"]


def test_merge_user_prompt_and_planner_tags_does_not_filter_model_output() -> None:
    prompt = merge_user_prompt_and_planner_tags(
        user_prompt="green-eyed mage",
        planner_tags=["blue eyes", "black hair", "robe"],
        style="anime key visual",
    )

    assert prompt == "green-eyed mage, blue eyes, black hair, robe, anime key visual"


def test_build_generation_job_with_planner_tags_uses_simple_merge() -> None:
    job = build_generation_job_with_planner_tags(
        message="green-eyed mage",
        planner_tags=["blue eyes", "black hair", "robe"],
    )

    assert "green-eyed mage" in job.prompt
    assert "blue eyes" in job.prompt
    assert "black hair" in job.prompt


def test_chat_and_tag_build_different_commands(tmp_path: Path) -> None:
    chat_cli = tmp_path / "llama-cli.exe"
    tag_cli = tmp_path / "llama-mtmd-cli.exe"
    chat_model = tmp_path / "chat.gguf"
    tag_model = tmp_path / "tag.gguf"
    mmproj = tmp_path / "tag.mmproj"
    image = tmp_path / "img.png"
    for path in (chat_cli, tag_cli, chat_model, tag_model, mmproj, image):
        path.write_bytes(b"x")

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    chat_result = chat(
        message="hello",
        config=ChatConfig(cli=chat_cli, model=chat_model),
        runner=fake_runner,
        timer=lambda: 1.0 if len(calls) == 0 else 1.5,
    )
    tag_result = tag_image(
        image_path=image,
        config=TagConfig(cli=tag_cli, model=tag_model, mmproj=mmproj),
        runner=fake_runner,
        timer=lambda: 2.0 if len(calls) == 1 else 2.5,
    )

    chat_cmd = calls[0][0]
    tag_cmd = calls[1][0]
    assert chat_result["mode"] == "chat"
    assert tag_result["mode"] == "tag"
    assert "--mmproj" not in chat_cmd
    assert "--mmproj" in tag_cmd
    assert "--single-turn" in chat_cmd
    assert "--simple-io" in chat_cmd
    assert "--no-display-prompt" in chat_cmd
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_tipo_planner_prefers_base_model_plus_lora(tmp_path: Path) -> None:
    cli = tmp_path / "llama-completion.exe"
    model = tmp_path / "base.gguf"
    lora = tmp_path / "adapter.gguf"
    default = tmp_path / "merged.gguf"
    for path in (cli, model, lora, default):
        path.write_bytes(b"x")

    calls = []

    def fake_popen(cmd, stdout=None, stderr=None, env=None):
        calls.append((cmd, env))
        if stdout:
            stdout.write("1girl, red dress")
        return SimpleNamespace(
            returncode=0,
            poll=lambda: 0,
            kill=lambda: None,
            wait=lambda timeout=None: 0,
        )

    result = run_tipo_planner(
        message="red dress",
        config=TipoPlannerConfig(cli=cli, model=model, lora=lora, merged_model_default=default, work_dir=tmp_path),
        runner=fake_popen,
        timer=lambda: 1.0,
    )

    cmd = calls[0][0]
    assert result["status"] == "completed"
    assert result["model"] == str(model)
    assert result["lora"] == str(lora)
    assert result["using_default_model"] is False
    assert "--lora" in cmd
    assert str(lora) in cmd
    assert "-no-cnv" in cmd
    assert "--no-display-prompt" in cmd


def test_tipo_planner_uses_merged_default_when_lora_missing(tmp_path: Path) -> None:
    cli = tmp_path / "llama-completion.exe"
    model = tmp_path / "base.gguf"
    default = tmp_path / "merged.gguf"
    for path in (cli, model, default):
        path.write_bytes(b"x")

    calls = []

    def fake_popen(cmd, stdout=None, stderr=None, env=None):
        calls.append((cmd, env))
        if stdout:
            stdout.write("1girl, red dress")
        return SimpleNamespace(
            returncode=0,
            poll=lambda: 0,
            kill=lambda: None,
            wait=lambda timeout=None: 0,
        )

    result = run_tipo_planner(
        message="red dress",
        config=TipoPlannerConfig(
            cli=cli,
            model=model,
            lora=tmp_path / "missing_adapter.gguf",
            merged_model_default=default,
            work_dir=tmp_path,
        ),
        runner=fake_popen,
        timer=lambda: 1.0,
    )

    cmd = calls[0][0]
    assert result["status"] == "completed"
    assert result["model"] == str(default)
    assert result["lora"] == ""
    assert result["using_default_model"] is True
    assert "--lora" not in cmd


def test_model_prototype_documents_chat_and_tag_assets() -> None:
    model_set = default_model_set()
    payload = model_set.to_json_dict()

    assert payload["name"] == "gemma4-local-chat-builtin-image-poc"
    assert payload["architecture"] == "gemma4-hidden-state-to-anima-synthesis"
    assert payload["single_gemma4_core_required"] is True
    assert payload["preserve_gemma4_chat"] is True
    assert payload["preserve_gemma4_multimodal_vision"] is True
    assert payload["separate_image_understanding_and_tagging"] is True
    assert payload["hidden_state_bridge_required"] is True
    assert payload["direct_anima_synthesis_required"] is True
    assert payload["role_split_ggufs_are_temporary"] is True
    assert payload["runtime_branch"] == "quantized-llamacpp"
    assert payload["future_runtime_branch"] == "unquantized-transformers-or-native"
    assert payload["quantized_runtime_required_now"] is True
    assert payload["core"]["architecture"] == "gemma4-hidden-state-to-anima-synthesis"
    assert payload["attached_modules"]["image_planner"]["task"] == "image_planner"
    assert payload["attached_modules"]["image_understander"]["task"] == "image_understanding"
    assert payload["attached_modules"]["image_understander"]["prompt_contract"].startswith("natural-language visual understanding")
    assert payload["attached_modules"]["vision_tagger"]["task"] == "tag"
    assert payload["attached_modules"]["vision_tagger"]["prompt_contract"] == "comma-separated English Danbooru tags only"
    assert payload["attached_modules"]["image_generator"]["runtime"] == "gemma4-hidden-state-anima-synthesizer"
    assert payload["attached_modules"]["image_generator"]["generator"] == "anima-gemmanima-image-generator"
    assert payload["attached_modules"]["image_generator"]["lineage"] == "Anima/GEMMANIMA"
    assert payload["attached_modules"]["image_generator"]["conditioning"] == "Gemma4 hidden states, not final prompt text"
    assert payload["attached_modules"]["image_generator"]["external_backend"] is False
    assert payload["chat"]["runtime"] == "llama-cli"
    assert payload["planner"]["task"] == "image_planner"
    assert payload["planner"]["runtime"] == "llama-completion"
    assert payload["planner"]["runtime_branch"] == "quantized-llamacpp"
    assert payload["planner"]["model"].endswith("gemma-4-E2B-it-heretic-ara-Q4_K_M.gguf")
    assert payload["planner"]["lora"].endswith("adapter_model.f16.gguf")
    assert payload["planner"]["lora_source"].endswith("adapter_model.safetensors")
    assert payload["planner"]["merged_model_default"].endswith("gemma4-tipo-ko-v2-Q4_K_M.gguf")
    assert payload["vision_understander"]["runtime"] == "llama-mtmd-cli"
    assert payload["vision_understander"]["mmproj"]
    assert payload["tag"]["runtime"] == "llama-mtmd-cli"
    assert payload["tag"]["mmproj"]
    assert payload["chat"]["task"] == "chat"
    assert payload["tag"]["task"] == "tag"


def test_model_health_checks_all_model_assets(tmp_path: Path) -> None:
    chat_cli = tmp_path / "llama-cli.exe"
    chat_model = tmp_path / "chat.gguf"
    planner_model = tmp_path / "planner.gguf"
    planner_lora = tmp_path / "planner_lora.gguf"
    planner_lora_source = tmp_path / "planner_lora.safetensors"
    planner_default = tmp_path / "planner_merged.gguf"
    planner_cli = tmp_path / "llama-completion.exe"
    vision_cli = tmp_path / "llama-mtmd-vision.exe"
    vision_model = tmp_path / "vision_understand.gguf"
    vision_mmproj = tmp_path / "vision_understand.mmproj"
    tag_cli = tmp_path / "llama-mtmd-cli.exe"
    tag_model = tmp_path / "tag.gguf"
    mmproj = tmp_path / "tag.mmproj"
    for path in (chat_cli, chat_model, planner_cli, planner_model, planner_lora, planner_lora_source, planner_default, vision_cli, vision_model, vision_mmproj, tag_cli, tag_model, mmproj):
        path.write_bytes(b"x")

    health = model_health(default_model_set(
        chat_cli=chat_cli,
        chat_model=chat_model,
        planner_cli=planner_cli,
        planner_model=planner_model,
        planner_lora=planner_lora,
        planner_lora_source=planner_lora_source,
        planner_merged_model_default=planner_default,
        vision_understand_cli=vision_cli,
        vision_understand_model=vision_model,
        vision_understand_mmproj=vision_mmproj,
        tag_cli=tag_cli,
        tag_model=tag_model,
        tag_mmproj=mmproj,
    ))

    assert health["ready"] is True
    assert health["assets"]["core.model"]["exists"] is True
    assert health["assets"]["chat.model"]["exists"] is True
    assert health["assets"]["planner.cli"]["exists"] is True
    assert health["assets"]["planner.model"]["exists"] is True
    assert health["assets"]["planner.lora"]["exists"] is True
    assert health["assets"]["planner.lora_source"]["exists"] is True
    assert health["assets"]["planner.merged_model_default"]["exists"] is True
    assert health["assets"]["vision_understander.cli"]["exists"] is True
    assert health["assets"]["vision_understander.model"]["exists"] is True
    assert health["assets"]["vision_understander.mmproj"]["exists"] is True
    assert health["assets"]["tag.mmproj"]["exists"] is True
