from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import time

from gemmanima.modules.tipo_runtime import (
    DEFAULT_TIPO_BASE_MODEL,
    DEFAULT_TIPO_TEXT_LORA,
    DEFAULT_TIPO_VISION_LORA,
    TipoTextConfig,
    TipoTextRuntime,
    TipoVisionConfig,
    build_chat_contract_harness,
    build_language_harness,
    clean_vision_tags,
    compress_chat_context_with_headroom,
    _compress_messages_with_embedded_headroom,
    _chat_prompt,
    _parse_text,
    parse_image_generation_contract,
    run_tipo_text_chat,
    run_tipo_vision_tag,
    tipo_text_health,
    tipo_vision_health,
)


def test_parse_text_strips_chat_contract_metadata_lines() -> None:
    raw = "반갑습니다. mode: chat\nOUTPUT_CONTRACT: natural_language_answer\n"

    assert _parse_text(raw) == "반갑습니다."


def test_tipo_vision_tag_builds_4070_pinned_mtmd_command(tmp_path) -> None:
    cli = tmp_path / "llama-mtmd-cli.exe"
    model = tmp_path / "vision.gguf"
    mmproj = tmp_path / "vision.mmproj"
    lora = tmp_path / "vision-lora.safetensors"
    image = tmp_path / "input.png"
    for path in (cli, model, mmproj, lora, image):
        path.write_bytes(b"x")

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="\n1girl, solo, cat ears\n\n", stderr="warn")

    original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    result = run_tipo_vision_tag(
        image_path=image,
        prompt="tag this image",
        config=TipoVisionConfig(
            cli=cli,
            model=model,
            mmproj=mmproj,
            lora_paths=(lora,),
            visible_devices="0",
            device="CUDA0",
            max_new_tokens=64,
            temperature=0.3,
        ),
        runner=fake_runner,
        timer=lambda: 10.0 if not calls else 10.5,
    )

    cmd, kwargs = calls[0]
    assert cmd[:3] == [str(cli), "-m", str(model)]
    assert str(mmproj) in cmd
    assert "--lora" in cmd
    assert cmd[cmd.index("--lora") + 1] == str(lora)
    assert "--image" in cmd
    assert str(image) in cmd
    assert "-dev" in cmd
    assert cmd[cmd.index("-dev") + 1] == "CUDA0"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_visible_devices
    assert result["tags"] == "1girl, solo, cat ears"
    assert result["seconds"] == 0.5


def test_clean_vision_tags_limits_count_and_removes_chat_template_leaks() -> None:
    raw = (
        "1girl, solo, <start_of_turn>user, You are a helpful assistant, "
        "Hello<end_of_turn>, <start_of_turn>model, Hi there<end_of_turn>, "
        "https://github.com/ggml-org/llama.cpp/discussions/13759, "
        + ", ".join(f"tag_{index}" for index in range(1, 40))
    )

    cleaned = clean_vision_tags(raw)
    tags = [tag.strip() for tag in cleaned.split(",")]

    assert len(tags) == 24
    assert tags[:4] == ["1girl", "solo", "tag_1", "tag_2"]
    assert not any("<start_of_turn>" in tag or "<end_of_turn>" in tag for tag in tags)
    assert not any("github.com" in tag for tag in tags)
    assert "You are a helpful assistant" not in tags
    assert "Hi there" not in tags


def test_tipo_vision_defaults_do_not_pin_a_machine_specific_gpu() -> None:
    cfg = TipoVisionConfig()

    assert cfg.visible_devices == ""
    assert cfg.device == ""


def test_tipo_vision_tag_omits_gpu_flags_when_unconfigured(tmp_path, monkeypatch) -> None:
    cli = tmp_path / "llama-mtmd-cli.exe"
    model = tmp_path / "vision.gguf"
    mmproj = tmp_path / "vision.mmproj"
    image = tmp_path / "input.png"
    for path in (cli, model, mmproj, image):
        path.write_bytes(b"x")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="1girl", stderr="")

    run_tipo_vision_tag(
        image_path=image,
        config=TipoVisionConfig(
            cli=cli,
            model=model,
            mmproj=mmproj,
            lora_paths=(),
            visible_devices="",
            device="",
        ),
        runner=fake_runner,
    )

    cmd, kwargs = calls[0]
    assert "-dev" not in cmd
    assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]


def test_tipo_vision_tag_reports_missing_assets(tmp_path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"x")

    result = run_tipo_vision_tag(
        image_path=image,
        config=TipoVisionConfig(
            cli=tmp_path / "missing-cli.exe",
            model=tmp_path / "missing.gguf",
            mmproj=tmp_path / "missing.mmproj",
        ),
    )

    assert result["status"] == "failed"
    assert "missing cli" in result["error"]
    assert result["error_code"] == "preflight_failed"
    assert result["preflight"]["ready"] is False
    assert result["preflight"]["blocking"] is True
    assert result["preflight"]["issues"][0]["message_ko"]


def test_tipo_text_health_reports_structured_preflight(tmp_path) -> None:
    lora = tmp_path / "missing-lora.gguf"
    health = tipo_text_health(
        TipoTextConfig(
            backend="cli",
            cli=tmp_path / "missing-cli.exe",
            model=tmp_path / "missing-model.gguf",
            lora_paths=(lora,),
        )
    )

    assert health["ready"] is False
    assert len(health["issues"]) == 3
    assert health["issues"][0]["code"].startswith("missing_tipo_text_")
    assert any(issue["asset"] == "lora_1" for issue in health["issues"])


def test_tipo_text_chat_builds_base_model_plus_lora_command(tmp_path) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    lora = tmp_path / "chat-lora.gguf"
    for path in (cli, model, lora):
        path.write_bytes(b"x")

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="\n안녕, 무엇을 도와줄까?\n", stderr="")

    result = run_tipo_text_chat(
        message="안녕",
        history=[{"role": "user", "content": "이전 질문"}],
        config=TipoTextConfig(
            cli=cli,
            model=model,
            lora_paths=(lora,),
            visible_devices="0",
            device="CUDA0",
            max_new_tokens=96,
        ),
        runner=fake_runner,
        timer=lambda: 3.0 if not calls else 4.25,
    )

    cmd, kwargs = calls[0]
    assert cmd[:3] == [str(cli), "-m", str(model)]
    assert "--mmproj" not in cmd
    assert "--lora" in cmd
    assert cmd[cmd.index("--lora") + 1] == str(lora)
    assert "-dev" in cmd
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "이전 질문" in cmd[-1]
    assert "안녕" in cmd[-1]
    assert result["status"] == "completed"
    assert result["message"] == "안녕, 무엇을 도와줄까?"
    assert result["seconds"] == 1.25


def test_tipo_text_defaults_do_not_pin_a_machine_specific_gpu() -> None:
    cfg = TipoTextConfig()

    assert cfg.visible_devices == ""
    assert cfg.device == ""


def test_tipo_text_config_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_DEVICE", "CUDA2")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_N_CTX", "2048")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_HEADROOM_ENABLED", "1")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_HEADROOM_TARGET_RATIO", "0.35")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_HEADROOM_MIN_CONTENT_LENGTH", "128")
    monkeypatch.setenv("GEMMANIMA_TIPO_TEXT_HEADROOM_PROTECT_RECENT", "3")

    cfg = TipoTextConfig()

    assert cfg.visible_devices == "2"
    assert cfg.device == "CUDA2"
    assert cfg.timeout_seconds == 17
    assert cfg.n_ctx == 2048
    assert cfg.headroom_enabled is True
    assert cfg.headroom_target_ratio == 0.35
    assert cfg.headroom_min_content_length == 128
    assert cfg.headroom_protect_recent == 3


def test_headroom_context_compression_is_disabled_by_default() -> None:
    def forbidden_loader():
        raise AssertionError("Headroom must not be imported when disabled")

    result = compress_chat_context_with_headroom(
        message="current question",
        history=[{"role": "user", "content": "older turn"}],
        enabled=False,
        compressor_loader=forbidden_loader,
    )

    assert result.message == "current question"
    assert result.history == [{"role": "user", "content": "older turn"}]
    assert result.status == "disabled"
    assert result.used is False
    assert result.warning == ""


def test_headroom_context_compression_uses_local_compressor_when_enabled() -> None:
    calls = []

    def fake_compress(messages, **kwargs):
        calls.append((messages, kwargs))
        return [
            {"role": "user", "content": "compressed history"},
            {"role": "user", "content": "compressed current question"},
        ]

    result = compress_chat_context_with_headroom(
        message="current question",
        history=[{"role": "assistant", "content": "older answer"}],
        enabled=True,
        model="gemma-local",
        compressor_loader=lambda: fake_compress,
    )

    assert calls
    assert calls[0][1]["model"] == "gemma-local"
    assert result.status == "compressed"
    assert result.used is True
    assert result.history == [{"role": "user", "content": "compressed history"}]
    assert result.message == "compressed current question"
    assert result.original_turns == 2
    assert result.compressed_turns == 2


def test_headroom_context_compression_accepts_headroom_result_object() -> None:
    class FakeHeadroomResult:
        messages = [
            {"role": "user", "content": "compressed history"},
            {"role": "user", "content": "compressed current question"},
        ]
        tokens_before = 1000
        tokens_after = 400
        tokens_saved = 600
        compression_ratio = 0.6
        transforms_applied = ["test-transform"]

    def fake_compress(messages, **kwargs):
        return FakeHeadroomResult()

    result = compress_chat_context_with_headroom(
        message="current question",
        history=[{"role": "assistant", "content": "older answer"}],
        enabled=True,
        compressor_loader=lambda: fake_compress,
    )

    assert result.status == "compressed"
    assert result.used is True
    assert result.history == [{"role": "user", "content": "compressed history"}]
    assert result.message == "compressed current question"


def test_headroom_embedded_compressor_compresses_old_long_turns() -> None:
    result = _compress_messages_with_embedded_headroom(
        [
            {"role": "user", "content": "very long old turn " * 20},
            {"role": "assistant", "content": "recent answer " * 20},
            {"role": "user", "content": "current question"},
        ],
        target_ratio=0.4,
        min_content_length=50,
        protect_recent=2,
    )

    assert result.messages[0]["content"].startswith("very long old turn")
    assert "compressed by GemmAnima embedded Headroom-style" in result.messages[0]["content"]
    assert result.messages[1]["content"].startswith("recent answer")
    assert result.messages[2]["content"] == "current question"
    assert result.tokens_saved > 0
    assert result.compression_ratio > 0.0
    assert result.transforms_applied == ["headroom:embedded"]


def test_headroom_default_loader_is_embedded_and_has_no_package_dependency() -> None:
    result = compress_chat_context_with_headroom(
        message="current question",
        history=[{"role": "user", "content": "very long old turn " * 80}],
        enabled=True,
        min_content_length=50,
        protect_recent=0,
    )

    assert result.status == "compressed"
    assert result.used is True
    assert result.tokens_saved > 0
    assert result.transforms_applied == ["headroom:embedded"]


def test_headroom_context_compression_falls_back_when_dependency_missing() -> None:
    def missing_loader():
        raise ImportError("No module named 'headroom'")

    result = compress_chat_context_with_headroom(
        message="current question",
        history=[{"role": "user", "content": "older turn"}],
        enabled=True,
        compressor_loader=missing_loader,
    )

    assert result.message == "current question"
    assert result.history == [{"role": "user", "content": "older turn"}]
    assert result.status == "unavailable"
    assert result.used is False
    assert "compressor is unavailable" in result.warning


def test_headroom_context_compression_times_out_without_blocking_chat() -> None:
    def slow_compress(messages, **kwargs):
        time.sleep(0.2)
        return [{"role": "user", "content": "too late"}]

    started = time.perf_counter()
    result = compress_chat_context_with_headroom(
        message="current question",
        history=[{"role": "user", "content": "older turn"}],
        enabled=True,
        compressor_loader=lambda: slow_compress,
        timeout_seconds=0.01,
    )

    assert time.perf_counter() - started < 0.15
    assert result.message == "current question"
    assert result.history == [{"role": "user", "content": "older turn"}]
    assert result.status == "timeout"
    assert result.used is False
    assert "timed out" in result.warning


def test_chat_prompt_packs_history_within_context_budget() -> None:
    prompt = _chat_prompt(
        "current question",
        [
            {"role": "user", "content": "old oversized context " * 1000},
            {"role": "assistant", "content": "recent answer"},
        ],
        language="en",
        chat_mode="general_chat",
        config=TipoTextConfig(n_ctx=768, max_new_tokens=64),
    )

    assert "current question" in prompt.text
    assert "recent answer" in prompt.text
    assert "old oversized context" not in prompt.text


def test_tipo_text_runtime_initializes_llama_cpp_once(tmp_path) -> None:
    model = tmp_path / "chat.gguf"
    lora = tmp_path / "adapter.gguf"
    model.write_bytes(b"x")
    lora.write_bytes(b"x")
    loads = []

    class FakeLlama:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "안녕하세요."}]}

    runtime = TipoTextRuntime(config=TipoTextConfig(model=model, lora_paths=(lora,), n_gpu_layers=-1, n_ctx=1024))
    runtime._llama_factory = FakeLlama

    first = runtime.initialize()
    second = runtime.initialize()

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert len(loads) == 1
    assert loads[0]["model_path"] == str(model)
    assert "lora_path" not in loads[0]
    assert loads[0]["n_gpu_layers"] == -1
    assert loads[0]["n_ctx"] == 1024
    assert runtime.status()["base_model_resident"] is True
    assert runtime.status()["initialized"] is True


def test_tipo_text_runtime_attaches_text_lora_for_chat_without_reloading_base(tmp_path) -> None:
    model = tmp_path / "chat.gguf"
    lora = tmp_path / "adapter.gguf"
    model.write_bytes(b"x")
    lora.write_bytes(b"x")
    loads = []
    applied_loras = []

    class FakeLlama:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "안녕하세요."}]}

    class FakeAdapterBackend:
        def __init__(self):
            self.active = []

        def apply(self, llama_model, lora_paths):
            applied_loras.append(tuple(lora_paths))
            self.active = [str(path) for path in lora_paths]
            return {"active_lora_paths": self.active}

        def status(self):
            return {"active_lora_paths": self.active, "cached_lora_paths": []}

    runtime = TipoTextRuntime(
        config=TipoTextConfig(model=model, lora_paths=(lora,), n_gpu_layers=-1),
        llama_factory=FakeLlama,
        adapter_backend=FakeAdapterBackend(),
    )

    result = runtime.chat(message="안녕")

    assert result["status"] == "completed"
    assert len(loads) == 1
    assert "lora_path" not in loads[0]
    assert applied_loras == [(lora,)]
    assert result["active_lora_paths"] == [str(lora)]
    assert runtime.status()["adapter_backend"]["active_lora_paths"] == [str(lora)]


def test_tipo_text_runtime_fails_when_gpu_offload_is_required_but_unavailable(tmp_path, monkeypatch) -> None:
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x")

    monkeypatch.setattr("gemmanima.modules.tipo_runtime._llama_supports_gpu_offload", lambda: False)

    def forbidden_factory(**kwargs):
        raise AssertionError("runtime must fail before loading a GPU-required model")

    monkeypatch.setattr("gemmanima.modules.tipo_runtime._load_llama_factory", lambda: forbidden_factory)
    runtime = TipoTextRuntime(
        config=TipoTextConfig(model=model, lora_paths=(), n_gpu_layers=-1),
    )

    status = runtime.initialize()

    assert status["status"] == "failed"
    assert status["error_code"] == "gpu_offload_unavailable"
    assert runtime.status()["initialized"] is False


def test_run_tipo_text_chat_uses_initialized_runtime_without_cli_runner(tmp_path) -> None:
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "status": "completed",
                "message": "이미 준비된 모델 응답입니다.",
                "raw": "이미 준비된 모델 응답입니다.",
                "stderr_tail": "",
                "seconds": 0.01,
                "command": [],
                "model": "resident.gguf",
                "device": "",
                "language": kwargs["language"],
                "chat_mode": kwargs["chat_mode"],
                "output_contract": "natural_language_answer",
                "runtime": "in-process",
            }

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("CLI runner must not be called when resident runtime is provided")

    runtime = FakeRuntime()
    result = run_tipo_text_chat(
        message="안녕",
        language="ko",
        chat_mode="general_chat",
        config=TipoTextConfig(headroom_enabled=True, headroom_model="gemma-local"),
        runtime=runtime,
        runner=forbidden_runner,
    )

    assert result["status"] == "completed"
    assert runtime.calls[0]["headroom_enabled"] is True
    assert runtime.calls[0]["headroom_model"] == "gemma-local"
    assert result["message"] == "이미 준비된 모델 응답입니다."
    assert runtime.calls[0]["message"] == "안녕"


def test_tipo_runtime_vision_tag_attaches_vision_lora_and_mmproj_without_reloading_base(tmp_path) -> None:
    model = tmp_path / "chat.gguf"
    text_lora = tmp_path / "text.gguf"
    vision_lora = tmp_path / "vision.gguf"
    mmproj = tmp_path / "vision.mmproj"
    image = tmp_path / "input.png"
    for path in (model, text_lora, vision_lora, mmproj, image):
        path.write_bytes(b"x")
    loads = []
    applied_loras = []
    handlers = []

    class FakeLlama:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "?덈뀞?섏꽭??"}]}

        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "1girl, solo, standing"}}]}

    class FakeAdapterBackend:
        def __init__(self):
            self.active = []

        def apply(self, llama_model, lora_paths):
            applied_loras.append(tuple(lora_paths))
            self.active = [str(path) for path in lora_paths]
            return {"active_lora_paths": self.active}

        def status(self):
            return {"active_lora_paths": self.active, "cached_lora_paths": self.active}

    def fake_handler_factory(path):
        handlers.append(path)
        return object()

    runtime = TipoTextRuntime(
        config=TipoTextConfig(model=model, lora_paths=(text_lora,), n_gpu_layers=-1),
        llama_factory=FakeLlama,
        adapter_backend=FakeAdapterBackend(),
        vision_handler_factory=fake_handler_factory,
    )

    result = runtime.vision_tag(
        image_path=image,
        prompt="tag this",
        vision_lora_paths=(vision_lora,),
        mmproj=mmproj,
    )

    assert result["status"] == "completed"
    assert len(loads) == 1
    assert "lora_path" not in loads[0]
    assert applied_loras == [(vision_lora,)]
    assert handlers == [mmproj]
    assert result["tags"] == "1girl, solo, standing"
    assert result["runtime"] == "in-process"


def test_run_tipo_vision_tag_uses_resident_runtime_without_cli_runner(tmp_path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"x")

    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def vision_tag(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "status": "completed",
                "tags": "1girl, solo",
                "raw": "1girl, solo",
                "seconds": 0.01,
                "model": "resident.gguf",
                "mmproj": str(kwargs["mmproj"]),
                "runtime": "in-process",
            }

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("vision CLI must not be called when resident runtime is available")

    runtime = FakeRuntime()
    result = run_tipo_vision_tag(
        image_path=image,
        config=TipoVisionConfig(
            model=Path("base.gguf"),
            mmproj=Path("vision.mmproj"),
            lora_paths=(Path("vision.gguf"),),
        ),
        runtime=runtime,
        runner=forbidden_runner,
    )

    assert result["status"] == "completed"
    assert result["tags"] == "1girl, solo"
    assert runtime.calls[0]["image_path"] == image


def test_tipo_text_chat_omits_gpu_flags_when_unconfigured(tmp_path, monkeypatch) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    for path in (cli, model):
        path.write_bytes(b"x")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    run_tipo_text_chat(
        message="hi",
        config=TipoTextConfig(cli=cli, model=model, lora_paths=(), visible_devices="", device=""),
        runner=fake_runner,
    )

    cmd, kwargs = calls[0]
    assert "-dev" not in cmd
    assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]


def test_tipo_text_cli_timeout_returns_structured_failure(tmp_path) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    for path in (cli, model):
        path.write_bytes(b"x")

    def fake_runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    result = run_tipo_text_chat(
        message="hi",
        config=TipoTextConfig(cli=cli, model=model, lora_paths=(), timeout_seconds=3),
        runner=fake_runner,
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "text_cli_timeout"
    assert "timed out" in result["error"]


def test_tipo_text_chat_passes_multiple_loras_as_comma_separated_argument(tmp_path) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    first_lora = tmp_path / "chat-a.gguf"
    second_lora = tmp_path / "chat-b.gguf"
    for path in (cli, model, first_lora, second_lora):
        path.write_bytes(b"x")

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    run_tipo_text_chat(
        message="hi",
        config=TipoTextConfig(cli=cli, model=model, lora_paths=(first_lora, second_lora)),
        runner=fake_runner,
    )

    cmd = calls[0]
    lora_indexes = [index for index, value in enumerate(cmd) if value == "--lora"]
    assert len(lora_indexes) == 1
    assert cmd[lora_indexes[0] + 1] == f"{first_lora},{second_lora}"


def test_tipo_defaults_share_base_model_and_use_lora_adapters() -> None:
    text_cfg = TipoTextConfig()
    vision_cfg = TipoVisionConfig()

    assert text_cfg.model == DEFAULT_TIPO_BASE_MODEL
    assert vision_cfg.model == DEFAULT_TIPO_BASE_MODEL
    assert text_cfg.lora_paths == (DEFAULT_TIPO_TEXT_LORA,)
    assert vision_cfg.lora_paths == (DEFAULT_TIPO_VISION_LORA,)
    assert DEFAULT_TIPO_TEXT_LORA.suffix == ".gguf"
    assert DEFAULT_TIPO_VISION_LORA.suffix == ".gguf"


def test_tipo_vision_health_reports_default_lora_asset() -> None:
    health = tipo_vision_health(
        TipoVisionConfig(
            cli=Path("missing-cli.exe"),
            model=Path("missing-base.gguf"),
            mmproj=Path("missing-mmproj.gguf"),
            lora_paths=(Path("missing-vision-lora.gguf"),),
        )
    )

    assert any(issue["asset"] == "lora_1" for issue in health["issues"])
    assert any(issue["message_ko"] == "비전 태깅 실행 파일을 찾을 수 없습니다." for issue in health["issues"])


def test_language_harness_forces_korean_user_communication() -> None:
    harness = build_language_harness("ko")

    assert "Korean" in harness
    assert "MUST communicate with the user in Korean only" in harness
    assert "Do not switch language" in harness
    assert "Danbooru tags" in harness
    assert "canonical English Danbooru tags" in harness
    assert "never translate tags into the selected interface language" in harness


def test_chat_contract_harness_forces_canonical_tag_output() -> None:
    harness = build_chat_contract_harness("tag_request")

    assert "MODE: tag_request" in harness
    assert "comma-separated canonical English Danbooru tags" in harness
    assert "Do not translate tags" in harness
    assert "No prose" in harness
    assert "No Markdown" in harness


def test_tipo_text_chat_injects_requested_language_harness(tmp_path) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    for path in (cli, model):
        path.write_bytes(b"x")

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="\n좋습니다.\n", stderr="")

    result = run_tipo_text_chat(
        message="Explain this in Korean even if prior logs are English.",
        language="ko",
        history=[{"role": "assistant", "content": "Previous English answer."}],
        config=TipoTextConfig(cli=cli, model=model, lora_paths=()),
        runner=fake_runner,
    )

    prompt = calls[0][0][-1]
    assert "MUST communicate with the user in Korean only" in prompt
    assert "Previous English answer." in prompt
    assert "Explain this in Korean" in prompt
    assert result["language"] == "ko"


def test_tipo_text_chat_injects_requested_chat_contract(tmp_path) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    for path in (cli, model):
        path.write_bytes(b"x")

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='{"intent":"generate_image","prompt":"1girl, solo","negative_prompt":"","notes":""}\n',
            stderr="",
        )

    result = run_tipo_text_chat(
        message="Make an image prompt from this idea.",
        language="en",
        chat_mode="image_generation_request",
        config=TipoTextConfig(cli=cli, model=model, lora_paths=()),
        runner=fake_runner,
    )

    prompt = calls[0][0][-1]
    assert "MODE: image_generation_request" in prompt
    assert "Return exactly one compact JSON object" in prompt
    assert '"prompt"' in prompt
    assert result["chat_mode"] == "image_generation_request"
    assert result["output_contract"] == "image_generation_json"


def test_tipo_text_chat_fails_when_image_contract_is_invalid(tmp_path) -> None:
    cli = tmp_path / "llama-cli.exe"
    model = tmp_path / "chat.gguf"
    for path in (cli, model):
        path.write_bytes(b"x")

    def fake_runner(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Sure, I can draw that.", stderr="")

    result = run_tipo_text_chat(
        message="Make an image.",
        chat_mode="image_generation_request",
        config=TipoTextConfig(cli=cli, model=model, lora_paths=()),
        runner=fake_runner,
    )

    assert result["status"] == "failed"
    assert "JSON contract" in result["error"]
    assert result["prompt"] == ""


def test_parse_image_generation_contract_accepts_fenced_noisy_json() -> None:
    parsed = parse_image_generation_contract(
        'assistant note\n```json\n{"intent":"generate_image","prompt":"1girl, solo, standing_pose",'
        '"negative_prompt":"low quality","notes":"Ready."}\n```\nextra'
    )

    assert parsed["status"] == "completed"
    assert parsed["prompt"] == "1girl, solo, standing_pose"
    assert parsed["negative_prompt"] == "low quality"
    assert parsed["notes"] == "Ready."


def test_parse_image_generation_contract_rejects_missing_prompt() -> None:
    parsed = parse_image_generation_contract('{"intent":"generate_image","negative_prompt":"bad"}')

    assert parsed["status"] == "failed"
    assert "prompt" in parsed["error"]
