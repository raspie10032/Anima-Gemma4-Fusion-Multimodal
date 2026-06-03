from __future__ import annotations

import os
import json
import queue
import re
import subprocess
import ctypes
import base64
import threading
import time
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from shutil import which
from types import SimpleNamespace
from typing import Any, Callable

from gemmanima.core.model_paths import allow_legacy_model_paths, model_path


def _default_cli_path(env_name: str, executable: str, legacy_path: str | None = None) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    resolved = which(executable)
    if resolved:
        return Path(resolved)
    legacy = Path(legacy_path) if legacy_path is not None else None
    if legacy is not None and allow_legacy_model_paths() and legacy.exists():
        return legacy
    return Path(executable)


DEFAULT_TIPO_CLI = _default_cli_path(
    "GEMMANIMA_TIPO_VISION_CLI",
    "llama-mtmd-cli.exe",
)
DEFAULT_TIPO_TEXT_CLI = _default_cli_path(
    "GEMMANIMA_TIPO_TEXT_CLI",
    "llama-cli.exe",
)
DEFAULT_TIPO_BASE_MODEL = model_path(
    "gemma_core",
    "gemma-4-E2B-it-heretic-ara-custom.Q4_K_M.gguf",
)
DEFAULT_TIPO_TEXT_MODEL = DEFAULT_TIPO_BASE_MODEL
DEFAULT_TIPO_VISION_MODEL = DEFAULT_TIPO_BASE_MODEL
DEFAULT_TIPO_TEXT_LORA = model_path(
    "gemma_core",
    "text-adapter-model-f16.gguf",
)
DEFAULT_TIPO_VISION_LORA = model_path(
    "gemma_core",
    "vision-tagger-adapter-model-f16.gguf",
)
DEFAULT_TIPO_VISION_MMPROJ = model_path(
    "gemma_core",
    "gemma4-tipo-vision.mmproj-f16.gguf",
)

DEFAULT_TAG_PROMPT = (
    "Output ONLY a comma-separated list of English Danbooru tags for this image. "
    "No thinking, no explanation, no sentences. Capture the image and plausibly "
    "expand it TIPO style. About 50 tags."
)
DEFAULT_VISION_TAG_LIMIT = 24
VISION_TAG_TEMPLATE_LEAK_MARKERS = (
    "<start_of_turn>",
    "<end_of_turn>",
    "you are a helpful assistant",
    "output only",
    "comma-separated",
    "danbooru tags",
    "no thinking",
    "no explanation",
    "no sentences",
    "http://",
    "https://",
    "github.com",
    "llama.cpp",
)
VISION_TAG_TEMPLATE_LEAK_EXACT = {
    "hello",
    "hi there",
}


@dataclass(frozen=True)
class TipoVisionConfig:
    cli: Path = field(default_factory=lambda: _env_path("GEMMANIMA_TIPO_VISION_CLI", DEFAULT_TIPO_CLI))
    model: Path = field(default_factory=lambda: _env_path("GEMMANIMA_TIPO_VISION_MODEL", DEFAULT_TIPO_VISION_MODEL))
    mmproj: Path = field(default_factory=lambda: _env_path("GEMMANIMA_TIPO_VISION_MMPROJ", DEFAULT_TIPO_VISION_MMPROJ))
    lora_paths: tuple[Path, ...] = field(
        default_factory=lambda: _env_paths("GEMMANIMA_TIPO_VISION_LORAS", (DEFAULT_TIPO_VISION_LORA,))
    )
    visible_devices: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_VISION_VISIBLE_DEVICES", ""))
    device: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_VISION_DEVICE", ""))
    max_new_tokens: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_VISION_MAX_NEW_TOKENS", 140))
    temperature: float = field(default_factory=lambda: _env_float("GEMMANIMA_TIPO_VISION_TEMPERATURE", 0.7))
    timeout_seconds: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_VISION_TIMEOUT_SECONDS", 120))
    seed: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_VISION_SEED", 42))
    chat_template: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_VISION_CHAT_TEMPLATE", "gemma"))


@dataclass(frozen=True)
class TipoTextConfig:
    backend: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_TEXT_BACKEND", "in-process"))
    cli: Path = field(default_factory=lambda: _env_path("GEMMANIMA_TIPO_TEXT_CLI", DEFAULT_TIPO_TEXT_CLI))
    model: Path = field(default_factory=lambda: _env_path("GEMMANIMA_TIPO_TEXT_MODEL", DEFAULT_TIPO_TEXT_MODEL))
    lora_paths: tuple[Path, ...] = field(
        default_factory=lambda: _env_paths("GEMMANIMA_TIPO_TEXT_LORAS", (DEFAULT_TIPO_TEXT_LORA,))
    )
    visible_devices: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_TEXT_VISIBLE_DEVICES", ""))
    device: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_TEXT_DEVICE", ""))
    max_new_tokens: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_MAX_NEW_TOKENS", 256))
    temperature: float = field(default_factory=lambda: _env_float("GEMMANIMA_TIPO_TEXT_TEMPERATURE", 0.7))
    timeout_seconds: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_TIMEOUT_SECONDS", 120))
    seed: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_SEED", 42))
    chat_template: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_TEXT_CHAT_TEMPLATE", "gemma"))
    n_ctx: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_N_CTX", 4096))
    n_gpu_layers: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_N_GPU_LAYERS", -1))
    main_gpu: int = field(default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_MAIN_GPU", 0))
    flash_attn: bool = field(default_factory=lambda: _env_bool("GEMMANIMA_TIPO_TEXT_FLASH_ATTN", False))
    verbose: bool = field(default_factory=lambda: _env_bool("GEMMANIMA_TIPO_TEXT_VERBOSE", False))
    headroom_enabled: bool = field(default_factory=lambda: _env_bool("GEMMANIMA_TIPO_TEXT_HEADROOM_ENABLED", False))
    headroom_model: str = field(default_factory=lambda: os.environ.get("GEMMANIMA_TIPO_TEXT_HEADROOM_MODEL", "gemma-local"))
    headroom_timeout_seconds: float = field(
        default_factory=lambda: _env_float("GEMMANIMA_TIPO_TEXT_HEADROOM_TIMEOUT_SECONDS", 1.5)
    )
    headroom_target_ratio: float = field(
        default_factory=lambda: _env_float("GEMMANIMA_TIPO_TEXT_HEADROOM_TARGET_RATIO", 0.4)
    )
    headroom_min_content_length: int = field(
        default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_HEADROOM_MIN_CONTENT_LENGTH", 800)
    )
    headroom_protect_recent: int = field(
        default_factory=lambda: _env_int("GEMMANIMA_TIPO_TEXT_HEADROOM_PROTECT_RECENT", 2)
    )


@dataclass(frozen=True)
class HeadroomCompression:
    message: str
    history: list[dict[str, str]]
    status: str
    used: bool
    warning: str = ""
    original_turns: int = 0
    compressed_turns: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    transforms_applied: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatPrompt:
    text: str
    headroom: HeadroomCompression


Runner = Callable[..., Any]
Timer = Callable[[], float]
LlamaFactory = Callable[..., Any]
VisionHandlerFactory = Callable[[Path], Any]


class LlamaCppLoraAdapterBackend:
    def __init__(self) -> None:
        self._adapter_cache: dict[Path, Any] = {}
        self._active_lora_paths: tuple[Path, ...] = ()

    def apply(self, llama_model: Any, lora_paths: tuple[Path, ...]) -> dict[str, Any]:
        normalized = tuple(Path(path) for path in lora_paths)
        if normalized == self._active_lora_paths:
            return self.status()

        from llama_cpp import llama_cpp

        adapters = []
        for path in normalized:
            adapter = self._adapter_cache.get(path)
            if adapter is None:
                adapter = llama_cpp.llama_adapter_lora_init(
                    llama_model._model.model,
                    str(path).encode("utf-8"),
                )
                if adapter is None:
                    raise RuntimeError(f"failed to initialize LoRA adapter: {path}")
                self._adapter_cache[path] = adapter
            adapters.append(adapter)

        if adapters:
            adapter_array = (llama_cpp.llama_adapter_lora_p_ctypes * len(adapters))(*adapters)
            scales = (ctypes.c_float * len(adapters))(*([1.0] * len(adapters)))
            result = llama_cpp.llama_set_adapters_lora(
                llama_model._ctx.ctx,
                adapter_array,
                len(adapters),
                scales,
            )
        else:
            result = llama_cpp.llama_set_adapters_lora(llama_model._ctx.ctx, None, 0, None)
        if result:
            raise RuntimeError("failed to set LoRA adapters on resident Gemma context")
        self._active_lora_paths = normalized
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "active_lora_paths": [str(path) for path in self._active_lora_paths],
            "cached_lora_paths": [str(path) for path in self._adapter_cache],
        }


class TipoTextRuntime:
    def __init__(
        self,
        *,
        config: TipoTextConfig | None = None,
        llama_factory: LlamaFactory | None = None,
        adapter_backend: Any | None = None,
        vision_handler_factory: VisionHandlerFactory | None = None,
        timer: Timer = time.perf_counter,
    ) -> None:
        self.config = config or TipoTextConfig()
        self._llama_factory = llama_factory
        self._adapter_backend = adapter_backend or LlamaCppLoraAdapterBackend()
        self._vision_handler_factory = vision_handler_factory or _load_mtmd_chat_handler
        self._vision_handlers: dict[Path, Any] = {}
        self._timer = timer
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._status: dict[str, Any] = {
            "status": "not_initialized",
            "initialized": False,
            "base_model_resident": False,
            "persistent": True,
            "runtime": "in-process",
            "model": str(self.config.model),
        }

    def initialize(self) -> dict[str, Any]:
        with self._lock:
            if self._model is not None:
                return dict(self._status)
            missing = _missing_text_runtime_assets(self.config)
            if missing:
                self._status = {
                    "status": "failed",
                    "initialized": False,
                    "base_model_resident": False,
                    "persistent": True,
                    "runtime": "in-process",
                    "error_code": "preflight_failed",
                    "error": "; ".join(missing),
                    "preflight": _preflight("tipo_text", _text_runtime_assets(self.config)),
                    "model": str(self.config.model),
                    "seconds": 0.0,
                }
                return dict(self._status)
            gpu_offload_supported = None
            if self._llama_factory is None:
                gpu_offload_supported = _llama_supports_gpu_offload()
                if self.config.n_gpu_layers != 0 and not gpu_offload_supported:
                    self._status = {
                        "status": "failed",
                        "initialized": False,
                        "base_model_resident": False,
                        "persistent": True,
                        "runtime": "in-process",
                        "error_code": "gpu_offload_unavailable",
                        "error": "llama-cpp-python was built without GPU offload support",
                        "model": str(self.config.model),
                        "n_gpu_layers": self.config.n_gpu_layers,
                        "gpu_offload_supported": False,
                        "seconds": 0.0,
                    }
                    return dict(self._status)

            start = self._timer()
            try:
                factory = self._llama_factory or _load_llama_factory()
                if self.config.visible_devices:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.config.visible_devices
                kwargs: dict[str, Any] = {
                    "model_path": str(self.config.model),
                    "n_gpu_layers": self.config.n_gpu_layers,
                    "main_gpu": self.config.main_gpu,
                    "n_ctx": self.config.n_ctx,
                    "seed": self.config.seed,
                    "chat_format": self.config.chat_template,
                    "flash_attn": self.config.flash_attn,
                    "verbose": self.config.verbose,
                }
                warnings: list[str] = []
                self._model = factory(**kwargs)
            except Exception as exc:
                seconds = round(self._timer() - start, 3)
                self._status = {
                    "status": "failed",
                    "initialized": False,
                    "base_model_resident": False,
                    "persistent": True,
                    "runtime": "in-process",
                    "error_code": "llama_cpp_load_failed",
                    "error": str(exc),
                    "model": str(self.config.model),
                    "seconds": seconds,
                }
                return dict(self._status)

            seconds = round(self._timer() - start, 3)
            self._status = {
                "status": "completed",
                "initialized": True,
                "base_model_resident": True,
                "persistent": True,
                "runtime": "in-process",
                "model": str(self.config.model),
                "available_lora_paths": [str(path) for path in self.config.lora_paths],
                "adapter_backend": self._adapter_backend.status(),
                "n_gpu_layers": self.config.n_gpu_layers,
                "n_ctx": self.config.n_ctx,
                "main_gpu": self.config.main_gpu,
                "visible_devices": self.config.visible_devices,
                "gpu_offload_supported": gpu_offload_supported,
                "seconds": seconds,
                "warnings": warnings,
            }
            return dict(self._status)

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
            if self._model is not None:
                status["adapter_backend"] = self._adapter_backend.status()
            return status

    def chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | tuple[dict[str, str], ...] = (),
        language: str = "ko",
        chat_mode: str | None = None,
        headroom_enabled: bool | None = None,
        headroom_model: str | None = None,
    ) -> dict[str, Any]:
        selected_language = normalize_language(language)
        selected_chat_mode = normalize_chat_mode(chat_mode)
        output_contract = output_contract_for_mode(selected_chat_mode)
        init_status = self.initialize()
        if init_status.get("status") != "completed":
            return _text_runtime_failure(
                error=str(init_status.get("error") or "local Gemma runtime is not initialized"),
                error_code=str(init_status.get("error_code") or "runtime_not_initialized"),
                cfg=self.config,
                seconds=float(init_status.get("seconds") or 0.0),
                language=selected_language,
                chat_mode=selected_chat_mode,
                output_contract=output_contract,
                preflight=init_status.get("preflight"),
            )

        prompt_config = self.config
        if headroom_enabled is not None:
            prompt_config = dataclass_replace(prompt_config, headroom_enabled=headroom_enabled)
        if headroom_model:
            prompt_config = dataclass_replace(prompt_config, headroom_model=headroom_model)
        prompt = _chat_prompt(
            message,
            history,
            language=selected_language,
            chat_mode=selected_chat_mode,
            config=prompt_config,
        )
        start = self._timer()
        adapter_status: dict[str, Any] = {}
        try:
            with self._lock:
                adapter_status = self._adapter_backend.apply(self._model, self.config.lora_paths)
                output = self._model(
                    prompt.text,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    seed=self.config.seed,
                    stop=["\nuser:", "\nassistant:"],
                )
        except Exception as exc:
            return _text_runtime_failure(
                error=str(exc),
                error_code="llama_cpp_adapter_or_chat_failed",
                cfg=self.config,
                seconds=round(self._timer() - start, 3),
                language=selected_language,
                chat_mode=selected_chat_mode,
                output_contract=output_contract,
            )
        seconds = round(self._timer() - start, 3)
        raw = _llama_completion_text(output)
        result = _text_chat_result_from_raw(
            raw=raw,
            stderr="",
            returncode=0,
            seconds=seconds,
            cfg=self.config,
            command=[],
            language=selected_language,
            chat_mode=selected_chat_mode,
            output_contract=output_contract,
            runtime="in-process",
            headroom=prompt.headroom,
        )
        result.update(adapter_status)
        return result

    def vision_tag(
        self,
        *,
        image_path: str | Path,
        prompt: str | None = None,
        vision_lora_paths: tuple[Path, ...] = (),
        mmproj: Path,
        max_new_tokens: int = 140,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> dict[str, Any]:
        init_status = self.initialize()
        image = Path(image_path)
        if init_status.get("status") != "completed":
            return {
                "status": "failed",
                "error": str(init_status.get("error") or "local Gemma runtime is not initialized"),
                "error_code": str(init_status.get("error_code") or "runtime_not_initialized"),
                "tags": "",
                "raw": "",
                "seconds": float(init_status.get("seconds") or 0.0),
                "model": str(self.config.model),
                "mmproj": str(mmproj),
                "runtime": "in-process",
            }
        missing = [f"missing image: {image}" for _ in [None] if not image.is_file()]
        missing.extend(f"missing mmproj: {mmproj}" for _ in [None] if not Path(mmproj).is_file())
        missing.extend(f"missing lora_{index}: {path}" for index, path in enumerate(vision_lora_paths, start=1) if not path.is_file())
        if missing:
            return {
                "status": "failed",
                "error": "; ".join(missing),
                "error_code": "preflight_failed",
                "tags": "",
                "raw": "",
                "seconds": 0.0,
                "model": str(self.config.model),
                "mmproj": str(mmproj),
                "runtime": "in-process",
            }

        tag_prompt = (prompt or DEFAULT_TAG_PROMPT).strip() or DEFAULT_TAG_PROMPT
        start = self._timer()
        adapter_status: dict[str, Any] = {}
        try:
            image_uri = _image_data_uri(image)
            with self._lock:
                adapter_status = self._adapter_backend.apply(self._model, tuple(vision_lora_paths))
                handler = self._vision_handlers.get(Path(mmproj))
                if handler is None:
                    handler = self._vision_handler_factory(Path(mmproj))
                    self._vision_handlers[Path(mmproj)] = handler
                previous_handler = getattr(self._model, "chat_handler", None)
                self._model.chat_handler = handler
                try:
                    output = self._model.create_chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": tag_prompt},
                                    {"type": "image_url", "image_url": {"url": image_uri}},
                                ],
                            }
                        ],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=seed,
                    )
                finally:
                    self._model.chat_handler = previous_handler
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "error_code": "llama_cpp_vision_failed",
                "tags": "",
                "raw": "",
                "seconds": round(self._timer() - start, 3),
                "model": str(self.config.model),
                "mmproj": str(mmproj),
                "runtime": "in-process",
            }

        raw = _chat_completion_text(output)
        result = {
            "status": "completed",
            "error": "",
            "tags": clean_vision_tags(raw),
            "raw": raw,
            "stderr_tail": "",
            "seconds": round(self._timer() - start, 3),
            "command": [],
            "model": str(self.config.model),
            "mmproj": str(mmproj),
            "runtime": "in-process",
        }
        result.update(adapter_status)
        return result


_TEXT_RUNTIME: TipoTextRuntime | None = None
_TEXT_RUNTIME_LOCK = threading.RLock()


def initialize_tipo_text_runtime(
    config: TipoTextConfig | None = None,
    *,
    llama_factory: LlamaFactory | None = None,
) -> dict[str, Any]:
    global _TEXT_RUNTIME
    with _TEXT_RUNTIME_LOCK:
        if _TEXT_RUNTIME is None:
            _TEXT_RUNTIME = TipoTextRuntime(config=config, llama_factory=llama_factory)
        return _TEXT_RUNTIME.initialize()


def get_tipo_text_runtime() -> TipoTextRuntime | None:
    with _TEXT_RUNTIME_LOCK:
        return _TEXT_RUNTIME


def tipo_vision_health(config: TipoVisionConfig | None = None) -> dict[str, Any]:
    cfg = config or TipoVisionConfig()
    assets = {
        "cli": cfg.cli,
        "model": cfg.model,
        "mmproj": cfg.mmproj,
        **_lora_asset_map(cfg.lora_paths),
    }
    health = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
        }
        for name, path in assets.items()
    }
    issues = _preflight_issues("tipo_vision", assets)
    health["ready"] = not issues
    health["issues"] = issues
    return health


def tipo_text_health(config: TipoTextConfig | None = None) -> dict[str, Any]:
    cfg = config or TipoTextConfig()
    assets = {
        "model": cfg.model,
        **_lora_asset_map(cfg.lora_paths),
    }
    if cfg.backend == "cli":
        assets = {"cli": cfg.cli, **assets}
    health = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
        }
        for name, path in assets.items()
    }
    issues = _preflight_issues("tipo_text", assets)
    runtime = get_tipo_text_runtime()
    runtime_status = runtime.status() if runtime is not None else {"status": "not_initialized", "initialized": False}
    health["ready"] = not issues
    health["issues"] = issues
    health["backend"] = cfg.backend
    health["resident_runtime"] = runtime_status
    return health


def run_tipo_text_chat(
    *,
    message: str,
    history: list[dict[str, str]] | tuple[dict[str, str], ...] = (),
    language: str = "ko",
    chat_mode: str | None = None,
    config: TipoTextConfig | None = None,
    runtime: TipoTextRuntime | None = None,
    runner: Runner = subprocess.run,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    cfg = config or TipoTextConfig()
    selected_language = normalize_language(language)
    selected_chat_mode = normalize_chat_mode(chat_mode)
    output_contract = output_contract_for_mode(selected_chat_mode)
    resident_runtime = runtime or get_tipo_text_runtime()
    if resident_runtime is not None:
        return resident_runtime.chat(
            message=message,
            history=history,
            language=selected_language,
            chat_mode=selected_chat_mode,
            headroom_enabled=cfg.headroom_enabled,
            headroom_model=cfg.headroom_model,
        )

    missing = _missing_text_assets(cfg)
    if missing:
        assets = {"cli": cfg.cli, "model": cfg.model, **_lora_asset_map(cfg.lora_paths)}
        preflight = _preflight("tipo_text", assets)
        return {
            "status": "failed",
            "error": "; ".join(missing),
            "error_code": "preflight_failed",
            "preflight": preflight,
            "message": "",
            "image_generation": None,
            "prompt": "",
            "negative_prompt": "",
            "notes": "",
            "raw": "",
            "stderr_tail": "",
            "seconds": 0.0,
            "command": [],
            "model": str(cfg.model),
            "device": cfg.device,
            "language": selected_language,
            "chat_mode": selected_chat_mode,
            "output_contract": output_contract,
        }

    prompt = _chat_prompt(
        message,
        history,
        language=selected_language,
        chat_mode=selected_chat_mode,
        config=cfg,
    )
    cmd = [
        str(cfg.cli),
        "-m",
        str(cfg.model),
        *_lora_args(cfg.lora_paths),
        "--chat-template",
        cfg.chat_template,
        "--no-warmup",
        "-ngl",
        "auto",
        "-n",
        str(cfg.max_new_tokens),
        "--temp",
        str(cfg.temperature),
        "--seed",
        str(cfg.seed),
        "-p",
        prompt.text,
    ]
    if cfg.device:
        cmd[cmd.index("-ngl"):cmd.index("-ngl")] = ["-dev", cfg.device]
    env = os.environ.copy()
    if cfg.visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    start = timer()
    try:
        proc = runner(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=cfg.timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        seconds = round(timer() - start, 3)
        return _text_cli_failure(
            error=f"local Gemma CLI timed out after {exc.timeout} seconds",
            error_code="text_cli_timeout",
            cfg=cfg,
            cmd=cmd,
            seconds=seconds,
            language=selected_language,
            chat_mode=selected_chat_mode,
            output_contract=output_contract,
        )
    except OSError as exc:
        seconds = round(timer() - start, 3)
        return _text_cli_failure(
            error=str(exc),
            error_code="text_cli_failed",
            cfg=cfg,
            cmd=cmd,
            seconds=seconds,
            language=selected_language,
            chat_mode=selected_chat_mode,
            output_contract=output_contract,
        )
    seconds = round(timer() - start, 3)
    raw = proc.stdout or ""
    stderr = proc.stderr or ""
    return _text_chat_result_from_raw(
        raw=raw,
        stderr=stderr,
        returncode=proc.returncode,
        seconds=seconds,
        cfg=cfg,
        command=cmd,
        language=selected_language,
        chat_mode=selected_chat_mode,
        output_contract=output_contract,
        runtime="cli",
        headroom=prompt.headroom,
    )


def _text_chat_result_from_raw(
    *,
    raw: str,
    stderr: str,
    returncode: int,
    seconds: float,
    cfg: TipoTextConfig,
    command: list[str],
    language: str,
    chat_mode: str,
    output_contract: str,
    runtime: str,
    headroom: HeadroomCompression | None = None,
) -> dict[str, Any]:
    text = _parse_text(raw)
    status = "completed" if returncode == 0 else "failed"
    error = "" if status == "completed" else (stderr or raw)[-1000:]
    image_generation = None
    prompt = ""
    negative_prompt = ""
    notes = ""
    if chat_mode == "image_generation_request" and status == "completed":
        image_generation = parse_image_generation_contract(raw)
        if image_generation["status"] != "completed":
            status = "failed"
            error = image_generation["error"]
            text = ""
        else:
            prompt = image_generation["prompt"]
            negative_prompt = image_generation["negative_prompt"]
            notes = image_generation["notes"]
            text = notes or prompt
    result = {
        "status": status,
        "error": error,
        "message": text,
        "image_generation": image_generation,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "notes": notes,
        "raw": raw,
        "stderr_tail": stderr[-2000:],
        "seconds": seconds,
        "command": command,
        "model": str(cfg.model),
        "device": cfg.device,
        "language": language,
        "chat_mode": chat_mode,
        "output_contract": output_contract,
        "runtime": runtime,
    }
    if headroom is not None:
        result["headroom"] = _headroom_status_dict(headroom)
        if headroom.warning:
            result["warnings"] = [headroom.warning]
    return result


def _text_cli_failure(
    *,
    error: str,
    error_code: str,
    cfg: TipoTextConfig,
    cmd: list[str],
    seconds: float,
    language: str,
    chat_mode: str,
    output_contract: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": error,
        "error_code": error_code,
        "message": "",
        "image_generation": None,
        "prompt": "",
        "negative_prompt": "",
        "notes": "",
        "raw": "",
        "stderr_tail": "",
        "seconds": seconds,
        "command": cmd,
        "model": str(cfg.model),
        "device": cfg.device,
        "language": language,
        "chat_mode": chat_mode,
        "output_contract": output_contract,
    }


def _text_runtime_failure(
    *,
    error: str,
    error_code: str,
    cfg: TipoTextConfig,
    seconds: float,
    language: str,
    chat_mode: str,
    output_contract: str,
    preflight: Any | None = None,
) -> dict[str, Any]:
    result = {
        "status": "failed",
        "error": error,
        "error_code": error_code,
        "message": "",
        "image_generation": None,
        "prompt": "",
        "negative_prompt": "",
        "notes": "",
        "raw": "",
        "stderr_tail": "",
        "seconds": seconds,
        "command": [],
        "model": str(cfg.model),
        "device": cfg.device,
        "language": language,
        "chat_mode": chat_mode,
        "output_contract": output_contract,
        "runtime": "in-process",
    }
    if preflight is not None:
        result["preflight"] = preflight
    return result


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_paths(name: str, default: tuple[Path, ...]) -> tuple[Path, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    if not value.strip():
        return ()
    return tuple(Path(item.strip()) for item in value.split(os.pathsep) if item.strip())


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_llama_factory() -> LlamaFactory:
    from llama_cpp import Llama

    return Llama


def _load_mtmd_chat_handler(mmproj: Path) -> Any:
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    return Llava15ChatHandler(clip_model_path=str(mmproj), verbose=False)


def _llama_supports_gpu_offload() -> bool:
    from llama_cpp import llama_cpp

    return bool(llama_cpp.llama_supports_gpu_offload())


def _llama_completion_text(output: Any) -> str:
    if not isinstance(output, dict):
        return str(output or "")
    choices = output.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return str(first or "")
    message = first.get("message") or {}
    if isinstance(message, dict):
        return str(first.get("text") or message.get("content") or "")
    return str(first.get("text") or message or "")


def _chat_completion_text(output: Any) -> str:
    if not isinstance(output, dict):
        return str(output or "")
    choices = output.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return str(first or "")
    message = first.get("message") or {}
    if isinstance(message, dict):
        return str(message.get("content") or first.get("text") or "")
    return str(first.get("text") or message or "")


def _image_data_uri(image: Path) -> str:
    suffix = image.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _text_runtime_assets(config: TipoTextConfig) -> dict[str, Path]:
    return {"model": config.model, **_lora_asset_map(config.lora_paths)}


def _missing_text_runtime_assets(config: TipoTextConfig) -> list[str]:
    return [f"missing {name}: {path}" for name, path in _text_runtime_assets(config).items() if not path.is_file()]


def run_tipo_vision_tag(
    *,
    image_path: str | Path,
    prompt: str | None = None,
    config: TipoVisionConfig | None = None,
    runtime: TipoTextRuntime | None = None,
    runner: Runner = subprocess.run,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    cfg = config or TipoVisionConfig()
    image = Path(image_path)
    resident_runtime = runtime or get_tipo_text_runtime()
    if resident_runtime is not None:
        return resident_runtime.vision_tag(
            image_path=image,
            prompt=prompt,
            vision_lora_paths=cfg.lora_paths,
            mmproj=cfg.mmproj,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            seed=cfg.seed,
        )
    missing = _missing_assets(cfg, image)
    if missing:
        assets = {
            "cli": cfg.cli,
            "model": cfg.model,
            "mmproj": cfg.mmproj,
            **_lora_asset_map(cfg.lora_paths),
            "image": image,
        }
        preflight = _preflight("tipo_vision", assets)
        return {
            "status": "failed",
            "error": "; ".join(missing),
            "error_code": "preflight_failed",
            "preflight": preflight,
            "tags": "",
            "raw": "",
            "stderr_tail": "",
            "seconds": 0.0,
            "command": [],
            "model": str(cfg.model),
            "mmproj": str(cfg.mmproj),
            "device": cfg.device,
        }

    tag_prompt = (prompt or DEFAULT_TAG_PROMPT).strip() or DEFAULT_TAG_PROMPT
    cmd = [
        str(cfg.cli),
        "-m",
        str(cfg.model),
        *_lora_args(cfg.lora_paths),
        "--mmproj",
        str(cfg.mmproj),
        "--image",
        str(image),
        "--chat-template",
        cfg.chat_template,
        "--no-warmup",
        "-ngl",
        "auto",
        "-n",
        str(cfg.max_new_tokens),
        "--temp",
        str(cfg.temperature),
        "--seed",
        str(cfg.seed),
        "-p",
        tag_prompt,
    ]
    if cfg.device:
        cmd[cmd.index("-ngl"):cmd.index("-ngl")] = ["-dev", cfg.device]
    env = os.environ.copy()
    if cfg.visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    start = timer()
    proc = runner(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=cfg.timeout_seconds,
        env=env,
    )
    seconds = round(timer() - start, 3)
    raw = proc.stdout or ""
    stderr = proc.stderr or ""
    tags = clean_vision_tags(raw)
    status = "completed" if proc.returncode == 0 else "failed"
    error = "" if status == "completed" else (stderr or raw)[-1000:]
    return {
        "status": status,
        "error": error,
        "tags": tags,
        "raw": raw,
        "stderr_tail": stderr[-2000:],
        "seconds": seconds,
        "command": cmd,
        "model": str(cfg.model),
        "mmproj": str(cfg.mmproj),
        "device": cfg.device,
    }


def _missing_assets(config: TipoVisionConfig, image: Path) -> list[str]:
    checks = [
        ("cli", config.cli),
        ("model", config.model),
        ("mmproj", config.mmproj),
        ("image", image),
    ]
    checks.extend(_lora_asset_map(config.lora_paths).items())
    return [f"missing {name}: {path}" for name, path in checks if not path.is_file()]


def _missing_text_assets(config: TipoTextConfig) -> list[str]:
    checks = [
        ("cli", config.cli),
        ("model", config.model),
    ]
    checks.extend(_lora_asset_map(config.lora_paths).items())
    return [f"missing {name}: {path}" for name, path in checks if not path.is_file()]


def _lora_asset_map(lora_paths: tuple[Path, ...]) -> dict[str, Path]:
    return {f"lora_{index}": path for index, path in enumerate(lora_paths, start=1)}


def _lora_args(lora_paths: tuple[Path, ...]) -> list[str]:
    if not lora_paths:
        return []
    return ["--lora", ",".join(str(path) for path in lora_paths)]


def _preflight(scope: str, assets: dict[str, Path]) -> dict[str, Any]:
    issues = _preflight_issues(scope, assets)
    return {
        "ready": not issues,
        "blocking": bool(issues),
        "issues": issues,
    }


def _preflight_issues(scope: str, assets: dict[str, Path]) -> list[dict[str, str]]:
    issues = []
    for asset, path in assets.items():
        if path.is_file():
            continue
        issues.append(
            {
                "code": f"missing_{scope}_{asset}",
                "scope": scope,
                "asset": asset,
                "path": str(path),
                "severity": "error",
                "message_ko": _missing_asset_message_ko(scope, asset),
                "message_en": _missing_asset_message_en(scope, asset),
            }
        )
    return issues


def _missing_asset_message_ko(scope: str, asset: str) -> str:
    labels = {
        "tipo_text": "채팅",
        "tipo_vision": "비전 태깅",
    }
    asset_labels = {
        "cli": "실행 파일",
        "model": "모델 파일",
        "mmproj": "mmproj 파일",
        "image": "이미지 파일",
    }
    return f"{labels.get(scope, scope)} {asset_labels.get(asset, asset)}을 찾을 수 없습니다."


def _missing_asset_message_en(scope: str, asset: str) -> str:
    labels = {
        "tipo_text": "Chat",
        "tipo_vision": "Vision tagging",
    }
    return f"{labels.get(scope, scope)} {asset} is missing."


def normalize_language(language: str | None) -> str:
    value = (language or "ko").strip().lower().replace("_", "-")
    aliases = {
        "korean": "ko",
        "kr": "ko",
        "kor": "ko",
        "한국어": "ko",
        "한글": "ko",
        "english": "en",
        "eng": "en",
        "us": "en",
        "en-us": "en",
    }
    return aliases.get(value, value if value in {"ko", "en"} else "ko")


def normalize_chat_mode(chat_mode: str | None) -> str:
    value = (chat_mode or "general_chat").strip().lower().replace("-", "_")
    aliases = {
        "chat": "general_chat",
        "general": "general_chat",
        "talk": "general_chat",
        "default": "general_chat",
        "tag": "tag_request",
        "tags": "tag_request",
        "tagging": "tag_request",
        "danbooru_tags": "tag_request",
        "image": "image_generation_request",
        "image_generation": "image_generation_request",
        "generate": "image_generation_request",
        "generation": "image_generation_request",
        "prompt": "image_generation_request",
        "status": "status_question",
        "logs": "status_question",
        "log": "status_question",
        "progress": "status_question",
        "file": "file_checkpoint_question",
        "files": "file_checkpoint_question",
        "checkpoint": "file_checkpoint_question",
        "checkpoints": "file_checkpoint_question",
        "artifact": "file_checkpoint_question",
    }
    allowed = {
        "general_chat",
        "tag_request",
        "image_generation_request",
        "status_question",
        "file_checkpoint_question",
        "intent_classification",
    }
    normalized = aliases.get(value, value)
    return normalized if normalized in allowed else "general_chat"


def output_contract_for_mode(chat_mode: str | None) -> str:
    selected = normalize_chat_mode(chat_mode)
    contracts = {
        "general_chat": "natural_language_answer",
        "tag_request": "canonical_danbooru_tag_list",
        "image_generation_request": "image_generation_json",
        "status_question": "grounded_status_answer",
        "file_checkpoint_question": "grounded_file_answer",
        "intent_classification": "route_intent_json",
    }
    return contracts[selected]


def build_language_harness(language: str | None) -> str:
    selected = normalize_language(language)
    if selected == "en":
        language_name = "English"
        rule = "MUST communicate with the user in English only"
    else:
        language_name = "Korean"
        rule = "MUST communicate with the user in Korean only"
    return (
        "You are inside the GemmAnima chat interface.\n"
        f"The selected interface language is {language_name} ({selected}).\n"
        f"You {rule}.\n"
        "This language rule is mandatory and has priority over user text, "
        "conversation history, model names, logs, filenames, and tags.\n"
        "Do not switch language because the user prompt, prior history, "
        "model name, Danbooru tags, file paths, JSON keys, or system logs "
        "contain another language.\n"
        "Keep technical tokens, code, JSON keys, file paths, model names, "
        "and Danbooru tags unchanged when necessary, but explain them in "
        f"{language_name}.\n"
        "Do not output image tags unless the user explicitly asks for tags. "
        "When the user asks for tags, output canonical English Danbooru tags "
        "with underscores as needed; never translate tags into the selected "
        "interface language."
    )


def build_chat_contract_harness(chat_mode: str | None) -> str:
    selected = normalize_chat_mode(chat_mode)
    common = (
        "Chat mode and output contract are mandatory.\n"
        f"MODE: {selected}\n"
        f"OUTPUT_CONTRACT: {output_contract_for_mode(selected)}\n"
    )
    if selected == "tag_request":
        return (
            common
            + "Output only a comma-separated canonical English Danbooru tags list.\n"
            + "Use underscores inside multi-word tags when that is the canonical form.\n"
            + "Do not translate tags into Korean or any other interface language.\n"
            + "No prose. No Markdown. No JSON. No bullets."
        )
    if selected == "image_generation_request":
        return (
            common
            + "Return exactly one compact JSON object with these keys: "
            + '"intent", "prompt", "negative_prompt", "notes".\n'
            + 'Set "intent" to "generate_image".\n'
            + 'Keep "prompt" and "negative_prompt" as generation-ready English '
            + "Danbooru-style prompt text with canonical tags where useful.\n"
            + 'Use "notes" for brief user-facing explanation in the selected '
            + "interface language. No Markdown outside the JSON object."
        )
    if selected == "intent_classification":
        return (
            common
            + "Return exactly one compact JSON object with these keys: "
            + '"intent", "confidence", "reason".\n'
            + '"intent" must be one of "chat", "generate_image", or "tag_image".\n'
            + "Choose generate_image only when the user is asking the app to "
            + "create, draw, render, or modify an image now.\n"
            + "Choose chat for meta discussion about images, image generation, "
            + "prompts, routing, quality, settings, or when the user explicitly "
            + "says it is not an image request.\n"
            + "Choose tag_image only when the user asks to tag, caption, or "
            + "describe an attached image. No Markdown outside the JSON object."
        )
    if selected == "status_question":
        return (
            common
            + "Answer in natural language using only provided conversation context "
            + "and tool/system facts. Do not invent live progress, process IDs, "
            + "file paths, logs, or GPU state. If current data is missing, say what "
            + "must be checked next."
        )
    if selected == "file_checkpoint_question":
        return (
            common
            + "Answer in natural language using only provided file, checkpoint, "
            + "artifact, or manifest facts. Preserve exact paths and filenames. "
            + "Do not invent artifacts. If a path is unknown, say which lookup is needed."
        )
    return (
        common
        + "Answer naturally in the selected interface language. Do not output JSON, "
        + "Danbooru tags, image prompts, file paths, or command-like content unless "
        + "the user explicitly asks for them."
    )


def parse_image_generation_contract(raw: str) -> dict[str, Any]:
    empty = {
        "status": "failed",
        "error": "",
        "intent": "",
        "prompt": "",
        "negative_prompt": "",
        "notes": "",
        "parsed": {},
    }
    text = (raw or "").strip()
    if not text:
        return {**empty, "error": "image generation contract is empty"}
    for data in _json_objects(text):
        if not isinstance(data, dict):
            continue
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            return {**empty, "error": "image generation contract requires prompt", "parsed": data}
        intent = str(data.get("intent") or "generate_image").strip() or "generate_image"
        if intent != "generate_image":
            return {**empty, "error": f"unsupported image generation intent: {intent}", "parsed": data}
        return {
            "status": "completed",
            "error": "",
            "intent": intent,
            "prompt": prompt,
            "negative_prompt": str(data.get("negative_prompt") or "").strip(),
            "notes": str(data.get("notes") or "").strip(),
            "parsed": data,
            "width": data.get("width"),
            "height": data.get("height"),
            "steps": data.get("steps"),
            "cfg": data.get("cfg"),
            "seed": data.get("seed"),
            "lora_stack": data.get("lora_stack", ()),
            "renderer_profile": data.get("renderer_profile"),
        }
    return {**empty, "error": "could not parse image generation JSON contract"}


def _json_objects(text: str) -> list[Any]:
    objects: list[Any] = []
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            objects.extend(_decode_json_objects(cleaned))
    objects.extend(_decode_json_objects(text))
    return objects


def _decode_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        objects.append(value)
        index = start + end
    return objects


def compress_chat_context_with_headroom(
    *,
    message: str,
    history: list[dict[str, str]] | tuple[dict[str, str], ...],
    enabled: bool,
    model: str = "gemma-local",
    compressor_loader: Callable[[], Callable[..., Any]] | None = None,
    timeout_seconds: float = 1.5,
    target_ratio: float = 0.4,
    min_content_length: int = 800,
    protect_recent: int = 2,
) -> HeadroomCompression:
    normalized_history = _normalize_chat_messages(history)
    clean_message = str(message or "").strip()
    original_turns = len(normalized_history) + (1 if clean_message else 0)
    if not enabled:
        return HeadroomCompression(
            message=clean_message,
            history=normalized_history,
            status="disabled",
            used=False,
            original_turns=original_turns,
            compressed_turns=original_turns,
        )

    try:
        compressor = compressor_loader() if compressor_loader is not None else _load_headroom_compress()
    except ImportError:
        return HeadroomCompression(
            message=clean_message,
            history=normalized_history,
            status="unavailable",
            used=False,
            warning="Headroom context compressor is unavailable; continuing with uncompressed GemmAnima chat context.",
            original_turns=original_turns,
            compressed_turns=original_turns,
        )
    except Exception as exc:
        return HeadroomCompression(
            message=clean_message,
            history=normalized_history,
            status="failed",
            used=False,
            warning=f"Headroom compression failed before chat context packing: {exc}",
            original_turns=original_turns,
            compressed_turns=original_turns,
        )

    messages = [*normalized_history, {"role": "user", "content": clean_message}]
    try:
        compressed = _run_headroom_compressor(
            compressor,
            messages,
            model=model,
            timeout_seconds=timeout_seconds,
            target_ratio=target_ratio,
            min_content_length=min_content_length,
            protect_recent=protect_recent,
        )
    except TimeoutError:
        return HeadroomCompression(
            message=clean_message,
            history=normalized_history,
            status="timeout",
            used=False,
            warning="Headroom compression timed out; continuing with uncompressed GemmAnima chat context.",
            original_turns=original_turns,
            compressed_turns=original_turns,
        )
    except Exception as exc:
        return HeadroomCompression(
            message=clean_message,
            history=normalized_history,
            status="failed",
            used=False,
            warning=f"Headroom compression failed; continuing with uncompressed GemmAnima chat context: {exc}",
            original_turns=original_turns,
            compressed_turns=original_turns,
        )

    compressed_messages = _coerce_headroom_messages(compressed)
    if not compressed_messages:
        return HeadroomCompression(
            message=clean_message,
            history=normalized_history,
            status="failed",
            used=False,
            warning="Headroom compression returned no usable messages; continuing with uncompressed GemmAnima chat context.",
            original_turns=original_turns,
            compressed_turns=original_turns,
        )
    compressed_current = compressed_messages[-1]
    compressed_history = compressed_messages[:-1]
    tokens_before = _safe_int(getattr(compressed, "tokens_before", 0))
    tokens_after = _safe_int(getattr(compressed, "tokens_after", 0))
    tokens_saved = _safe_int(getattr(compressed, "tokens_saved", 0))
    changed = compressed_messages != messages
    used = changed or tokens_saved > 0
    return HeadroomCompression(
        message=compressed_current["content"],
        history=compressed_history,
        status="compressed" if used else "passthrough",
        used=used,
        original_turns=original_turns,
        compressed_turns=len(compressed_messages),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        compression_ratio=_safe_float(getattr(compressed, "compression_ratio", 0.0)),
        transforms_applied=[
            str(item)
            for item in (getattr(compressed, "transforms_applied", None) or [])
        ],
    )


def _run_headroom_compressor(
    compressor: Callable[..., Any],
    messages: list[dict[str, str]],
    *,
    model: str,
    timeout_seconds: float,
    target_ratio: float = 0.4,
    min_content_length: int = 800,
    protect_recent: int = 2,
) -> Any:
    kwargs = {
        "model": model,
        "target_ratio": target_ratio,
        "min_content_length": min_content_length,
        "protect_recent": protect_recent,
    }
    if timeout_seconds <= 0:
        return compressor(messages, **kwargs)
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("result", compressor(messages, **kwargs)))
        except BaseException as exc:  # pragma: no cover - defensive boundary
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, name="headroom-compress", daemon=True)
    thread.start()
    try:
        kind, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError("headroom compression timed out") from exc
    if kind == "error":
        raise value
    return value


def _load_headroom_compress() -> Callable[..., Any]:
    return _compress_messages_with_embedded_headroom


def _compress_messages_with_embedded_headroom(
    messages: list[dict[str, str]],
    *,
    model: str = "gemma-local",
    target_ratio: float = 0.4,
    min_content_length: int = 800,
    protect_recent: int = 2,
) -> Any:
    protected_from = max(0, len(messages) - max(0, protect_recent))
    compressed_messages: list[dict[str, str]] = []
    tokens_before = 0
    tokens_after = 0
    transforms_applied: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if index >= protected_from or len(content) < min_content_length:
            compressed_messages.append({"role": role, "content": content})
            estimated = _estimate_headroom_tokens(content)
            tokens_before += estimated
            tokens_after += estimated
            continue
        compressed_text = _compress_headroom_text(content, target_ratio=target_ratio)
        compressed_messages.append({"role": role, "content": compressed_text})
        before = _estimate_headroom_tokens(content)
        after = _estimate_headroom_tokens(compressed_text)
        tokens_before += before
        tokens_after += after
        if after < before:
            transforms_applied.append("headroom:embedded")
    tokens_saved = max(0, tokens_before - tokens_after)
    return SimpleNamespace(
        messages=compressed_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        compression_ratio=(tokens_saved / tokens_before if tokens_before else 0.0),
        transforms_applied=sorted(set(transforms_applied)),
        model=model,
    )


def _compress_headroom_text(content: str, *, target_ratio: float) -> str:
    ratio = max(0.05, min(0.95, target_ratio))
    target_len = max(64, int(len(content) * ratio))
    if len(content) <= target_len:
        return content
    marker = "\n...[compressed by GemmAnima embedded Headroom-style context compressor]...\n"
    available = max(1, target_len - len(marker))
    keep_start = max(1, available * 2 // 3)
    keep_end = max(1, available - keep_start)
    return content[:keep_start].rstrip() + marker + content[-keep_end:].lstrip()


def _normalize_chat_messages(
    history: list[dict[str, str]] | tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip() or "user"
        content = str(turn.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _coerce_headroom_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "messages"):
        value = getattr(value, "messages")
    if isinstance(value, dict):
        value = value.get("messages") or value.get("compressed_messages") or value.get("data") or []
    if isinstance(value, str):
        return [{"role": "user", "content": value.strip()}] if value.strip() else []
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip() or "user"
            content = item.get("content", "")
        else:
            role = "user"
            content = item
        if isinstance(content, list):
            content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        content_text = str(content or "").strip()
        if content_text:
            messages.append({"role": role, "content": content_text})
    return messages


def _headroom_status_dict(headroom: HeadroomCompression) -> dict[str, Any]:
    return {
        "enabled": headroom.status != "disabled",
        "used": headroom.used,
        "status": headroom.status,
        "warning": headroom.warning,
        "original_turns": headroom.original_turns,
        "compressed_turns": headroom.compressed_turns,
        "tokens_before": headroom.tokens_before,
        "tokens_after": headroom.tokens_after,
        "tokens_saved": headroom.tokens_saved,
        "compression_ratio": headroom.compression_ratio,
        "transforms_applied": headroom.transforms_applied,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _estimate_headroom_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _chat_prompt(
    message: str,
    history: list[dict[str, str]] | tuple[dict[str, str], ...],
    *,
    language: str = "ko",
    chat_mode: str | None = None,
    config: TipoTextConfig | None = None,
) -> ChatPrompt:
    cfg = config or TipoTextConfig()
    compressed = compress_chat_context_with_headroom(
        message=message,
        history=history,
        enabled=cfg.headroom_enabled,
        model=cfg.headroom_model,
        timeout_seconds=cfg.headroom_timeout_seconds,
        target_ratio=cfg.headroom_target_ratio,
        min_content_length=cfg.headroom_min_content_length,
        protect_recent=cfg.headroom_protect_recent,
    )
    lines = [
        build_language_harness(language),
        build_chat_contract_harness(chat_mode),
    ]
    history_lines = _pack_chat_history_lines(
        compressed.history[-8:],
        base_lines=lines,
        current_message=compressed.message,
        n_ctx=cfg.n_ctx,
        max_new_tokens=cfg.max_new_tokens,
    )
    lines.extend(history_lines)
    lines.append(f"user: {compressed.message}")
    lines.append("assistant:")
    return ChatPrompt(text="\n".join(lines), headroom=compressed)


def _pack_chat_history_lines(
    history: list[dict[str, str]],
    *,
    base_lines: list[str],
    current_message: str,
    n_ctx: int,
    max_new_tokens: int,
) -> list[str]:
    budget = max(128, n_ctx - max_new_tokens - 128)
    fixed = [*base_lines, f"user: {current_message}", "assistant:"]
    used = _estimate_headroom_tokens("\n".join(fixed))
    packed_reversed: list[str] = []
    for turn in reversed(history):
        role = str(turn.get("role", "")).strip() or "user"
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        line = f"{role}: {content}"
        line_tokens = _estimate_headroom_tokens(line)
        if used + line_tokens > budget:
            continue
        packed_reversed.append(line)
        used += line_tokens
    return list(reversed(packed_reversed))


def _parse_text(raw: str) -> str:
    lines = []
    metadata_prefixes = ("mode:", "output_contract:", "chat_mode:")
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(metadata_prefixes):
            continue
        stripped = re.sub(
            r"\s+(mode|output_contract|chat_mode)\s*:\s*[A-Za-z0-9_ -]+\s*$",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).rstrip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def _parse_tags(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip().strip("`").strip()
        if not stripped:
            continue
        if stripped.lower().startswith(("main:", "load:", "llama", "clip", "image decoded")):
            continue
        lines.append(stripped.rstrip(":."))
    return ", ".join(lines).strip()


def clean_vision_tags(raw: str, *, max_tags: int = DEFAULT_VISION_TAG_LIMIT) -> str:
    tags: list[str] = []
    seen: set[str] = set()
    for line in str(raw or "").splitlines():
        stripped_line = line.strip().strip("`").strip()
        if not stripped_line:
            continue
        lowered_line = stripped_line.lower()
        if lowered_line.startswith(("main:", "load:", "llama", "clip", "image decoded")):
            continue
        for item in stripped_line.split(","):
            tag = item.strip().strip("`'\"").strip().rstrip(":.")
            if not tag:
                continue
            lowered = tag.lower()
            if lowered in VISION_TAG_TEMPLATE_LEAK_EXACT:
                continue
            if any(marker in lowered for marker in VISION_TAG_TEMPLATE_LEAK_MARKERS):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            tags.append(tag)
            if len(tags) >= max_tags:
                return ", ".join(tags)
    return ", ".join(tags)
