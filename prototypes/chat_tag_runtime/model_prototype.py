from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from purpose import CURRENT_RUNTIME_BRANCH, FUTURE_RUNTIME_BRANCH, INTERNAL_GENERATOR_ID, MODEL_ARCHITECTURE_ID, REQUIRED_IMAGE_ENGINE


DEFAULT_LLAMA_DIR = Path(r"D:\Projects\training\llama_b9209_cuda")
DEFAULT_CHAT_CLI = DEFAULT_LLAMA_DIR / "llama-cli.exe"
DEFAULT_COMPLETION_CLI = DEFAULT_LLAMA_DIR / "llama-completion.exe"
DEFAULT_TAG_CLI = DEFAULT_LLAMA_DIR / "llama-mtmd-cli.exe"
DEFAULT_CHAT_MODEL = Path(r"D:\Projects\training\out\gemma-4-E2B-it-heretic-ara-Q4_K_M.gguf")
DEFAULT_PLANNER_MODEL = Path(r"D:\Projects\training\out\gemma4-tipo-ko-v2-Q4_K_M.gguf")
DEFAULT_PLANNER_LORA_SOURCE = Path(r"D:\Projects\training\out\lora\adapter_model.safetensors")
DEFAULT_PLANNER_LORA_GGUF = Path(r"D:\Projects\training\out\lora\adapter_model.f16.gguf")
DEFAULT_VISION_UNDERSTAND_MODEL = Path(r"D:\Projects\training\out\_completed\gemma4-tipo-vision-Q4_K_M.gguf")
DEFAULT_VISION_UNDERSTAND_MMPROJ = Path(r"D:\Projects\training\out\_completed\gemma4-tipo-vision.mmproj-f16.gguf")
DEFAULT_TAG_MODEL = Path(r"D:\Projects\training\out\_completed\gemma4-tipo-vision-Q4_K_M.gguf")
DEFAULT_TAG_MMPROJ = Path(r"D:\Projects\training\out\_completed\gemma4-tipo-vision.mmproj-f16.gguf")


@dataclass(frozen=True)
class GemmaCorePrototype:
    runtime: str
    cli: Path
    model: Path
    architecture: str = MODEL_ARCHITECTURE_ID
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "runtime": self.runtime,
            "cli": str(self.cli),
            "model": str(self.model),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ModelPrototype:
    task: str
    runtime: str
    cli: Path
    model: Path
    lora: Path | None = None
    lora_source: Path | None = None
    merged_model_default: Path | None = None
    mmproj: Path | None = None
    prompt_contract: str = ""
    runtime_branch: str = CURRENT_RUNTIME_BRANCH
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "runtime": self.runtime,
            "runtime_branch": self.runtime_branch,
            "cli": str(self.cli),
            "model": str(self.model),
            "lora": str(self.lora) if self.lora else "",
            "lora_source": str(self.lora_source) if self.lora_source else "",
            "merged_model_default": str(self.merged_model_default) if self.merged_model_default else "",
            "mmproj": str(self.mmproj) if self.mmproj else "",
            "prompt_contract": self.prompt_contract,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ChatTagModelSet:
    name: str
    core: GemmaCorePrototype
    chat: ModelPrototype
    planner: ModelPrototype
    vision_understander: ModelPrototype
    tag: ModelPrototype
    visible_devices: str = "0"
    device: str = "CUDA0"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": MODEL_ARCHITECTURE_ID,
            "single_gemma4_core_required": True,
            "preserve_gemma4_chat": True,
            "preserve_gemma4_multimodal_vision": True,
            "separate_image_understanding_and_tagging": True,
            "hidden_state_bridge_required": True,
            "direct_anima_synthesis_required": True,
            "role_split_ggufs_are_temporary": True,
            "runtime_branch": CURRENT_RUNTIME_BRANCH,
            "future_runtime_branch": FUTURE_RUNTIME_BRANCH,
            "quantized_runtime_required_now": True,
            "visible_devices": self.visible_devices,
            "device": self.device,
            "core": self.core.to_json_dict(),
            "attached_modules": {
                "chat": self.chat.to_json_dict(),
                "image_planner": self.planner.to_json_dict(),
                "image_understander": self.vision_understander.to_json_dict(),
                "vision_tagger": self.tag.to_json_dict(),
                "image_generator": {
                    "task": "image",
                    "runtime": REQUIRED_IMAGE_ENGINE,
                    "generator": INTERNAL_GENERATOR_ID,
                    "lineage": "Anima/GEMMANIMA",
                    "conditioning": "Gemma4 hidden states, not final prompt text",
                    "external_backend": False,
                },
            },
            "chat": self.chat.to_json_dict(),
            "planner": self.planner.to_json_dict(),
            "vision_understander": self.vision_understander.to_json_dict(),
            "tag": self.tag.to_json_dict(),
        }


def default_model_set(
    *,
    chat_cli: Path = DEFAULT_CHAT_CLI,
    chat_model: Path = DEFAULT_CHAT_MODEL,
    planner_cli: Path = DEFAULT_COMPLETION_CLI,
    planner_model: Path = DEFAULT_CHAT_MODEL,
    planner_lora: Path = DEFAULT_PLANNER_LORA_GGUF,
    planner_lora_source: Path = DEFAULT_PLANNER_LORA_SOURCE,
    planner_merged_model_default: Path = DEFAULT_PLANNER_MODEL,
    vision_understand_cli: Path = DEFAULT_TAG_CLI,
    vision_understand_model: Path = DEFAULT_VISION_UNDERSTAND_MODEL,
    vision_understand_mmproj: Path = DEFAULT_VISION_UNDERSTAND_MMPROJ,
    tag_cli: Path = DEFAULT_TAG_CLI,
    tag_model: Path = DEFAULT_TAG_MODEL,
    tag_mmproj: Path = DEFAULT_TAG_MMPROJ,
) -> ChatTagModelSet:
    return ChatTagModelSet(
        name="gemma4-local-chat-builtin-image-poc",
        core=GemmaCorePrototype(
            runtime="llama-cli",
            cli=chat_cli,
            model=chat_model,
            notes=(
                "target architecture preserves Gemma4 chat and multimodal vision",
                "Anima synthesis must consume Gemma4 hidden states directly",
                "current role-specific GGUFs are temporary smoke-test stand-ins",
                "current runtime branch is quantized llama.cpp; unquantized execution is a later branch",
            ),
        ),
        chat=ModelPrototype(
            task="chat",
            runtime="llama-cli",
            cli=chat_cli,
            model=chat_model,
            prompt_contract="general assistant response; never force image tags unless asked",
            notes=("stand-in module; smoke passed with a normal assistant reply",),
        ),
        planner=ModelPrototype(
            task="image_planner",
            runtime="llama-completion",
            cli=planner_cli,
            model=planner_model,
            lora=planner_lora,
            lora_source=planner_lora_source,
            merged_model_default=planner_merged_model_default,
            prompt_contract="TIPO-style Partial tags continuation for image generation conditioning",
            notes=(
                "auxiliary stand-in only; final generation path is hidden-state conditioning into Anima",
                "current design loads the quantized Gemma4 base GGUF and applies a llama.cpp-compatible f16 LoRA GGUF",
                "PEFT safetensors LoRA is source material and must be converted before llama.cpp --lora use",
                "merged planner GGUF remains a smoke default, not the target module layout",
            ),
        ),
        vision_understander=ModelPrototype(
            task="image_understanding",
            runtime="llama-mtmd-cli",
            cli=vision_understand_cli,
            model=vision_understand_model,
            mmproj=vision_understand_mmproj,
            prompt_contract="natural-language visual understanding and hidden-state extraction; not comma-tag output",
            notes=(
                "required vision module 1/2: understand image content for chat and hidden-state conditioning",
                "must stay separate from image-to-tags even if early smoke assets share a llama-mtmd runtime",
                "final module should preserve Gemma4 multimodal image understanding, not collapse into tag planning",
            ),
        ),
        tag=ModelPrototype(
            task="tag",
            runtime="llama-mtmd-cli",
            cli=tag_cli,
            model=tag_model,
            mmproj=tag_mmproj,
            prompt_contract="comma-separated English Danbooru tags only",
            notes=("auxiliary stand-in only; final image vision must preserve Gemma4 multimodal behavior",),
        ),
    )


def model_health(model_set: ChatTagModelSet | None = None) -> dict[str, Any]:
    models = model_set or default_model_set()
    assets = {
        "core.model": models.core.model,
        "chat.cli": models.chat.cli,
        "chat.model": models.chat.model,
        "planner.cli": models.planner.cli,
        "planner.model": models.planner.model,
        "planner.lora": models.planner.lora,
        "planner.lora_source": models.planner.lora_source,
        "planner.merged_model_default": models.planner.merged_model_default,
        "vision_understander.cli": models.vision_understander.cli,
        "vision_understander.model": models.vision_understander.model,
        "vision_understander.mmproj": models.vision_understander.mmproj,
        "tag.cli": models.tag.cli,
        "tag.model": models.tag.model,
        "tag.mmproj": models.tag.mmproj,
    }
    checked = {
        name: {"path": str(path), "exists": bool(path and path.is_file())}
        for name, path in assets.items()
    }
    return {
        "name": models.name,
        "ready": all(item["exists"] for item in checked.values()),
        "assets": checked,
        "models": models.to_json_dict(),
    }
