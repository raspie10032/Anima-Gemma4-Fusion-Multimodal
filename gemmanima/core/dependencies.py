from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from typing import Literal


DependencyKind = Literal["stdlib", "embedded", "runtime_engine", "developer", "training"]


@dataclass(frozen=True)
class RuntimeDependency:
    name: str
    import_name: str
    kind: DependencyKind
    required_for: tuple[str, ...]
    bundled_policy: str

    def available(self) -> bool:
        if self.kind == "stdlib":
            return True
        return importlib.util.find_spec(self.import_name) is not None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "import_name": self.import_name,
            "kind": self.kind,
            "available": self.available(),
            "required_for": list(self.required_for),
            "bundled_policy": self.bundled_policy,
        }


DEPENDENCIES: tuple[RuntimeDependency, ...] = (
    RuntimeDependency(
        "Python standard library",
        "json",
        "stdlib",
        ("launcher", "chat harness", "model downloader", "embedded context compression"),
        "Built in to Python; never installed by GemmAnima.",
    ),
    RuntimeDependency(
        "Embedded Headroom-style context compressor",
        "gemmanima.modules.tipo_runtime",
        "embedded",
        ("chat context compression",),
        "Implemented inside GemmAnima; chopratejas/headroom is a reference, not a runtime package dependency.",
    ),
    RuntimeDependency(
        "llama-cpp-python",
        "llama_cpp",
        "runtime_engine",
        ("resident Gemma GGUF chat", "vision tagger"),
        "Bundle with the app/runtime environment. GemmAnima must not pip-install it at launch.",
    ),
    RuntimeDependency(
        "PyTorch",
        "torch",
        "runtime_engine",
        ("in-process Anima rendering", "bridge tensors"),
        "Bundle with the selected CUDA runtime environment. Keep training/cache GPU policy separate.",
    ),
    RuntimeDependency(
        "Pillow",
        "PIL",
        "runtime_engine",
        ("image IO", "comparison utilities", "render outputs"),
        "Bundle with the app/runtime environment when real image rendering or image inspection is enabled.",
    ),
    RuntimeDependency(
        "NumPy",
        "numpy",
        "runtime_engine",
        ("image comparison", "render metrics"),
        "Bundle with image runtime utilities; not installed dynamically.",
    ),
    RuntimeDependency(
        "safetensors",
        "safetensors",
        "runtime_engine",
        ("bridge/model tensor loading",),
        "Bundle with model runtime utilities; not installed dynamically.",
    ),
    RuntimeDependency(
        "PyYAML",
        "yaml",
        "runtime_engine",
        ("optional YAML config snapshots",),
        "Prefer JSON for app-critical config; bundle only if YAML config editing is enabled.",
    ),
    RuntimeDependency(
        "pytest",
        "pytest",
        "developer",
        ("test suite",),
        "Developer/test dependency only; not required for end-user runtime.",
    ),
    RuntimeDependency(
        "transformers",
        "transformers",
        "training",
        ("training/cache helper scripts",),
        "Training environment dependency only; not required for the standalone chat GUI.",
    ),
)


def dependency_audit() -> dict[str, object]:
    items = [dependency.to_json_dict() for dependency in DEPENDENCIES]
    missing_runtime = [
        item["name"]
        for item in items
        if item["kind"] == "runtime_engine" and not item["available"]
    ]
    missing_developer = [
        item["name"]
        for item in items
        if item["kind"] == "developer" and not item["available"]
    ]
    missing_training = [
        item["name"]
        for item in items
        if item["kind"] == "training" and not item["available"]
    ]
    return {
        "status": "ready" if not missing_runtime else "missing_runtime_dependencies",
        "python": sys.version.split()[0],
        "auto_install_policy": "disabled",
        "network_policy": "model assets only; no Python package installation at app launch",
        "missing_runtime": missing_runtime,
        "missing_developer": missing_developer,
        "missing_training": missing_training,
        "dependencies": items,
    }


def main() -> int:
    print(json.dumps(dependency_audit(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
