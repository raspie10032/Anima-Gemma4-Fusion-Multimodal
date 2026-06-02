import glob
import importlib.util
import os
import sys
from pathlib import Path


CORE_TRAINER = Path(r"E:\anima_gemma_swap\scripts\core\08_train_stream_batched.py")


def load_core():
    spec = importlib.util.spec_from_file_location("train_stream_batched_core", CORE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load trainer: {CORE_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shard_paths_multi(root):
    roots = []
    for part in str(root).split(";"):
        part = part.strip().strip('"')
        if part:
            roots.append(part)
    paths = []
    for item in roots:
        paths.extend(glob.glob(os.path.join(item, "*.pt")))
    return sorted(paths)


def main():
    core = load_core()
    core.shard_paths = shard_paths_multi
    core.main()


if __name__ == "__main__":
    main()
