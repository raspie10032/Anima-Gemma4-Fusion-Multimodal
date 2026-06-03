from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_VISIBLE_DEVICES", "1")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_MAIN_GPU", "0")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_MAX_NEW_TOKENS", "64")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_HEADROOM_ENABLED", "1")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_HEADROOM_TIMEOUT_SECONDS", "2.0")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_HEADROOM_MIN_CONTENT_LENGTH", "500")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_HEADROOM_TARGET_RATIO", "0.4")
os.environ.setdefault("GEMMANIMA_TIPO_TEXT_HEADROOM_PROTECT_RECENT", "1")

from gemmanima.modules.tipo_runtime import TipoTextConfig, TipoTextRuntime, run_tipo_text_chat  # noqa: E402


def main() -> int:
    cfg = TipoTextConfig()
    preflight = {
        "visible_devices": cfg.visible_devices,
        "main_gpu": cfg.main_gpu,
        "n_ctx": cfg.n_ctx,
        "headroom_enabled": cfg.headroom_enabled,
        "headroom_timeout_seconds": cfg.headroom_timeout_seconds,
        "headroom_min_content_length": cfg.headroom_min_content_length,
        "headroom_target_ratio": cfg.headroom_target_ratio,
        "headroom_protect_recent": cfg.headroom_protect_recent,
        "model_exists": cfg.model.is_file(),
        "lora_exists": [path.is_file() for path in cfg.lora_paths],
    }
    runtime = TipoTextRuntime(config=cfg)
    init_status = runtime.initialize()
    result = run_tipo_text_chat(
        message="Reply in Korean. Tell me whether Headroom compression was applied.",
        language="ko",
        chat_mode="general_chat",
        history=[
            {
                "role": "user",
                "content": (
                    "Old GemmAnima context: local Gemma resident model, Anima image bridge, "
                    "pose/action tag stability, Korean chat harness, English Danbooru tags, "
                    "RTX 5060 chat path, RTX 4070 training path. "
                    * 160
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Recent answer should remain protected because the user is asking about the "
                    "current state of the chat feature and the Headroom integration. "
                    * 20
                ),
            },
        ],
        config=cfg,
        runtime=runtime,
    )
    print(
        json.dumps(
            {
                "preflight": preflight,
                "init_status": init_status,
                "status": result.get("status"),
                "error_code": result.get("error_code"),
                "error": result.get("error"),
                "message": result.get("message"),
                "runtime": result.get("runtime"),
                "language": result.get("language"),
                "chat_mode": result.get("chat_mode"),
                "seconds": result.get("seconds"),
                "headroom": result.get("headroom"),
                "warnings": result.get("warnings", []),
                "model": result.get("model"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
