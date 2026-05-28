from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemmanima.rendering.comfy_bootstrap import bootstrap_comfy
from gemmanima.rendering.t5_tokenizer import build_t5_tokenizer_provider, t5_tokenizer_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Anima T5XXL tokenizer provider.")
    parser.add_argument("--prompt", default="masterpiece, best quality, bright forest")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    bootstrap_comfy()
    payload = t5_tokenizer_environment(load_tokenizer=True)
    if payload.get("tokenizer_loaded"):
        provider = build_t5_tokenizer_provider()
        ids, weights = provider.encode_ids_weights(args.prompt)
        payload.update(
            {
                "prompt": args.prompt,
                "ids_shape": list(ids.shape),
                "weights_shape": list(weights.shape),
                "ids_dtype": str(ids.dtype),
                "weights_dtype": str(weights.dtype),
                "ready": ids.numel() > 0 and ids.shape == weights.shape,
            }
        )
    else:
        payload["ready"] = False

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
