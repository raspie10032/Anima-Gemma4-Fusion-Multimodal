from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemmanima.rendering.gemma_hidden import gemma_hidden_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Gemma hidden provider environment.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = gemma_hidden_environment()
    payload["provider_import"] = True
    payload["ready"] = bool(payload["model_safetensors"] and payload["tokenizer_json"])
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
