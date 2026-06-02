from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_prototype import default_model_set
from generator import BuiltinImageConfig, BuiltinImageGenerator, build_generation_job, build_generation_job_with_planner_tags
from planner import run_tipo_planner
from runtime import chat, create_image_job, dumps, health, route_request, tag_image


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, health())
            return
        self._json(404, {"status": "failed", "error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/route":
            self._json(404, {"status": "failed", "error": "not found"})
            return
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._json(400, {"status": "failed", "error": "invalid json"})
            return
        result = route_request(payload)
        self._json(400 if result.get("status") == "failed" else 200, result)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone chat/tag runtime prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("message")
    chat_parser.add_argument("--json", action="store_true")

    tag_parser = subparsers.add_parser("tag")
    tag_parser.add_argument("image_path")
    tag_parser.add_argument("--prompt", default=None)
    tag_parser.add_argument("--json", action="store_true")

    image_parser = subparsers.add_parser("image")
    image_parser.add_argument("message")
    image_parser.add_argument("--style", default="anime illustration, clean line art, detailed lighting")
    image_parser.add_argument("--negative-prompt", default=None)
    image_parser.add_argument("--use-planner", action="store_true")
    image_parser.add_argument("--generate", action="store_true", help="Run the Anima/GEMMANIMA builtin generator slot.")
    image_parser.add_argument("--json", action="store_true")

    route_parser = subparsers.add_parser("route")
    route_parser.add_argument("--task", default="auto")
    route_parser.add_argument("--message", default="")
    route_parser.add_argument("--image", default="")
    route_parser.add_argument("--style", default="")
    route_parser.add_argument("--negative-prompt", default="")
    route_parser.add_argument("--json", action="store_true")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8787)

    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--json", action="store_true")

    models_parser = subparsers.add_parser("models")
    models_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "chat":
        return _print(chat(message=args.message), args.json)
    if args.command == "tag":
        return _print(tag_image(image_path=args.image_path, prompt=args.prompt), args.json)
    if args.command == "image":
        payload = {"message": args.message, "style": args.style}
        if args.negative_prompt:
            payload["negative_prompt"] = args.negative_prompt
        if args.generate:
            config = BuiltinImageConfig(
                style=args.style,
                negative_prompt=args.negative_prompt or BuiltinImageConfig.negative_prompt,
            )
            planner_result = None
            if args.use_planner:
                planner_result = run_tipo_planner(message=args.message)
                if planner_result.get("status") != "completed":
                    return _print(planner_result, args.json)
                job = build_generation_job_with_planner_tags(
                    message=args.message,
                    planner_tags=list(planner_result.get("tags") or []),
                    config=config,
                )
            else:
                job = build_generation_job(message=args.message, config=config)
            result = BuiltinImageGenerator().generate(job)
            if planner_result is not None:
                result["planner"] = {
                    "status": planner_result.get("status"),
                    "tags": planner_result.get("tags"),
                    "seconds": planner_result.get("seconds"),
                }
            return _print(result, args.json)
        return _print(create_image_job(**payload), args.json)
    if args.command == "route":
        return _print(route_request({
            "task": args.task,
            "message": args.message,
            "image_path": args.image,
            "style": args.style,
            "negative_prompt": args.negative_prompt,
        }), args.json)
    if args.command == "health":
        return _print(health(), args.json)
    if args.command == "models":
        payload = default_model_set().to_json_dict()
        if args.json:
            print(dumps(payload))
        else:
            print(dumps(payload))
        return 0
    if args.command == "serve":
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"chat/tag prototype listening on http://{args.host}:{args.port}")
        server.serve_forever()
        return 0
    return 2


def _print(payload: dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(dumps(payload))
    else:
        print(payload.get("message") or payload.get("error") or dumps(payload))
    return 0 if payload.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
