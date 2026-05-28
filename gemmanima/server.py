from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from gemmanima.api import handle_chat_payload, handle_health_payload
from gemmanima.ui import GUI_HTML


class GemmAnimaRequestHandler(BaseHTTPRequestHandler):
    base_dir = Path("runs")

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(200, GUI_HTML)
            return
        if self.path == "/v1/health":
            self._send_json(200, handle_health_payload())
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            result = handle_chat_payload(payload, base_dir=self.base_dir)
            status = 400 if "error" in result else 200
            self._send_json(status, result)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GemmAnima HTTP backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    GemmAnimaRequestHandler.base_dir = Path(args.base_dir)
    server = ThreadingHTTPServer((args.host, args.port), GemmAnimaRequestHandler)
    print(f"GemmAnima backend listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
