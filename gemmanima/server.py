from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from gemmanima.api import handle_chat_payload, handle_health_payload
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.modules.tipo_runtime import initialize_tipo_text_runtime
from gemmanima.ui import GUI_HTML


DEFAULT_CHATBOT_NAME = "GemmAnima"


def user_settings_path(app_root: str | Path | None = None) -> Path:
    root = Path(app_root) if app_root is not None else Path.cwd()
    return root / "settings" / "user_settings.json"


def sanitize_chatbot_name(value: Any) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    return (cleaned[:32] or DEFAULT_CHATBOT_NAME)


def load_user_settings(app_root: str | Path | None = None) -> dict[str, Any]:
    path = user_settings_path(app_root)
    payload: dict[str, Any] = {}
    configured = path.is_file()
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["chatbot_name"] = sanitize_chatbot_name(payload.get("chatbot_name"))
    payload["configured"] = configured
    return payload


def save_user_settings(app_root: str | Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    current = load_user_settings(app_root)
    if "chatbot_name" in payload:
        current["chatbot_name"] = sanitize_chatbot_name(payload.get("chatbot_name"))
    current["configured"] = True
    path = user_settings_path(app_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current


def resolve_image_artifact(base_dir: str | Path, raw_path: str) -> Path | None:
    normalized = unquote(raw_path).replace("\\", "/")
    rel = PurePosixPath(normalized)
    if rel.is_absolute() or not rel.parts:
        return None
    if any(part in {"", ".", ".."} or ":" in part for part in rel.parts):
        return None
    root = (Path(base_dir) / "images").resolve()
    target = (root / Path(*rel.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def save_upload_data_url(base_dir: str | Path, file_name: str, data_url: str) -> Path:
    if "," not in data_url:
        raise ValueError("invalid upload data")
    meta, encoded = data_url.split(",", 1)
    if not meta.startswith("data:image/"):
        raise ValueError("only image uploads are supported")
    suffix = Path(file_name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    if suffix not in allowed:
        mime = meta.split(";", 1)[0].removeprefix("data:image/")
        suffix = ".jpg" if mime == "jpeg" else f".{mime}"
    if suffix not in allowed:
        raise ValueError("unsupported image extension")
    raw = base64.b64decode(encoded, validate=True)
    if not raw:
        raise ValueError("empty upload")
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("upload is too large")
    target_dir = (Path(base_dir) / "uploads").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / f"{uuid4().hex}{suffix}").resolve()
    target.write_bytes(raw)
    return target


class ModelDownloadManager:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = self._initial_state()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "status": "idle",
            "started_at": None,
            "updated_at": None,
            "completed_at": None,
            "total_assets": len(self.registry.assets()),
            "completed_assets": 0,
            "current_asset": "",
            "current_path": "",
            "current_downloaded_bytes": 0,
            "current_total_bytes": 0,
            "assets": {},
            "error": "",
        }

    def start(self, *, overwrite: bool = False, names: set[str] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status()
            self._state = self._initial_state()
            self._state.update(
                {
                    "status": "running",
                    "started_at": time.time(),
                    "updated_at": time.time(),
                    "overwrite": overwrite,
                }
            )
            self._thread = threading.Thread(
                target=self._run_download,
                kwargs={"overwrite": overwrite, "names": names},
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._state)
            payload["assets"] = dict(self._state.get("assets", {}))
            payload["running"] = bool(self._thread and self._thread.is_alive())
            return payload

    def _update(self, event: dict[str, Any]) -> None:
        with self._lock:
            assets = dict(self._state.get("assets", {}))
            name = str(event.get("name") or "")
            if name:
                previous = dict(assets.get(name, {}))
                previous.update(event)
                assets[name] = previous
                self._state["current_asset"] = name
                self._state["current_path"] = str(event.get("path") or previous.get("path") or "")
                self._state["current_downloaded_bytes"] = int(event.get("downloaded_bytes") or 0)
                self._state["current_total_bytes"] = int(event.get("total_bytes") or 0)
                if event.get("status") in {"downloaded", "exists"}:
                    self._state["completed_assets"] = len(
                        [item for item in assets.values() if item.get("status") in {"downloaded", "exists"}]
                    )
            if "total_assets" in event:
                self._state["total_assets"] = int(event["total_assets"])
            if "completed_assets" in event:
                self._state["completed_assets"] = int(event["completed_assets"])
            if "status" in event and event["status"] in {"running", "completed", "failed"}:
                self._state["status"] = str(event["status"])
            self._state["assets"] = assets
            self._state["updated_at"] = time.time()

    def _run_download(self, overwrite: bool = False, names: set[str] | None = None) -> dict[str, Any]:
        try:
            result = self.registry.ensure_assets(overwrite=overwrite, names=names, on_progress=self._update)
            with self._lock:
                self._state["status"] = "completed"
                self._state["completed_at"] = time.time()
                self._state["updated_at"] = time.time()
                self._state["result"] = result
            return result
        except Exception as exc:  # pragma: no cover - defensive server boundary
            with self._lock:
                self._state["status"] = "failed"
                self._state["error"] = str(exc)
                self._state["completed_at"] = time.time()
                self._state["updated_at"] = time.time()
            return {"status": "failed", "error": str(exc)}


MODEL_DOWNLOAD_MANAGER = ModelDownloadManager()


class GemmAnimaRequestHandler(BaseHTTPRequestHandler):
    base_dir = Path("runs")
    app_root = Path.cwd()
    download_manager = MODEL_DOWNLOAD_MANAGER

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(200, GUI_HTML)
            return
        if parsed.path == "/v1/health":
            self._send_json(200, handle_health_payload())
            return
        if parsed.path == "/v1/models/download/status":
            self._send_json(200, self.download_manager.status())
            return
        if parsed.path == "/v1/settings/chatbot-name":
            self._send_json(200, load_user_settings(self.app_root))
            return
        if parsed.path.startswith("/artifacts/images/"):
            raw_path = parsed.path.removeprefix("/artifacts/images/")
            target = resolve_image_artifact(self.base_dir, raw_path)
            if target is None or not target.is_file():
                self._send_json(404, {"error": "image artifact not found"})
                return
            self._send_file(target)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/v1/settings/chatbot-name":
            try:
                payload = self._read_json()
                self._send_json(200, save_user_settings(self.app_root, payload))
            except json.JSONDecodeError:
                self._send_json(400, {"status": "failed", "error": "invalid json"})
            return
        if self.path == "/v1/models/download":
            try:
                payload = self._read_json()
                names = payload.get("names")
                selected_names = {str(item) for item in names} if isinstance(names, list) else None
                status = self.download_manager.start(
                    overwrite=bool(payload.get("overwrite")),
                    names=selected_names,
                )
                self._send_json(200, status)
            except json.JSONDecodeError:
                self._send_json(400, {"status": "failed", "error": "invalid json"})
            return
        if self.path == "/v1/uploads":
            try:
                payload = self._read_json()
                target = save_upload_data_url(
                    self.base_dir,
                    str(payload.get("file_name") or "upload.png"),
                    str(payload.get("data_url") or ""),
                )
                self._send_json(200, {"status": "completed", "path": str(target)})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"status": "failed", "error": str(exc)})
            return
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

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def initialize_server_runtime() -> dict[str, Any]:
    return {"tipo_text": initialize_tipo_text_runtime()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GemmAnima HTTP backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    GemmAnimaRequestHandler.base_dir = Path(args.base_dir)
    GemmAnimaRequestHandler.app_root = Path.cwd()
    init_status = initialize_server_runtime()
    text_status = init_status.get("tipo_text", {})
    if text_status.get("status") == "completed":
        print(f"Gemma text runtime initialized: {text_status.get('model')}")
    else:
        print(f"Gemma text runtime initialization failed: {text_status.get('error')}")
    server = ThreadingHTTPServer((args.host, args.port), GemmAnimaRequestHandler)
    print(f"GemmAnima backend listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
