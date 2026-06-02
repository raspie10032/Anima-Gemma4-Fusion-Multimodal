from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote


DEFAULT_GEMMANIMA_ADAPTER_REPO = os.environ.get("GEMMANIMA_ADAPTER_REPO", "raspie/gemmanima-adapter-bundle")
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class ModelSource:
    origin: str
    repo_id: str
    filename: str
    url: str
    license_id: str = ""
    license_note: str = ""
    note: str = ""

    def to_json_dict(self) -> dict[str, str]:
        return {
            "origin": self.origin,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "url": self.url,
            "license_id": self.license_id,
            "license_note": self.license_note,
            "note": self.note,
        }


def hf_source(
    *,
    origin: str,
    repo_id: str,
    filename: str,
    license_id: str = "",
    license_note: str = "",
    note: str = "",
) -> ModelSource:
    encoded_filename = "/".join(quote(part) for part in filename.split("/"))
    return ModelSource(
        origin=origin,
        repo_id=repo_id,
        filename=filename,
        url=f"https://huggingface.co/{repo_id}/resolve/main/{encoded_filename}",
        license_id=license_id,
        license_note=license_note,
        note=note,
    )


def adapter_source(filename: str, *, note: str = "") -> ModelSource:
    return hf_source(
        origin="gemmanima_adapter_bundle",
        repo_id=DEFAULT_GEMMANIMA_ADAPTER_REPO,
        filename=filename,
        license_id="other",
        license_note="GemmAnima adapter artifacts do not relicense upstream base models; use is subject to this adapter bundle notice and applicable upstream model licenses.",
        note=note or "GemmAnima-owned adapter/checkpoint artifact.",
    )


def download_hf_source(
    source: ModelSource,
    target: Path,
    *,
    overwrite: bool = False,
    on_progress: ProgressCallback | None = None,
    chunk_size: int = 1024 * 1024,
) -> dict[str, object]:
    target = Path(target)
    if target.exists() and not overwrite:
        size = target.stat().st_size
        if on_progress:
            on_progress(
                {
                    "status": "exists",
                    "path": str(target),
                    "downloaded_bytes": size,
                    "total_bytes": size,
                    "source": source.to_json_dict(),
                }
            )
        return {
            "status": "exists",
            "path": str(target),
            "bytes": size,
            "source": source.to_json_dict(),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".download")
    request = urllib.request.Request(source.url)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response, temp_path.open("wb") as handle:
        total_bytes = _content_length(response)
        downloaded_bytes = 0
        if on_progress:
            on_progress(
                {
                    "status": "downloading",
                    "path": str(target),
                    "downloaded_bytes": downloaded_bytes,
                    "total_bytes": total_bytes,
                    "source": source.to_json_dict(),
                }
            )
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded_bytes += len(chunk)
            if on_progress:
                on_progress(
                    {
                        "status": "downloading",
                        "path": str(target),
                        "downloaded_bytes": downloaded_bytes,
                        "total_bytes": total_bytes,
                        "source": source.to_json_dict(),
                    }
                )
    temp_path.replace(target)
    final_size = target.stat().st_size
    if on_progress:
        on_progress(
            {
                "status": "downloaded",
                "path": str(target),
                "downloaded_bytes": final_size,
                "total_bytes": final_size,
                "source": source.to_json_dict(),
            }
        )
    return {
        "status": "downloaded",
        "path": str(target),
        "bytes": final_size,
        "source": source.to_json_dict(),
    }


def _content_length(response) -> int:
    try:
        value = response.headers.get("Content-Length")
    except AttributeError:
        return 0
    if not value:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def write_download_manifest(results: list[dict[str, object]], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"assets": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
