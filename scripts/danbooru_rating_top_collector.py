"""Collect and download high-score Danbooru posts by rating.

The credential file is read at runtime and is never copied into reports.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RATING_ALIASES = {
    "g": "g",
    "general": "g",
    "s": "s",
    "sensitive": "s",
    "q": "q",
    "questionable": "q",
    "e": "e",
    "explicit": "e",
}

RATING_NAMES = {
    "g": "general",
    "s": "sensitive",
    "q": "questionable",
    "e": "explicit",
}

DEFAULT_BLOCKED_TAGS = {
    "age_difference",
    "aged_down",
    "child",
    "children",
    "elementary_school",
    "elementary_schooler",
    "elementary_school_uniform",
    "kindergarten",
    "kindergarten_uniform",
    "loli",
    "lolicon",
    "middle_school",
    "middle_schooler",
    "minor",
    "preschool",
    "preschooler",
    "school_swimsuit",
    "shota",
    "shotacon",
    "toddler",
    "underage",
    "young",
    "younger",
}

DEFAULT_REJECT_TAGS = {
    "ai-generated",
    "ai_assisted",
    "ai_art",
    "animated",
    "flash",
    "video",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CloudflareChallengeError(RuntimeError):
    """Raised when Danbooru returns an interactive browser challenge."""


@dataclass(frozen=True)
class Credentials:
    username: str
    api_key: str


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def parse_credentials(path: Path) -> Credentials:
    text = path.read_text(encoding="utf-8-sig")
    username = ""
    for line in text.splitlines():
        match = re.match(r"\s*id\s*:\s*(\S+)\s*$", line, flags=re.IGNORECASE)
        if match:
            username = match.group(1)
            break
    if not username:
        raise ValueError(f"Could not find `id:` in credential file: {path}")

    api_key = ""
    for line in text.splitlines():
        if "hf_" in line:
            continue
        tokens = re.findall(r"\b[A-Za-z0-9]{20,}\b", line)
        if tokens:
            api_key = max(tokens, key=len)
            break
    if not api_key:
        raise ValueError(f"Could not find a Danbooru API key in credential file: {path}")
    return Credentials(username=username, api_key=api_key)


def auth_header(credentials: Credentials) -> str:
    raw = f"{credentials.username}:{credentials.api_key}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def normalize_ratings(raw: Iterable[str]) -> list[str]:
    ratings: list[str] = []
    for item in raw:
        key = item.lower().strip()
        if key not in RATING_ALIASES:
            raise ValueError(f"Unsupported rating {item!r}; use g/s/q/e")
        rating = RATING_ALIASES[key]
        if rating not in ratings:
            ratings.append(rating)
    return ratings


def post_tags(post: dict) -> set[str]:
    tags: set[str] = set()
    for key, value in post.items():
        if key.startswith("tag_string") and isinstance(value, str):
            tags.update(tag for tag in value.split() if tag)
    return tags


def is_rejected(post: dict, blocked_tags: set[str], reject_tags: set[str]) -> tuple[bool, str]:
    tags = post_tags(post)
    blocked = sorted(tags & blocked_tags)
    if blocked:
        return True, "blocked_age_tag:" + ",".join(blocked[:5])
    rejected = sorted(tags & reject_tags)
    if rejected:
        return True, "reject_tag:" + ",".join(rejected[:5])
    if not usable_image_url(post):
        return True, "missing_image_url"
    return False, ""


def usable_image_url(post: dict) -> str:
    for key in ("file_url", "large_file_url"):
        value = post.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return ""


def image_extension(post: dict, url: str) -> str:
    ext = str(post.get("file_ext") or "").strip().lower()
    if ext:
        ext = "." + ext.lstrip(".")
    if ext not in IMAGE_EXTENSIONS:
        parsed = urllib.parse.urlparse(url)
        ext = Path(parsed.path).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    return ext


def slim_post(post: dict, rating: str) -> dict:
    url = usable_image_url(post)
    return {
        "id": post.get("id"),
        "rating": rating,
        "rating_name": RATING_NAMES[rating],
        "score": post.get("score"),
        "fav_count": post.get("fav_count"),
        "up_score": post.get("up_score"),
        "down_score": post.get("down_score"),
        "file_ext": post.get("file_ext"),
        "file_size": post.get("file_size"),
        "image_width": post.get("image_width"),
        "image_height": post.get("image_height"),
        "md5": post.get("md5"),
        "created_at": post.get("created_at"),
        "source": post.get("source"),
        "tag_string_general": post.get("tag_string_general", ""),
        "tag_string_character": post.get("tag_string_character", ""),
        "tag_string_copyright": post.get("tag_string_copyright", ""),
        "tag_string_artist": post.get("tag_string_artist", ""),
        "tag_string_meta": post.get("tag_string_meta", ""),
        "file_url": url,
    }


def request_json(url: str, headers: dict[str, str], attempts: int = 6, timeout: float = 60.0) -> list[dict]:
    last_error: Exception | None = None
    attempt = 0
    while attempts <= 0 or attempt < attempts:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read()
            text = body.decode("utf-8")
            lowered = text[:4096].lower()
            if "just a moment" in lowered and "cloudflare" in lowered:
                raise CloudflareChallengeError(
                    "Danbooru returned a Cloudflare browser challenge instead of JSON. "
                    "Open Danbooru manually in a normal browser or change VPN route, then retry."
                )
            return json.loads(text)
        except CloudflareChallengeError:
            raise
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504}:
                delay = min(120, 3 * (2**min(attempt, 6)))
                print(
                    json.dumps({"phase": "retry", "error": f"http_{exc.code}", "delay_sec": delay}, sort_keys=True),
                    flush=True,
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            delay = min(120, 3 * (2**min(attempt, 6)))
            print(
                json.dumps({"phase": "retry", "error": type(exc).__name__, "delay_sec": delay}, sort_keys=True),
                flush=True,
            )
            time.sleep(delay)
            attempt += 1
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


def load_jsonl_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            post_id = value.get("id") or value.get("post_id")
            if isinstance(post_id, int):
                ids.add(post_id)
    return ids


def scan_existing_image_ids(roots: Iterable[Path]) -> set[int]:
    ids: set[int] = set()
    for root in roots:
        image_root = root / "images"
        if not image_root.exists():
            continue
        for path in image_root.rglob("*"):
            if not path.is_file():
                continue
            match = re.match(r"^(\d+)", path.stem)
            if match:
                ids.add(int(match.group(1)))
    return ids


def load_skip_manifest_ids(paths: Iterable[Path]) -> set[int]:
    ids: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("source_id", "id", "post_id"):
                    post_id = value.get(key)
                    if isinstance(post_id, int):
                        ids.add(post_id)
                        break
                else:
                    image = value.get("image")
                    if isinstance(image, str):
                        match = re.match(r"^(\d+)", Path(image).stem)
                        if match:
                            ids.add(int(match.group(1)))
    return ids


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def collect_rating(
    *,
    base_url: str,
    headers: dict[str, str],
    output_root: Path,
    rating: str,
    target_count: int,
    existing_ids: set[int],
    blocked_tags: set[str],
    reject_tags: set[str],
    api_sleep: float,
    page_start: int,
    max_pages: int,
    limit: int,
    request_attempts: int,
    request_timeout: float,
) -> dict:
    manifest_path = output_root / "manifests" / f"rating_{rating}.jsonl"
    reject_path = output_root / "manifests" / f"rating_{rating}_rejected.jsonl"
    accepted_ids = load_jsonl_ids(manifest_path)
    seen_ids = set(existing_ids) | accepted_ids
    accepted_total = len(accepted_ids)
    limiter = RateLimiter(api_sleep)
    pages_read = 0
    rejected_total = 0
    duplicate_total = 0

    for page in range(page_start, page_start + max_pages):
        if accepted_total >= target_count:
            break
        tags = f"rating:{rating} order:score"
        params = urllib.parse.urlencode({"tags": tags, "limit": limit, "page": page})
        url = f"{base_url.rstrip('/')}/posts.json?{params}"
        limiter.wait()
        posts = request_json(url, headers, attempts=request_attempts, timeout=request_timeout)
        pages_read += 1
        if not posts:
            break

        accepted_batch: list[dict] = []
        rejected_batch: list[dict] = []
        for post in posts:
            post_id = post.get("id")
            if not isinstance(post_id, int):
                continue
            if post_id in seen_ids:
                duplicate_total += 1
                continue
            rejected, reason = is_rejected(post, blocked_tags, reject_tags)
            if rejected:
                rejected_batch.append(
                    {
                        "id": post_id,
                        "rating": rating,
                        "score": post.get("score"),
                        "reason": reason,
                    }
                )
                seen_ids.add(post_id)
                rejected_total += 1
                continue
            accepted_batch.append(slim_post(post, rating))
            seen_ids.add(post_id)
            accepted_total += 1
            if accepted_total >= target_count:
                break

        if accepted_batch:
            write_jsonl(manifest_path, accepted_batch)
        if rejected_batch:
            write_jsonl(reject_path, rejected_batch)

        if page % 10 == 0 or accepted_total >= target_count:
            print(
                json.dumps(
                    {
                        "phase": "collect",
                        "rating": rating,
                        "page": page,
                        "accepted": accepted_total,
                        "target": target_count,
                        "rejected": rejected_total,
                        "duplicates": duplicate_total,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return {
        "rating": rating,
        "accepted": accepted_total,
        "target": target_count,
        "pages_read": pages_read,
        "rejected": rejected_total,
        "duplicates": duplicate_total,
        "manifest": str(manifest_path),
    }


def image_path(output_root: Path, post: dict) -> Path:
    post_id = int(post["id"])
    rating = str(post["rating"])
    url = str(post["file_url"])
    ext = image_extension(post, url)
    shard = f"{post_id % 1000:03d}"
    return output_root / "images" / rating / shard / f"{post_id}{ext}"


def download_one(post: dict, output_root: Path, headers: dict[str, str], min_bytes: int) -> dict:
    post_id = int(post["id"])
    url = str(post["file_url"])
    dest = image_path(output_root, post)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return {"id": post_id, "status": "exists", "path": str(dest), "bytes": dest.stat().st_size}

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=str(dest.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    sha256 = hashlib.sha256()
    bytes_written = 0
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                sha256.update(chunk)
                bytes_written += len(chunk)
        if bytes_written < min_bytes:
            tmp_path.unlink(missing_ok=True)
            return {"id": post_id, "status": "too_small", "bytes": bytes_written}
        tmp_path.replace(dest)
        return {
            "id": post_id,
            "status": "downloaded",
            "path": str(dest),
            "bytes": bytes_written,
            "sha256": sha256.hexdigest(),
        }
    except Exception as exc:  # noqa: BLE001 - long-running downloader should record and continue.
        tmp_path.unlink(missing_ok=True)
        return {"id": post_id, "status": "error", "error": repr(exc)}


def iter_manifest_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def download_rating(
    *,
    output_root: Path,
    rating: str,
    workers: int,
    download_sleep: float,
    min_bytes: int,
    user_agent: str,
) -> dict:
    manifest_path = output_root / "manifests" / f"rating_{rating}.jsonl"
    log_path = output_root / "manifests" / f"rating_{rating}_downloaded.jsonl"
    done_ids = load_jsonl_ids(log_path)
    records = [record for record in iter_manifest_records(manifest_path) if int(record["id"]) not in done_ids]
    limiter = RateLimiter(download_sleep)
    headers = {"User-Agent": user_agent}
    counts: dict[str, int] = {}
    completed = 0

    def task(record: dict) -> dict:
        limiter.wait()
        return download_one(record, output_root, headers, min_bytes)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_record = {executor.submit(task, record): record for record in records}
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            for future in concurrent.futures.as_completed(future_to_record):
                result = future.result()
                result["rating"] = rating
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                status = str(result.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
                completed += 1
                if completed % 100 == 0 or completed == len(records):
                    print(
                        json.dumps(
                            {
                                "phase": "download",
                                "rating": rating,
                                "completed": completed,
                                "queued": len(records),
                                "counts": counts,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    return {"rating": rating, "queued": len(records), "counts": counts, "log": str(log_path)}


def write_summary(output_root: Path, summary: dict) -> None:
    path = output_root / "collection_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\Projects\danbooru_rating_top_200k"))
    parser.add_argument("--base-url", default="https://danbooru.donmai.us")
    parser.add_argument("--ratings", nargs="+", default=["g", "s", "q", "e"])
    parser.add_argument("--target-per-rating", type=int, default=200_000)
    parser.add_argument("--skip-root", type=Path, action="append", default=[])
    parser.add_argument("--skip-manifest", type=Path, action="append", default=[])
    parser.add_argument("--api-sleep", type=float, default=0.5)
    parser.add_argument("--download-sleep", type=float, default=0.1)
    parser.add_argument("--image-workers", type=int, default=3)
    parser.add_argument("--page-start", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--request-attempts", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--min-bytes", type=int, default=4096)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--allow-ai", action="store_true")
    parser.add_argument("--allow-age-risk-tags", action="store_true")
    parser.add_argument("--user-agent", default="gemmanima-danbooru-rating-top-collector/0.1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ratings = normalize_ratings(args.ratings)
    args.output_root.mkdir(parents=True, exist_ok=True)
    credentials = parse_credentials(args.credential_file)
    headers = {
        "Authorization": auth_header(credentials),
        "User-Agent": args.user_agent,
    }
    blocked_tags = set() if args.allow_age_risk_tags else set(DEFAULT_BLOCKED_TAGS)
    reject_tags = set() if args.allow_ai else set(DEFAULT_REJECT_TAGS)

    existing_ids = load_skip_manifest_ids(args.skip_manifest)
    existing_ids.update(scan_existing_image_ids(args.skip_root))
    summary: dict = {
        "output_root": str(args.output_root),
        "ratings": ratings,
        "target_per_rating": args.target_per_rating,
        "skip_roots": [str(path) for path in args.skip_root],
        "skip_manifests": [str(path) for path in args.skip_manifest],
        "existing_ids_scanned": len(existing_ids),
        "blocked_tag_filter_enabled": bool(blocked_tags),
        "ai_reject_filter_enabled": bool(reject_tags),
        "collect": [],
        "download": [],
    }

    if not args.download_only:
        for rating in ratings:
            summary["collect"].append(
                collect_rating(
                    base_url=args.base_url,
                    headers=headers,
                    output_root=args.output_root,
                    rating=rating,
                    target_count=args.target_per_rating,
                    existing_ids=existing_ids,
                    blocked_tags=blocked_tags,
                    reject_tags=reject_tags,
                    api_sleep=args.api_sleep,
                    page_start=args.page_start,
                    max_pages=args.max_pages,
                    limit=args.limit,
                    request_attempts=args.request_attempts,
                    request_timeout=args.request_timeout,
                )
            )
            write_summary(args.output_root, summary)

    if not args.metadata_only:
        for rating in ratings:
            summary["download"].append(
                download_rating(
                    output_root=args.output_root,
                    rating=rating,
                    workers=args.image_workers,
                    download_sleep=args.download_sleep,
                    min_bytes=args.min_bytes,
                    user_agent=args.user_agent,
                )
            )
            write_summary(args.output_root, summary)

    write_summary(args.output_root, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
