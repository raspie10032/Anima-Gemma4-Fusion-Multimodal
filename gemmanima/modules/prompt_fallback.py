from __future__ import annotations

import re


_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_ASCII_TOKEN_RE = re.compile(r"@[A-Za-z0-9_][A-Za-z0-9_.-]*|[A-Za-z][A-Za-z0-9_+-]*")

_BASE_TAGS = ("anime illustration", "high quality", "detailed")
_GENERIC_TAGS = ("1girl", "solo", "character")
_NEGATIVE_PROMPT = "low quality, blurry, bad anatomy, extra fingers, text, watermark"

_KOREAN_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("\uc18c\ub140", "\uc5ec\uc790", "\uce90\ub9ad\ud130", "\uc560\ub2c8"), ("1girl", "solo")),
    (("\uc232", "\uc232\uc18d"), ("forest",)),
    (("\ub79c\ud134", "\ub4f1\ubd88"), ("lantern",)),
    (("\uc11c \uc788", "\uc11c\uc788", "\uc11c \uc788\ub294"), ("standing",)),
    (("\ubc24", "\uc57c\uacbd"), ("night",)),
    (("\ube44", "\uc624\ub294 \ubc24"), ("rain",)),
    (("\uce74\ud398",), ("cafe",)),
    (("\ucc3d\uac00", "\ucc3d"), ("window",)),
    (("\uc549\uc544", "\uc549\uc740", "\uc549\uc544 \uc788"), ("sitting",)),
    (("\ub530\ub73b", "\uc544\ub291"), ("warm lighting", "cozy atmosphere")),
    (("\uc810\ud504", "\ub6f0\uc5b4"), ("jumping", "dynamic pose")),
    (("\uc5ed\ub3d9",), ("dynamic pose",)),
    (("\ud30c\ub780", "\ud30c\ub780\uc0c9"), ("blue jacket",)),
    (("\uc7ac\ud0b7", "\uc790\ucf13"), ("jacket",)),
    (("\uc6b4\ub3d9\ud654",), ("sneakers",)),
    (("\ub9c8\ubc95",), ("magic",)),
    (("\ub3c4\uc11c\uad00",), ("library",)),
    (("\ucc45",), ("book", "open book")),
    (("\ub9c8\ub140",), ("witch", "witch hat")),
    (("\ub208", "\ub0b4\ub9ac\ub294"), ("snow",)),
    (("\uac70\ub9ac",), ("street",)),
    (("\ube68\uac04", "\ube68\uac04 \ubaa9\ub3c4\ub9ac"), ("red scarf",)),
    (("\ubaa9\ub3c4\ub9ac",), ("scarf",)),
    (("\ucd08\uc0c1", "\ucd08\uc0c1\ud654"), ("portrait",)),
    (("\ubd80\ub4dc\ub7ec\uc6b4 \uc0c9\uac10", "\ubd80\ub4dc\ub7ec\uc6b4"), ("soft colors",)),
    (("\uaf43\ubc2d", "\uaf43"), ("flower field", "flowers")),
    (("\uc77c\ub7ec\uc2a4\ud2b8",), ("illustration",)),
)


def build_safe_generation_prompt(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        return ", ".join((*_BASE_TAGS, *_GENERIC_TAGS))
    if not _HANGUL_RE.search(text):
        return text

    tags: list[str] = []
    for token in _ASCII_TOKEN_RE.findall(text):
        _append_unique(tags, token)
    for needles, mapped_tags in _KOREAN_TAG_RULES:
        if any(needle in text for needle in needles):
            for tag in mapped_tags:
                _append_unique(tags, tag)
    if not any(tag in tags for tag in ("1girl", "1boy", "solo", "character")):
        for tag in _GENERIC_TAGS:
            _append_unique(tags, tag)
    for tag in _BASE_TAGS:
        _append_unique(tags, tag)
    return ", ".join(tags)


def enrich_generation_prompt(message: str, prompt: str) -> str:
    base = str(prompt or "").strip()
    fallback = build_safe_generation_prompt(message)
    if not base:
        return fallback
    tags: list[str] = []
    for tag in base.split(","):
        _append_unique(tags, tag)
    for tag in fallback.split(","):
        tag = tag.strip()
        if tag in _BASE_TAGS:
            continue
        if tag in _GENERIC_TAGS and any(existing in tags for existing in _GENERIC_TAGS):
            continue
        _append_unique(tags, tag)
    return ", ".join(tags)


def build_safe_negative_prompt(negative_prompt: str | None = None) -> str:
    text = str(negative_prompt or "").strip()
    return text or _NEGATIVE_PROMPT


def _append_unique(tags: list[str], value: str) -> None:
    tag = value.strip().strip(",")
    if tag and tag.lower() not in {item.lower() for item in tags}:
        tags.append(tag)
