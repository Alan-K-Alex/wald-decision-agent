from __future__ import annotations

import re
import hashlib
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z0-9_.%-]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def slugify(value: str, limit: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value[:limit] or "artifact"


def coerce_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def path_label(path: Path) -> str:
    return path.name


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
