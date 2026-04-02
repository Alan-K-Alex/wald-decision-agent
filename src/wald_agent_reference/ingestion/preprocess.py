from __future__ import annotations

from ..utils import compact_whitespace


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = compact_whitespace(text)
    if len(cleaned) <= chunk_size:
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    cleaned = compact_whitespace(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [(cleaned, 0, len(cleaned))]

    chunks: list[tuple[str, int, int]] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(cleaned), step):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end]
        if chunk:
            chunks.append((chunk, start, end))
    return chunks
