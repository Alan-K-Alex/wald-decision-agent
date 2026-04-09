from __future__ import annotations

from dataclasses import dataclass

from ..core.config import AppSettings


class MemoryBackend:
    """Base class for memory backends. Historically supported Supermemory, now simplified."""
    def sync_document(self, document: object) -> None:
        pass

    def sync_table(self, table: object) -> None:
        pass

    def sync_visual(self, visual: object) -> None:
        pass

    def sync_chunk(self, chunk: object) -> None:
        pass

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        return []

    def delete_container(self) -> None:
        pass


class NullMemoryBackend(MemoryBackend):
    pass


def build_memory_backend(settings: AppSettings) -> MemoryBackend:
    """Returns a NullMemoryBackend since external memory backends are deprecated."""
    return NullMemoryBackend()
