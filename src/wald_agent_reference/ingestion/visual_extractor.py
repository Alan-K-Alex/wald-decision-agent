from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.config import AppSettings
from ..core.models import VisualArtifact
from ..utils import compact_whitespace, sha256_file, slugify


class VisualExtractor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def parse_file(self, path: Path) -> VisualArtifact | None:
        suffix = path.suffix.lower()
        if suffix == ".svg":
            return self._parse_svg(path)

        artifact = self._parse_with_gemini(path)
        if artifact is not None:
            return artifact
        return None

    def _parse_svg(self, path: Path) -> VisualArtifact | None:
        text = path.read_text(encoding="utf-8")
        labels = re.findall(r"<text[^>]*>(.*?)</text>", text, flags=re.IGNORECASE | re.DOTALL)
        extracted_text = compact_whitespace(" ".join(re.sub(r"<[^>]+>", " ", label) for label in labels))
        title_match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        title = compact_whitespace(title_match.group(1)) if title_match else path.stem
        if not extracted_text:
            return None
        return VisualArtifact(
            artifact_id=f"{slugify(path.stem)}:{sha256_file(path)[:8]}",
            source_path=path,
            source_type="svg",
            extracted_text=extracted_text,
            summary=f"Visual artifact '{title}' contains extracted labels: {extracted_text}.",
            metadata={"title": title, "extraction_backend": "svg-text"},
        )

    def _parse_with_gemini(self, path: Path) -> VisualArtifact | None:
        if not self.settings.gemini_api_key:
            return None
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return None

        prompt = (
            "You are extracting factual information from a chart, graph, or business visual. "
            "Return strict JSON with keys: title, visual_type, extracted_text, key_insights, axes, series. "
            "Preserve all numeric values you can read. Do not invent data."
        )
        try:
            client = genai.Client(api_key=self.settings.gemini_api_key)
            response = client.models.generate_content(
                model=self.settings.vision_model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=self._mime_type(path)),
                ],
            )
            data = json.loads(response.text)
            extracted_text = compact_whitespace(data.get("extracted_text", ""))
            insights = data.get("key_insights", [])
            summary = compact_whitespace(
                f"{data.get('title', path.stem)}. "
                f"Type: {data.get('visual_type', 'visual')}. "
                f"Insights: {' '.join(insights) if isinstance(insights, list) else insights}"
            )
            return VisualArtifact(
                artifact_id=f"{slugify(path.stem)}:{sha256_file(path)[:8]}",
                source_path=path,
                source_type="image",
                extracted_text=extracted_text,
                summary=summary,
                metadata={
                    "title": data.get("title", path.stem),
                    "visual_type": data.get("visual_type", "visual"),
                    "axes": data.get("axes", {}),
                    "series": data.get("series", []),
                    "extraction_backend": "gemini-vision",
                },
            )
        except Exception:
            return None

    @staticmethod
    def _mime_type(path: Path) -> str:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }.get(path.suffix.lower(), "application/octet-stream")
