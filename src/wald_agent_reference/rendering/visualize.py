from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ..core.config import AppSettings
from ..core.models import CalculationResult, VisualizationResult
from ..utils import slugify


class VisualizationEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.output_dir = settings.plots_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_visualize(self, question: str, calculation: CalculationResult | None) -> bool:
        lowered = question.lower()
        requested = any(term in lowered for term in ["graph", "plot", "chart", "visual", "trend", "compare", "comparison"])
        return requested and calculation is not None and calculation.chart_data is not None

    def create(self, question: str, calculation: CalculationResult, suffix: str | None = None) -> VisualizationResult | None:
        chart_data = calculation.chart_data
        if not chart_data:
            return None

        chart_type = chart_data["type"]
        labels = chart_data["labels"]
        values = chart_data["values"]
        title = chart_data.get("title", "Leadership insight chart")
        filename_stem = slugify(question if not suffix else f"{question} {suffix}")
        filename = self.output_dir / f"{filename_stem}.png"

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=self.settings.plot_dpi)
        if chart_type == "line":
            ax.plot(labels, values, marker="o", linewidth=2, color="#0b7285")
        else:
            ax.bar(labels, values, color="#f08c00")
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.set_xlabel("Category")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        
        # Save to file
        fig.savefig(filename)
        
        # Also generate base64-encoded image for embedding in response
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

        return VisualizationResult(
            path=filename,
            caption=f"{title} chart saved to {filename}.",
            chart_type=chart_type,
            base64_image=base64_image,
        )
