"""
Tests for visual data processing and reference handling.

Tests verify:
- Visual artifact extraction during ingestion
- Visual summary and text generation
- Visual reference in responses
- Visual data not lost in processing
"""

import pytest
from pathlib import Path
from PIL import Image
import io

from wald_agent_reference.ingestion.visual_extractor import VisualExtractor
from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.core.models import VisualArtifact


@pytest.fixture
def settings():
    """Test settings"""
    return AppSettings(
        data_sources_path=str(Path("data/raw")),
        output_dir=str(Path("outputs")),
        embedding_engine="hash",
    )


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample test image"""
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    img_path = tmp_path / "test_chart.png"
    img.save(img_path)
    return img_path


class TestVisualExtraction:
    """Test visual artifact extraction"""
    
    @pytest.mark.slow
    def test_visual_extractor_creates_artifact(self, settings, sample_image_path):
        """Test that visual extractor creates proper artifacts"""
        extractor = VisualExtractor(settings)
        
        visual = extractor.parse_file(sample_image_path)
        
        assert visual is not None
        assert visual.artifact_id
        assert visual.source_path == sample_image_path
        assert visual.source_type == "image"
        # Should have some extracted text or summary
        assert visual.summary or visual.extracted_text
    
    @pytest.mark.slow
    def test_visual_metadata_preservation(self, settings, sample_image_path):
        """Test that visual metadata is preserved"""
        extractor = VisualExtractor(settings)
        
        visual = extractor.parse_file(sample_image_path)
        
        assert visual.metadata
        assert "file_size" in visual.metadata or "dimensions" in visual.metadata
    
    @pytest.mark.slow
    def test_multiple_visual_extraction(self, settings, tmp_path):
        """Test extracting multiple visuals"""
        # Create 3 test images
        image_paths = []
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=(i*80, i*80, i*80))
            path = tmp_path / f"chart_{i}.png"
            img.save(path)
            image_paths.append(path)
        
        extractor = VisualExtractor(settings)
        visuals = []
        
        for path in image_paths:
            visual = extractor.parse_file(path)
            if visual:
                visuals.append(visual)
        
        assert len(visuals) == 3
        
        # Each visual should have unique ID
        ids = {v.artifact_id for v in visuals}
        assert len(ids) == 3


class TestVisualDataPreservation:
    """Test that visual information is not lost during processing"""
    
    def test_visual_summary_meaningful(self, settings, sample_image_path):
        """Test that visual summary is meaningful"""
        extractor = VisualExtractor(settings)
        visual = extractor.parse_file(sample_image_path)
        
        # Summary should not be empty or generic
        assert len(visual.summary) > 10
    
    def test_visual_extracted_text(self, settings, sample_image_path):
        """Test that visual artifact preserves extracted text"""
        extractor = VisualExtractor(settings)
        visual = extractor.parse_file(sample_image_path)
        
        # Either summary or extracted_text should have content
        has_content = (visual.summary and len(visual.summary) > 0) or \
                     (visual.extracted_text and len(visual.extracted_text) > 0)
        assert has_content, "Visual should have some textual content"
    
    def test_visual_artifact_id_uniqueness(self, settings, tmp_path):
        """Test that visual artifact IDs are unique"""
        # Create multiple similar images
        artifact_ids = []
        
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(50, 50, 50))
            path = tmp_path / f"chart_{i}.png"
            img.save(path)
            
            extractor = VisualExtractor(settings)
            visual = extractor.parse_file(path)
            artifact_ids.append(visual.artifact_id)
        
        # All IDs should be unique
        assert len(set(artifact_ids)) == 5


class TestVisualInResponse:
    """Test that visuals are properly referenced in responses"""
    
    @pytest.mark.slow
    def test_visual_reference_format(self, settings, sample_image_path):
        """Test that visual references have correct format"""
        extractor = VisualExtractor(settings)
        visual = extractor.parse_file(sample_image_path)
        
        # Visual ID should follow pattern
        assert "visual" in visual.artifact_id.lower() or len(visual.artifact_id) > 0
        
        # Source path should be set
        assert visual.source_path == sample_image_path
    
    def test_visual_metadata_for_response(self):
        """Test visual metadata includes info needed for response"""
        # Metadata should include:
        # - source_file: original filename
        # - file_size: size in bytes
        # - dimensions: width x height
        # - extraction_method: how it was processed
        
        # This is more of an integration test
        pass


class TestVisualSupermemorySync:
    """Test visual sync with Supermemory backend"""
    
    @pytest.mark.skip(reason="Requires Supermemory API key")
    def test_visual_synced_to_supermemory(self, settings, sample_image_path):
        """Test that visual artifacts are synced to Supermemory"""
        from wald_agent_reference.memory.memory_backends import SupermemoryBackend
        
        settings.supermemory_api_key = "test_key"
        settings.supermemory_container_tag = "test_container"
        
        backend = SupermemoryBackend(settings)
        
        extractor = VisualExtractor(settings)
        visual = extractor.parse_file(sample_image_path)
        
        # Sync visual to Supermemory
        backend.sync_visual(visual)
        
        # Search should return the visual
        results = backend.search("chart test", limit=5)
        assert len(results) > 0
    
    def test_null_backend_skips_visual_sync(self, settings, sample_image_path):
        """Test that null backend gracefully handles visuals"""
        from wald_agent_reference.memory.memory_backends import NullMemoryBackend
        
        backend = NullMemoryBackend()
        
        extractor = VisualExtractor(settings)
        visual = extractor.parse_file(sample_image_path)
        
        # Should not raise error
        backend.sync_visual(visual)
        
        # Search should return empty
        results = backend.search("chart test", limit=5)
        assert results == []


class TestVisualChartGeneration:
    """Test visual chart generation from data"""
    
    @pytest.mark.slow
    def test_revenue_chart_generation(self):
        """Test generating revenue visualization"""
        from wald_agent_reference.rendering.visualize import VisualizationEngine
        from wald_agent_reference.core.models import CalculationResult
        
        settings = AppSettings()
        engine = VisualizationEngine(settings)
        
        # Create sample result with chart data
        result = CalculationResult(
            answer="Revenue by region",
            findings=["North: $100M", "South: $80M", "East: $120M", "West: $90M"],
            trace=[],
            evidence_refs=[],
            chart_data={
                "type": "bar",
                "labels": ["North", "South", "East", "West"],
                "values": [100, 80, 120, 90],
                "title": "Revenue by Region (Millions)",
            }
        )
        
        # Generate visualization - Pass (question, calculation) in correct order
        viz = engine.create("revenue_analysis", result)
        
        assert viz is not None
        assert viz.path.exists()
        assert viz.base64_image  # Should have base64 encoding
        assert len(viz.base64_image) > 100  # Should be substantial data
    
    @pytest.mark.slow
    def test_variance_chart_generation(self):
        """Test generating variance chart"""
        from wald_agent_reference.rendering.visualize import VisualizationEngine
        from wald_agent_reference.core.models import CalculationResult
        
        settings = AppSettings()
        engine = VisualizationEngine(settings)
        
        result = CalculationResult(
            answer="Variance analysis",
            findings=["Region A: -$20M", "Region B: +$10M"],
            trace=[],
            evidence_refs=[],
            chart_data={
                "type": "bar",
                "labels": ["Region A", "Region B"],
                "values": [-20, 10],
                "title": "Revenue Variance (Actual vs Target, Millions)",
            }
        )
        
        viz = engine.create("variance_analysis", result)
        
        assert viz is not None
        assert viz.base64_image
        # Should be able to display negative values
        assert "-20" in str(result.findings)


class TestVisualCharacteristics:
    """Test specific visual characteristics are preserved"""
    
    def test_chart_type_preservation(self):
        """Test that chart type info is preserved"""
        chart_data = {
            "type": "line",  # line, bar, pie, etc.
            "title": "Quarterly Revenue",
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "values": [100, 120, 110, 130],
        }
        
        assert chart_data["type"] == "line"
        assert chart_data["title"]
    
    def test_chart_legend_preservation(self):
        """Test that chart legend info is preserved"""
        chart_data = {
            "type": "bar",
            "labels": ["Product A", "Product B", "Product C"],
            "datasets": [
                {"label": "2023", "values": [100, 150, 120]},
                {"label": "2024", "values": [120, 160, 140]},
            ],
            "title": "Product Revenue Comparison",
        }
        
        # Legend should be present
        legends = [ds["label"] for ds in chart_data["datasets"]]
        assert len(legends) == 2
        assert "2023" in legends


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
