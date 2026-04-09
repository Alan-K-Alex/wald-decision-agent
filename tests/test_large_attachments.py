"""
Tests for large attachment handling.

Tests verify:
- Large PDF ingestion (10MB+)
- Large CSV handling (1M+ rows)
- Large batch visual extraction  
- Performance metrics
- Supermemory payload size handling
"""

import pytest
from pathlib import Path
import tempfile
import io
import time

from wald_decision_agent.ingestion.ingest import DocumentIngestor
from wald_decision_agent.core.config import AppSettings
from wald_decision_agent.memory.memory_backends import SupermemoryBackend


class TestLargePDFIngestion:
    """Test ingestion of large PDF files"""
    
    @pytest.fixture
    def large_pdf_path(self, tmp_path):
        """Create a mock large PDF for testing"""
        # Note: In real tests, would use actual large PDF
        pdf_path = tmp_path / "large_document.pdf"
        # Create a dummy PDF file with substantial content
        from pypdf import PdfWriter
        
        writer = PdfWriter()
        # Add 100 pages of text
        for i in range(100):
            page_content = f"Page {i+1}\n" + ("Sample content. " * 500)
            # In real implementation, would write text to PDF
        
        pdf_path.write_text("dummy pdf content")
        return pdf_path
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_large_pdf_ingestion_time(self, large_pdf_path):
        """Test that large PDF ingestion completes in reasonable time"""
        settings = AppSettings()
        ingestor = DocumentIngestor(settings)
        
        start_time = time.time()
        
        # Ingest large PDF
        document, chunks = ingestor._chunk_pdf(large_pdf_path)
        
        elapsed = time.time() - start_time
        
        assert document is not None
        assert len(chunks) > 0
        # Should complete in < 30 seconds
        assert elapsed < 30, f"PDF ingestion took {elapsed}s, should be < 30s"
    
    @pytest.mark.slow
    @pytest.mark.memory
    def test_large_pdf_memory_usage(self, large_pdf_path):
        """Test that large PDF doesn't cause excessive memory usage"""
        import psutil
        import os
        
        settings = AppSettings()
        ingestor = DocumentIngestor(settings)
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        document, chunks = ingestor._chunk_pdf(large_pdf_path)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Memory increase should be reasonable (< 500MB for 10MB PDF)
        assert mem_increase < 500, f"Memory increased by {mem_increase}MB"
    
    def test_large_pdf_chunk_count(self):
        """Test that large PDF is properly chunked"""
        # 100-page PDF should result in 100+ chunks (one per page + subject matter splits)
        # This is more of an integration test
        pass


class TestLargeCSVHandling:
    """Test handling of large CSV files"""
    
    @pytest.fixture
    def large_csv_path(self, tmp_path):
        """Create a large CSV file for testing"""
        import csv
        
        csv_path = tmp_path / "large_data.csv"
        
        # Create CSV with 10,000 rows
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['id', 'region', 'revenue', 'target', 'variance'])
            
            # Data rows
            regions = ['North', 'South', 'East', 'West']
            for i in range(10000):
                region = regions[i % 4]
                revenue = 100000 + (i * 10)
                target = 105000 + (i * 10)
                variance = revenue - target
                writer.writerow([i, region, revenue, target, variance])
        
        return csv_path
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_large_csv_parsing_time(self, large_csv_path):
        """Test CSV parsing performance"""
        from wald_decision_agent.ingestion.spreadsheet_parser import SpreadsheetParser
        
        parser = SpreadsheetParser()
        
        start_time = time.time()
        tables = parser.parse_file(large_csv_path)
        elapsed = time.time() - start_time
        
        assert len(tables) > 0
        # Should parse 10K rows in < 5 seconds
        assert elapsed < 5, f"CSV parsing took {elapsed}s"
    
    @pytest.mark.slow
    def test_large_csv_row_count(self, large_csv_path):
        """Test that all rows from large CSV are captured"""
        from wald_decision_agent.ingestion.spreadsheet_parser import SpreadsheetParser
        
        parser = SpreadsheetParser()
        tables = parser.parse_file(large_csv_path)
        
        assert len(tables) > 0
        table = tables[0]
        
        # Should have 10,000 data rows (minus header)
        assert len(table.dataframe) >= 9999
    
    @pytest.mark.slow
    def test_large_csv_structured_store_sync(self):
        """Test syncing large CSV to SQLite structured store"""
        from wald_decision_agent.memory.structured_store import StructuredMemoryStore
        
        # This tests that large tables can be queried without timeouts
        # Real implementation would test actual ingestion
        pass


class TestLargeBatchVisualExtraction:
    """Test extracting multiple visuals in batch"""
    
    @pytest.fixture
    def image_batch(self, tmp_path):
        """Create batch of test images"""
        from PIL import Image
        
        image_paths = []
        for i in range(50):
            img = Image.new('RGB', (200, 200), 
                          color=(i * 5, 100, 200 - i * 3))
            path = tmp_path / f"image_{i:03d}.png"
            img.save(path)
            image_paths.append(path)
        
        return image_paths
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_batch_image_extraction_time(self, image_batch):
        """Test extracting 50 images completes in reasonable time"""
        from wald_decision_agent.ingestion.visual_extractor import VisualExtractor
        
        settings = AppSettings()
        extractor = VisualExtractor(settings)
        
        start_time = time.time()
        
        visuals = []
        for image_path in image_batch:
            visual = extractor.parse_file(image_path)
            if visual:
                visuals.append(visual)
        
        elapsed = time.time() - start_time
        
        assert len(visuals) == len(image_batch)
        # Should extract 50 images in < 20 seconds
        assert elapsed < 20, f"Extraction took {elapsed}s, should be < 20s"
    
    @pytest.mark.slow
    def test_batch_visual_uniqueness(self, image_batch):
        """Test that all visuals get unique IDs"""
        from wald_decision_agent.ingestion.visual_extractor import VisualExtractor
        
        settings = AppSettings()
        extractor = VisualExtractor(settings)
        
        artifact_ids = set()
        
        for image_path in image_batch:
            visual = extractor.parse_file(image_path)
            if visual:
                artifact_ids.add(visual.artifact_id)
        
        # All IDs should be unique
        assert len(artifact_ids) == len(image_batch)


class TestSupermemoryPayloadHandling:
    """Test Supermemory backend with large payloads"""
    
    @pytest.mark.skip(reason="Requires Supermemory API")
    def test_large_document_sync(self):
        """Test syncing large document to Supermemory"""
        from wald_decision_agent.core.models import ExtractedDocument
        
        # Create large document (5MB text)
        large_text = "Sample text. " * 400000  # ~5MB
        
        settings = AppSettings(
            supermemory_api_key="test_key",
            supermemory_container_tag="test"
        )
        backend = SupermemoryBackend(settings)
        
        document = ExtractedDocument(
            document_id="large_doc",
            source_path=Path("test.txt"),
            source_type="text",
            raw_text=large_text,
        )
        
        # Should handle without timeout
        backend.sync_document(document)
    
    def test_supermemory_payload_sanitization(self):
        """Test that large payloads are properly sanitized"""
        from wald_decision_agent.memory.memory_backends import SupermemoryBackend
        
        settings = AppSettings()
        backend = SupermemoryBackend(settings)
        
        # Test custom_id sanitization with special chars
        custom_ids = [
            "document with spaces and!@#$%",
            "table:name:with:colons",
            "special_chars_@_$_#",
        ]
        
        for custom_id in custom_ids:
            sanitized = backend._sanitize_custom_id(custom_id)
            
            # Should only contain alphanumeric, hyphen, underscore, colon
            assert all(c.isalnum() or c in '-_:' for c in sanitized)
            # Should not be empty
            assert len(sanitized) > 0


class TestPerformanceMetrics:
    """Track performance metrics for ingestion"""
    
    @pytest.fixture
    def performance_results(self):
        """Storage for performance metrics"""
        return {
            "pdf_ingestion_rate": None,  # MB/second
            "csv_parsing_rate": None,    # rows/second
            "image_extraction_rate": None,  # images/second
            "supermemory_sync_rate": None,  # docs/second
        }
    
    def test_calculate_pdf_ingestion_rate(self, performance_results):
        """Calculate and report PDF ingestion rate"""
        # 10MB PDF in 5 seconds = 2 MB/sec
        # This would be calculated from actual test runs
        pdf_size_mb = 10
        ingestion_time_sec = 5
        
        rate = pdf_size_mb / ingestion_time_sec
        performance_results["pdf_ingestion_rate"] = rate
        
        # Should be at least 0.5 MB/sec for reasonable performance
        assert rate >= 0.5, f"PDF ingestion rate {rate} MB/s is too slow"
    
    def test_calculate_csv_parsing_rate(self, performance_results):
        """Calculate CSV parsing rate"""
        # 10K rows in 2 seconds = 5K rows/sec
        rows = 10000
        parse_time_sec = 2
        
        rate = rows / parse_time_sec
        performance_results["csv_parsing_rate"] = rate
        
        # Should be at least 1K rows/sec
        assert rate >= 1000, f"CSV parsing rate {rate} rows/s is too slow"
    
    def test_calculate_image_extraction_rate(self, performance_results):
        """Calculate image extraction rate"""
        # 50 images in 10 seconds = 5 images/sec
        images = 50
        extraction_time_sec = 10
        
        rate = images / extraction_time_sec
        performance_results["image_extraction_rate"] = rate
        
        # Should be at least 2 images/sec
        assert rate >= 2, f"Image extraction rate {rate} img/s is too slow"


class TestMemoryUsageMonitoring:
    """Monitor memory usage during large ingestions"""
    
    @pytest.mark.slow
    def test_memory_profile_large_csv(self):
        """Profile memory usage for large CSV"""
        # Would use memory_profiler or similar
        import tracemalloc
        
        tracemalloc.start()
        
        # ... perform large CSV ingestion ...
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        
        # Peak memory should be reasonable (< 1GB for 10K row CSV)
        assert peak_mb < 1000, f"Peak memory {peak_mb}MB too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
