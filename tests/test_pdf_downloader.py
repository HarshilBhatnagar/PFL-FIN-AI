import pytest
from pathlib import Path
import json
from scripts.download_pdfs import PDFDownloader
import shutil

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture
def downloader(temp_dir):
    """Create a PDFDownloader instance with temporary directory."""
    downloader = PDFDownloader()
    downloader.base_dir = temp_dir / "pdfs"
    downloader.base_dir.mkdir(parents=True, exist_ok=True)
    downloader.metadata_file = downloader.base_dir / "metadata.json"
    return downloader

def test_metadata_creation(downloader):
    """Test metadata file creation and loading."""
    # Test initial metadata
    assert downloader.metadata == {"downloads": []}
    
    # Test saving and loading metadata
    test_data = {
        "downloads": [{
            "nbfc": "Test NBFC",
            "url": "http://example.com/test.pdf",
            "filename": "test.pdf",
            "download_date": "2024-01-01T00:00:00"
        }]
    }
    
    downloader.metadata = test_data
    downloader._save_metadata()
    
    # Create new instance to test loading
    new_downloader = PDFDownloader()
    new_downloader.base_dir = downloader.base_dir
    new_downloader.metadata_file = downloader.metadata_file
    new_downloader.metadata = new_downloader._load_metadata()
    
    assert new_downloader.metadata == test_data

def test_nbfc_list(downloader):
    """Test NBFC list initialization."""
    assert len(downloader.nbfcs) > 0
    assert "Bajaj Finance" in downloader.nbfcs
    assert "HDFC Bank" in downloader.nbfcs

def test_directory_creation(downloader):
    """Test PDF directory creation."""
    assert downloader.base_dir.exists()
    assert downloader.base_dir.is_dir()

def test_url_parsing(downloader):
    """Test URL parsing for NBFCs."""
    urls = downloader._get_nbfc_urls("Bajaj Finance")
    assert isinstance(urls, list) 