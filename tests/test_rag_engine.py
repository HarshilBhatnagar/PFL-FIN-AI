import pytest
from pathlib import Path
import json
import shutil
from app.services.rag_engine import RAGEngine

@pytest.fixture
def rag_engine():
    # Create test directories
    base_dir = Path("data")
    chunks_dir = base_dir / "chunks"
    index_dir = base_dir / "embeddings"
    
    # Clean up existing test data
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    # Create test data
    chunks_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    
    # Create test PDF directory
    test_pdf_dir = chunks_dir / "test_pdf"
    test_pdf_dir.mkdir()
    
    # Create test chunks
    test_chunks = [
        "The company's NPA ratio was 2.5% in Q4 2023.",
        "Total assets grew by 15% year-over-year.",
        "Net profit margin improved to 18% in the last quarter."
    ]
    
    # Save test chunks
    with open(test_pdf_dir / "financial_highlights_chunks.json", 'w') as f:
        json.dump(test_chunks, f)
    
    # Initialize RAG engine
    engine = RAGEngine()
    return engine

def test_index_creation(rag_engine):
    """Test index creation and loading."""
    assert rag_engine.index is not None
    assert len(rag_engine.chunk_metadata) > 0
    assert rag_engine.index.ntotal > 0

def test_hybrid_search(rag_engine):
    """Test hybrid search functionality."""
    query = "What was the NPA ratio?"
    results = rag_engine._hybrid_search(query, k=2)
    
    assert len(results) > 0
    assert "NPA" in results[0]["text"]
    assert "metadata" in results[0]
    assert "pdf" in results[0]["metadata"]
    assert "section" in results[0]["metadata"]

def test_query_vectorstore(rag_engine):
    """Test vector store querying."""
    query = "What was the profit margin?"
    results = rag_engine.query_vectorstore(query, k=2)
    
    assert len(results) > 0
    assert any("profit" in result.lower() for result in results)

def test_generate_answer(rag_engine):
    """Test answer generation."""
    query = "What was the NPA ratio?"
    chunks = ["The company's NPA ratio was 2.5% in Q4 2023."]
    context = {"quarter": "Q4 2023"}
    
    answer = rag_engine.generate_answer(query, chunks, context)
    
    assert answer is not None
    assert "2.5%" in answer
    assert "NPA" in answer

def test_update_index(rag_engine):
    """Test index updating."""
    # Get initial index size
    initial_size = rag_engine.index.ntotal
    
    # Add new chunks
    test_pdf_dir = rag_engine.chunks_dir / "test_pdf2"
    test_pdf_dir.mkdir()
    
    new_chunks = [
        "The company's revenue grew by 20% in Q1 2024.",
        "Operating expenses decreased by 5%."
    ]
    
    with open(test_pdf_dir / "financial_highlights_chunks.json", 'w') as f:
        json.dump(new_chunks, f)
    
    # Update index
    rag_engine.update_index()
    
    # Check if index was updated
    assert rag_engine.index.ntotal > initial_size

def test_error_handling(rag_engine):
    """Test error handling in various scenarios."""
    # Test with empty query
    results = rag_engine.query_vectorstore("", k=2)
    assert len(results) == 0
    
    # Test with invalid k value
    results = rag_engine.query_vectorstore("test", k=-1)
    assert len(results) == 0
    
    # Test with empty chunks
    answer = rag_engine.generate_answer("test", [], None)
    assert "error" in answer.lower() or "apologize" in answer.lower()

def test_metadata_persistence(rag_engine):
    """Test metadata persistence after index reload."""
    # Get initial metadata
    initial_metadata = rag_engine.chunk_metadata.copy()
    
    # Create new RAG engine instance
    new_engine = RAGEngine()
    
    # Check if metadata is preserved
    assert len(new_engine.chunk_metadata) == len(initial_metadata)
    assert new_engine.chunk_metadata[0]["pdf"] == initial_metadata[0]["pdf"]
    assert new_engine.chunk_metadata[0]["section"] == initial_metadata[0]["section"] 