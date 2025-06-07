import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import logging
from typing import List, Dict
import shutil
import re
from src.rag_engine import RAGEngine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def is_table_line(line: str) -> bool:
    # Heuristic: line contains at least 2 pipes and is not empty
    return line.count('|') >= 2 and len(line.strip()) > 0

def chunk_text_with_tables(text: str, min_chunk_size: int = 200, max_chunk_size: int = 1000) -> List[str]:
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    in_table = False

    for line in lines:
        if is_table_line(line):
            if not in_table:
                # If we were in narrative, flush the chunk
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                in_table = True
            current_chunk.append(line)
        else:
            if in_table:
                # End of table, flush the table chunk
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                in_table = False
            current_chunk.append(line)
            # Optionally, flush narrative chunks if they get too big
            if sum(len(l) for l in current_chunk) > max_chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return [c for c in chunks if c.strip()]

def process_text_file(file_path: str) -> str:
    """Process text file and clean up the content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Clean up the content
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove extra whitespace
            line = ' '.join(line.split())
            # Skip empty lines
            if not line.strip():
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return ""

def process_and_store_texts():
    """Process text files and store them in the vector store."""
    try:
        # Get all text files
        text_files = glob.glob("data/texts/*.txt")
        rag_engine = RAGEngine()
        for file_path in text_files:
            try:
                logger.info(f"Processing file: {file_path}")
                # Process and clean the text
                processed_text = process_text_file(file_path)
                if not processed_text:
                    continue
                # Create chunks using table-aware logic
                chunks = chunk_text_with_tables(processed_text)
                # Create documents with metadata
                documents = []
                for i, chunk in enumerate(chunks):
                    is_table = any(is_table_line(line) for line in chunk.split('\n'))
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "is_table": is_table
                        }
                    )
                    documents.append(doc)
                # Add documents to vector store
                if documents:
                    rag_engine.vector_store.add_documents(documents)
                    logger.info(f"Added {len(documents)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        logger.info("Text processing and storage completed")
    except Exception as e:
        logger.error(f"Error in process_and_store_texts: {str(e)}")
        raise

def clean_directories():
    """Clean up existing chunks and embeddings directories."""
    logger.info("Step 1: Cleaning up existing data")
    # Clean up chunks directory
    chunks_dir = Path("data/chunks")
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    logger.info("Cleaned up chunks directory")
    # Clean up embeddings directory
    embeddings_dir = Path("data/embeddings")
    if embeddings_dir.exists():
        shutil.rmtree(embeddings_dir)
    logger.info("Cleaned up embeddings directory")
    # Create fresh directories
    chunks_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created fresh directories for chunks and embeddings")

def main():
    try:
        clean_directories()
        process_and_store_texts()
        logger.info("Process completed successfully!")
    except Exception as e:
        logger.error(f"Error during ingestion process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 