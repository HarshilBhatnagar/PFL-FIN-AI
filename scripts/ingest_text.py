import os
import sys
import logging
from pathlib import Path
import shutil

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_vector_store():
    """Clear the existing vector store."""
    try:
        persist_dir = "data/chroma_db"
        if os.path.exists(persist_dir):
            logger.info("Clearing existing vector store...")
            shutil.rmtree(persist_dir)
            logger.info("Vector store cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise

def main():
    try:
        # Clear existing vector store
        clear_vector_store()
        
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine()
        
        # Process text files
        text_dir = "data/texts"
        if not os.path.exists(text_dir):
            raise ValueError(f"Text directory not found: {text_dir}")
            
        logger.info(f"Processing text files from {text_dir}")
        results = rag_engine.process_directory(text_dir)
        
        # Log results
        logger.info("Ingestion complete!")
        logger.info(f"Files processed: {results['processed']}")
        logger.info(f"Files skipped: {results['skipped']}")
        logger.info(f"Files failed: {results['failed']}")
        
        if results['errors']:
            logger.warning("Errors encountered during processing:")
            for error in results['errors']:
                logger.warning(f"- {error}")
                
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 