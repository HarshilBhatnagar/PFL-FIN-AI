import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_engine import RAGEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Reindex all documents in the data/chunks directory."""
    try:
        logger.info("Starting reindexing process")
        rag_engine = RAGEngine()
        rag_engine.update_index()
        logger.info("Reindexing completed successfully")
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 