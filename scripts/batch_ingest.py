import os
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.data_extractor import FinancialDataExtractor
from app.services.data_validator import DataValidator
from app.services.rag_engine import RAGEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize services
data_extractor = FinancialDataExtractor()
data_validator = DataValidator()
rag_engine = RAGEngine()

# Define paths
pdfs_dir = Path("data/pdfs")
chunks_dir = Path("data/chunks")

# Ensure chunks directory exists
chunks_dir.mkdir(parents=True, exist_ok=True)

def process_pdf(pdf_path):
    """Process a single PDF file: extract, validate, and save chunks."""
    pdf_name = pdf_path.stem
    pdf_chunks_dir = chunks_dir / pdf_name

    # Skip if already processed
    if pdf_chunks_dir.exists():
        logger.info(f"Skipping {pdf_name} - already processed.")
        return False

    logger.info(f"Processing {pdf_name}...")
    try:
        # Extract structured data and save chunks
        structured_data = data_extractor.extract_structured_data(pdf_path)
        logger.info(f"Extracted data from {pdf_name}")

        # Validate data
        validation_results = data_validator.validate_metrics(structured_data.get("metrics", {}), pdf_path)
        if not validation_results["is_valid"]:
            logger.warning(f"Validation failed for {pdf_name}: {validation_results['errors']}")
            return False

        logger.info(f"Successfully processed {pdf_name}")
        return True
    except Exception as e:
        logger.error(f"Error processing {pdf_name}: {str(e)}")
        return False

def main():
    """Process all new PDFs in the data/pdfs folder."""
    processed_count = 0
    skipped_count = 0

    for pdf_file in pdfs_dir.glob("*.pdf"):
        if process_pdf(pdf_file):
            processed_count += 1
        else:
            skipped_count += 1

    # Update RAG engine index once after processing all PDFs
    logger.info("Updating RAG engine index...")
    rag_engine.update_index()

    logger.info(f"Batch ingestion complete. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main() 