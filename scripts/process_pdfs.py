import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.data_extractor import FinancialDataExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)

def process_pdfs():
    """Process all PDFs in the data/pdfs directory."""
    try:
        # Initialize data extractor
        extractor = FinancialDataExtractor()
        
        # Get all PDF files
        pdf_dir = Path("data/pdfs")
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing {pdf_path.name}")
                extractor.extract_structured_data(pdf_path)
                logger.info(f"Successfully processed {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                continue
        
        logger.info("PDF processing completed")
        
    except Exception as e:
        logger.error(f"Error during PDF processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    process_pdfs() 