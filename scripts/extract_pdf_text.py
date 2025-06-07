import os
from pathlib import Path
from services.pdf_ingestor import extract_text_from_pdf
from app.utils.logger import get_logger

logger = get_logger(__name__)

def process_pdfs():
    # Setup directories
    pdf_dir = Path("data/pdfs")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(exist_ok=True)

    # Process each PDF
    for pdf_file in pdf_dir.glob("*.pdf"):
        logger.info(f"Processing {pdf_file.name}")
        
        try:
            # Extract text
            text = extract_text_from_pdf(str(pdf_file))
            
            # Save to raw text file
            output_file = raw_dir / f"{pdf_file.stem}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            logger.info(f"Saved text to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")

if __name__ == "__main__":
    process_pdfs() 