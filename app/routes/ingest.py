from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.data_extractor import FinancialDataExtractor
from app.services.data_validator import DataValidator
from app.services.rag_engine import RAGEngine
from app.utils.logger import get_logger
from pathlib import Path
import shutil
import os

router = APIRouter()
logger = get_logger(__name__)

# Initialize services
data_extractor = FinancialDataExtractor()
data_validator = DataValidator()
rag_engine = RAGEngine()

# Ensure data/chunks directory exists
chunks_dir = Path("data/chunks")
chunks_dir.mkdir(parents=True, exist_ok=True)

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a financial document (PDF) and process it for querying.
    """
    try:
        # Create temporary directory for processing
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = temp_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract structured data
        logger.info(f"Extracting data from {file.filename}")
        structured_data = data_extractor.extract_structured_data(file_path)
        
        # Validate data
        logger.info("Validating extracted data")
        validation_results = data_validator.validate_metrics(
            structured_data.get("metrics", {}),
            file_path
        )
        
        if not validation_results["is_valid"]:
            logger.warning(f"Validation failed: {validation_results['errors']}")
            return {
                "status": "error",
                "message": "Data validation failed",
                "errors": validation_results["errors"],
                "warnings": validation_results["warnings"]
            }
        
        # Update RAG engine index
        logger.info("Updating search index")
        rag_engine.update_index()
        
        # Clean up
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": "Document processed successfully",
            "validation_results": validation_results,
            "quality_score": validation_results["quality_score"]
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir) 