import os
import sys
import logging
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Query the RAG system.')
        parser.add_argument('--query', type=str, default="What was the total income and net profit for Axis Finance Limited in Q4 FY2025?", help='The query to ask the RAG system.')
        args = parser.parse_args()

        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine()
        
        # Query
        logger.info(f"Querying: {args.query}")
        
        # Get response
        response = rag_engine.query(args.query)
        
        # Log response
        logger.info("Query response:")
        logger.info(f"Response: {response['response']}")
        logger.info(f"Context used: {response['context_used']}")
        logger.info(f"Processing time: {response['processing_time']} seconds")
        logger.info(f"Status: {response['status']}")
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 