import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.logger import get_logger

logger = get_logger(__name__)

def chunk_text_files():
    # Setup directories
    raw_dir = Path("data/raw")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(exist_ok=True)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    # Process each raw text file
    for text_file in raw_dir.glob("*.txt"):
        logger.info(f"Chunking {text_file.name}")
        
        try:
            # Read text
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split into chunks
            chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Save chunks
            output_file = chunks_dir / f"{text_file.stem}_chunks.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"--- Chunk {i+1} ---\n")
                    f.write(chunk)
                    f.write("\n\n")
            
            logger.info(f"Saved chunks to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {text_file.name}: {str(e)}")

if __name__ == "__main__":
    chunk_text_files() 