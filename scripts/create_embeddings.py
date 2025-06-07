import os
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from app.utils.logger import get_logger

logger = get_logger(__name__)

def create_embeddings():
    # Setup directories
    chunks_dir = Path("data/chunks")
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(exist_ok=True)

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Collect all chunks
    documents = []
    for chunk_file in chunks_dir.glob("*_chunks.txt"):
        logger.info(f"Processing chunks from {chunk_file.name}")
        
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split into individual chunks
            chunk_texts = [c.strip() for c in text.split("--- Chunk") if c.strip()]
            
            # Create documents
            for i, chunk in enumerate(chunk_texts):
                # Remove chunk number from text
                chunk_text = chunk.split("---\n", 1)[1].strip()
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": chunk_file.stem,
                        "chunk_id": i
                    }
                )
                documents.append(doc)
            
            logger.info(f"Added {len(chunk_texts)} chunks from {chunk_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {chunk_file.name}: {str(e)}")

    if not documents:
        logger.error("No documents found to create embeddings")
        return

    # Create FAISS index
    logger.info(f"Creating FAISS index with {len(documents)} documents")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    # Save index
    index_path = embeddings_dir / "faiss_index"
    vectorstore.save_local(str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")

if __name__ == "__main__":
    create_embeddings() 