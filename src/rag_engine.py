import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
from pathlib import Path
import logging
from datetime import datetime
import sys

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_engine.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import GraphGenerator
from graph_utils import GraphGenerator

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine with necessary components."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=self.embeddings
        )
        
        # Initialize session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Created new session: {self.session_id}")
        
        # Initialize graph generator
        try:
            self.graph_generator = GraphGenerator()
            logger.info("GraphGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GraphGenerator: {str(e)}")
            self.graph_generator = None
        
        # Load existing documents if any
        count = self.vector_store._collection.count()
        logger.info(f"Vector store contains {count} documents")
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                model="llama3.2",
                temperature=0.5,
                max_tokens=512,
                base_url="http://localhost:11434"  # Default Ollama URL
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            return None
            
    def _get_relevant_context(self, query: str) -> List[Document]:
        """Get relevant context for a query."""
        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=5)
            logger.info(f"Found {len(docs)} relevant documents")
            return docs
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return []
            
    def _generate_response(self, query: str, context: List[Document]) -> str:
        """Generate a response using the LLM."""
        if not self.llm:
            return "Error: Language model not initialized."
            
        try:
            # Prepare context
            context_text = "\n".join([doc.page_content for doc in context])
            
            # Generate response
            prompt = f"""Based on the following context, answer the query.
            If the query asks for a graph or visualization, explain that you'll generate one.
            
            Query: {query}
            
            Context:
            {context_text}
            
            Answer:"""
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
            
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return a response with graph if needed."""
        try:
            # Get context
            context = self._get_relevant_context(query)
            
            # Generate response
            response = self._generate_response(query, context)
            
            # Check if graph is needed
            graph_data = None
            embed_code = None
            filepath = None
            
            if self.graph_generator and any(word in query.lower() for word in ["graph", "plot", "chart", "visualize", "show"]):
                try:
                    graph_result = self.graph_generator.generate_graph(
                        query=query,
                        context=[doc.page_content for doc in context],
                        context_docs=context
                    )
                    
                    if "error" not in graph_result:
                        graph_data = graph_result.get("graph_data")
                        embed_code = graph_result.get("embed_code")
                        filepath = graph_result.get("filepath")
                        logger.info(f"Graph generated successfully: {filepath}")
                except Exception as e:
                    logger.error(f"Error generating graph: {str(e)}")
            
            return {
                "response": response,
                "context_used": context,
                "graph_data": graph_data,
                "embed_code": embed_code,
                "filepath": filepath
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "context_used": [],
                "graph_data": None,
                "embed_code": None,
                "filepath": None
            }

    def _initialize_vector_store(self):
        try:
            persist_directory = "data/chroma_db"
            # Always clear existing vector store
            if os.path.exists(persist_directory):
                logger.info("Clearing existing vector store...")
                import shutil
                shutil.rmtree(persist_directory)
                logger.info("Vector store cleared")
            
            # Create new vector store
            logger.info("Creating new vector store...")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("New vector store created")
            
            # Generate new session ID
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Created new session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _load_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if not content:
                    logger.warning(f"Empty file: {file_path}")
                    return ""
                return content
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return ""

    def _create_chunks(self, text: str, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        try:
            if not text.strip():
                return []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create documents with proper metadata
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "file_name": Path(file_path).name,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            return documents
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return []

    def _load_chunks_file(self, file_path: str) -> List[Document]:
        """Load chunks from a processed chunks file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if not content:
                    logger.warning(f"Empty chunks file: {file_path}")
                    return []
                
                # Split content into chunks based on the chunk markers
                chunks = []
                current_chunk = []
                
                for line in content.split('\n'):
                    if line.startswith('--- Chunk'):
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = []
                    else:
                        current_chunk.append(line)
                
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                
                logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
                
                # Create documents with proper metadata
                documents = []
                for chunk in chunks:
                    if chunk.strip():
                        doc = Document(
                            page_content=chunk.strip(),
                            metadata={
                                "source": file_path,
                                "file_name": Path(file_path).name,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                
                logger.info(f"Created {len(documents)} documents from chunks")
                return documents
        except Exception as e:
            logger.error(f"Error loading chunks file {file_path}: {str(e)}")
            return []

    def process_text_file(self, file_path: str) -> None:
        """Process a text file and add its contents to the vector store."""
        try:
            logger.info(f"Processing {file_path}")
            
            # First try to load from chunks file
            chunks_file = Path("data/chunks") / f"{Path(file_path).stem}_chunks.txt"
            if chunks_file.exists():
                logger.info(f"Found chunks file: {chunks_file}")
                with open(chunks_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Parse chunks
                chunks = []
                current_chunk = []
                for line in content.split("\n"):
                    if line.startswith("--- Chunk"):
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                            current_chunk = []
                    else:
                        current_chunk.append(line)
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                
                logger.info(f"Loaded {len(chunks)} chunks from file")
                
                # Create documents with metadata
                documents = []
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
                
                # Add to vector store
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
                
                # Persist the vector store
                self.vector_store.persist()
                
                # Verify documents were added
                count = self.vector_store._collection.count()
                logger.info(f"Vector store now contains {count} documents")
                
                return
            
            # If no chunks file, process the text file directly
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Create a single document for the entire file
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "chunk_id": 0,
                    "total_chunks": 1
                }
            )
            
            # Add to vector store
            self.vector_store.add_documents([doc])
            logger.info(f"Added document to vector store")
            
            # Persist the vector store
            self.vector_store.persist()
            
            # Verify document was added
            count = self.vector_store._collection.count()
            logger.info(f"Vector store now contains {count} documents")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        try:
            results = {
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": []
            }
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory not found: {directory_path}")
            
            # Clear existing vector store before processing directory
            self._initialize_vector_store()
            
            # Process both .txt and _chunks.txt files
            text_files = list(directory.glob("**/*.txt"))
            total_files = len(text_files)
            logger.info(f"Found {total_files} text files to process")
            
            for file_path in text_files:
                try:
                    if self.process_text_file(str(file_path)):
                        results["processed"] += 1
                    else:
                        results["skipped"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{file_path}: {str(e)}")
            
            # Force final persist
            self.vector_store.persist()
            
            logger.info(f"Directory processing complete. Results: {json.dumps(results, indent=2)}")
            return results
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

    def _call_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama and return the response."""
        try:
            payload = {
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

    def query(self, query: str) -> Dict[str, Any]:
        """Process a query and return a response with metadata."""
        try:
            # Check if vector store is empty
            count = self.vector_store._collection.count()
            if count == 0:
                logger.warning("Vector store is empty")
                return {
                    "response": "I apologize, but I don't have any documents loaded to answer your question.",
                    "source_files": [],
                    "file_names": [],
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": "Vector store is empty"
                }
            
            # Get relevant context
            context_docs = self._get_relevant_context(query)
            if not context_docs:
                return {
                    "response": "I apologize, but I couldn't find any relevant information to answer your question.",
                    "source_files": [],
                    "file_names": [],
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": "No relevant context found"
                }
            
            # Generate response
            response = self._generate_response(query, context_docs)
            
            # Extract source files and file names
            source_files = {doc.metadata.get('source', 'Unknown') for doc in context_docs}
            file_names = {Path(source).name for source in source_files}
            
            return {
                "response": response,
                "source_files": list(source_files),
                "file_names": list(file_names),
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "source_files": [],
                "file_names": [],
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }

    def clear_session(self):
        """Clear current session and reinitialize vector store."""
        try:
            self._initialize_vector_store()
            logger.info("Session cleared and vector store reinitialized")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}")
            return False 