import PyPDF2
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
import spacy
from app.utils.logger import get_logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = get_logger(__name__)

class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.base_dir = Path("data")
        self.raw_dir = self.base_dir / "raw"
        self.chunks_dir = self.base_dir / "chunks"
        self.metadata_file = self.base_dir / "text_metadata.json"
        
        # Create necessary directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
        
        # Define financial section patterns
        self.section_patterns = {
            "financial_highlights": r"Financial Highlights|Key Financial Indicators",
            "balance_sheet": r"Balance Sheet|Financial Position",
            "income_statement": r"Income Statement|Profit and Loss|P&L",
            "cash_flow": r"Cash Flow|Cash Flow Statement",
            "notes": r"Notes to Accounts|Notes to Financial Statements",
            "audit": r"Auditor's Report|Independent Auditor's Report",
            "management": r"Management Discussion and Analysis|MD&A"
        }
        
        # Initialize TF-IDF vectorizer for semantic chunking
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"processed_files": []}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF with section identification."""
        try:
            text_sections = {}
            current_section = "header"
            current_text = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    "filename": pdf_path.name,
                    "pages": len(pdf_reader.pages),
                    "processed_date": datetime.now().isoformat()
                }
                
                # Process each page
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    # Identify sections
                    for section_name, pattern in self.section_patterns.items():
                        if re.search(pattern, text, re.I):
                            if current_text:
                                text_sections[current_section] = "\n".join(current_text)
                            current_section = section_name
                            current_text = []
                            break
                    
                    current_text.append(text)
                
                # Add the last section
                if current_text:
                    text_sections[current_section] = "\n".join(current_text)
                
                return {
                    "metadata": metadata,
                    "sections": text_sections
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s.,;:()%$]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()

    def identify_entities(self, text: str) -> List[Dict[str, Any]]:
        """Identify financial entities in text."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'PERCENT', 'DATE', 'ORG']:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return entities

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return sent_tokenize(text)

    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences using TF-IDF."""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([sent1, sent2])
            return (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]
        except:
            return 0.0

    def _identify_topic_boundaries(self, sentences: List[str], similarity_threshold: float = 0.3) -> List[int]:
        """Identify topic boundaries based on sentence similarity."""
        boundaries = [0]  # Start with first sentence
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = self._calculate_sentence_similarity(sentences[i-1], sentences[i])
            
            # If similarity is below threshold, mark as boundary
            if similarity < similarity_threshold:
                boundaries.append(i)
        
        return boundaries

    def _merge_small_chunks(self, chunks: List[str], min_chunk_size: int = 100) -> List[str]:
        """Merge small chunks to maintain minimum size."""
        merged_chunks = []
        current_chunk = []
        current_size = 0
        
        for chunk in chunks:
            if current_size + len(chunk) <= min_chunk_size:
                current_chunk.append(chunk)
                current_size += len(chunk)
            else:
                if current_chunk:
                    merged_chunks.append(" ".join(current_chunk))
                current_chunk = [chunk]
                current_size = len(chunk)
        
        if current_chunk:
            merged_chunks.append(" ".join(current_chunk))
        
        return merged_chunks

    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create sophisticated text chunks using multiple strategies."""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Identify topic boundaries
        boundaries = self._identify_topic_boundaries(sentences)
        
        # Create initial chunks based on topic boundaries
        chunks = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            chunk = " ".join(sentences[start:end])
            chunks.append(chunk)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks, min_chunk_size=chunk_size//2)
        
        # Add overlapping chunks for context
        final_chunks = []
        for i, chunk in enumerate(chunks):
            # Add current chunk
            final_chunks.append(chunk)
            
            # Add overlapping chunk if not the last chunk
            if i < len(chunks) - 1:
                overlap_text = chunk[-overlap:] + " " + chunks[i+1][:overlap]
                final_chunks.append(overlap_text)
        
        return final_chunks

    def process_pdf(self, pdf_path: Path) -> bool:
        """Process a PDF file and save its contents."""
        try:
            # Extract text and sections
            result = self.extract_text_from_pdf(pdf_path)
            if not result:
                return False
            
            # Process each section
            for section_name, text in result["sections"].items():
                # Clean text
                cleaned_text = self.clean_text(text)
                
                # Identify entities
                entities = self.identify_entities(cleaned_text)
                
                # Create chunks with different strategies
                semantic_chunks = self.create_chunks(cleaned_text, chunk_size=1000, overlap=200)
                
                # Save processed data
                section_dir = self.raw_dir / pdf_path.stem
                section_dir.mkdir(exist_ok=True)
                
                # Save text
                with open(section_dir / f"{section_name}.txt", 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                # Save entities
                with open(section_dir / f"{section_name}_entities.json", 'w') as f:
                    json.dump(entities, f, indent=2)
                
                # Save chunks
                chunks_dir = self.chunks_dir / pdf_path.stem
                chunks_dir.mkdir(exist_ok=True)
                with open(chunks_dir / f"{section_name}_chunks.json", 'w') as f:
                    json.dump(semantic_chunks, f, indent=2)
            
            # Update metadata
            self.metadata["processed_files"].append({
                "filename": pdf_path.name,
                "processed_date": datetime.now().isoformat(),
                "sections": list(result["sections"].keys())
            })
            self._save_metadata()
            
            logger.info(f"Successfully processed {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return False

    def process_all_pdfs(self):
        """Process all PDFs in the data/pdfs directory."""
        pdf_dir = self.base_dir / "pdfs"
        if not pdf_dir.exists():
            logger.error("PDF directory not found")
            return
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            if pdf_file.name not in [f["filename"] for f in self.metadata["processed_files"]]:
                self.process_pdf(pdf_file) 