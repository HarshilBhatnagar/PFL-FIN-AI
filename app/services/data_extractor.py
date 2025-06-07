from typing import Dict, List, Any, Optional
import re
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from app.utils.logger import get_logger
import PyPDF2
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import pdfplumber
import pytesseract
from PIL import Image
import io

logger = get_logger(__name__)

class FinancialDataExtractor:
    """Extracts structured financial data from PDF documents."""
    
    def __init__(self):
        # Patterns for identifying financial sections
        self.section_patterns = {
            "balance_sheet": [
                r"(?i)balance\s+sheet",
                r"(?i)financial\s+position",
                r"(?i)assets\s+and\s+liabilities",
                r"(?i)statement\s+of\s+financial\s+position"
            ],
            "income_statement": [
                r"(?i)income\s+statement",
                r"(?i)profit\s+and\s+loss",
                r"(?i)statement\s+of\s+operations",
                r"(?i)statement\s+of\s+comprehensive\s+income"
            ],
            "cash_flow": [
                r"(?i)cash\s+flow",
                r"(?i)cash\s+and\s+cash\s+equivalents",
                r"(?i)statement\s+of\s+cash\s+flows"
            ],
            "financial_highlights": [
                r"(?i)financial\s+highlights",
                r"(?i)key\s+financial\s+indicators",
                r"(?i)performance\s+highlights",
                r"(?i)financial\s+summary"
            ],
            "notes_to_accounts": [
                r"(?i)notes\s+to\s+accounts",
                r"(?i)notes\s+to\s+financial\s+statements",
                r"(?i)significant\s+accounting\s+policies"
            ]
        }
        
        # Patterns for financial metrics
        self.metric_patterns = {
            "asset_quality": {
                "patterns": [
                    (r"(?i)Gross\s+NPA\s*:?\s*(\d+\.?\d*)%?", "gross_npa"),
                    (r"(?i)Net\s+NPA\s*:?\s*(\d+\.?\d*)%?", "net_npa"),
                    (r"(?i)Provision\s+Coverage\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "pcr"),
                    (r"(?i)Restructured\s+Assets\s*:?\s*(\d+\.?\d*)%?", "restructured_assets"),
                    (r"(?i)Substandard\s+Assets\s*:?\s*(\d+\.?\d*)%?", "substandard_assets"),
                    (r"(?i)Doubtful\s+Assets\s*:?\s*(\d+\.?\d*)%?", "doubtful_assets"),
                    (r"(?i)Loss\s+Assets\s*:?\s*(\d+\.?\d*)%?", "loss_assets"),
                    (r"(?i)SMA\s+[12]\s*:?\s*(\d+\.?\d*)%?", "sma_1_2"),
                    (r"(?i)SMA\s+[0]\s*:?\s*(\d+\.?\d*)%?", "sma_0")
                ]
            },
            "profitability": {
                "patterns": [
                    (r"(?i)Net\s+Profit\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "net_profit"),
                    (r"(?i)ROE\s*:?\s*(\d+\.?\d*)%?", "roe"),
                    (r"(?i)ROA\s*:?\s*(\d+\.?\d*)%?", "roa"),
                    (r"(?i)Operating\s+Profit\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "operating_profit"),
                    (r"(?i)Net\s+Interest\s+Income\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "net_interest_income"),
                    (r"(?i)Net\s+Interest\s+Margin\s*:?\s*(\d+\.?\d*)%?", "nim"),
                    (r"(?i)Operating\s+Margin\s*:?\s*(\d+\.?\d*)%?", "operating_margin"),
                    (r"(?i)Net\s+Margin\s*:?\s*(\d+\.?\d*)%?", "net_margin"),
                    (r"(?i)EPS\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "eps")
                ]
            },
            "capital": {
                "patterns": [
                    (r"(?i)Capital\s+Adequacy\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "car"),
                    (r"(?i)Tier\s+1\s+Capital\s*:?\s*(\d+\.?\d*)%?", "tier1_capital"),
                    (r"(?i)Tier\s+2\s+Capital\s*:?\s*(\d+\.?\d*)%?", "tier2_capital"),
                    (r"(?i)CET1\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "cet1_ratio"),
                    (r"(?i)Total\s+Capital\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "total_capital"),
                    (r"(?i)Risk\s+Weighted\s+Assets\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "rwa")
                ]
            },
            "liquidity": {
                "patterns": [
                    (r"(?i)LCR\s*:?\s*(\d+\.?\d*)%?", "lcr"),
                    (r"(?i)NSFR\s*:?\s*(\d+\.?\d*)%?", "nsfr"),
                    (r"(?i)Current\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "current_ratio"),
                    (r"(?i)Quick\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "quick_ratio"),
                    (r"(?i)Cash\s+Reserve\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "crr"),
                    (r"(?i)Statutory\s+Liquidity\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "slr")
                ]
            },
            "growth": {
                "patterns": [
                    (r"(?i)YoY\s+Growth\s*:?\s*(\d+\.?\d*)%?", "yoy_growth"),
                    (r"(?i)QoQ\s+Growth\s*:?\s*(\d+\.?\d*)%?", "qoq_growth"),
                    (r"(?i)Asset\s+Growth\s*:?\s*(\d+\.?\d*)%?", "asset_growth"),
                    (r"(?i)Loan\s+Growth\s*:?\s*(\d+\.?\d*)%?", "loan_growth"),
                    (r"(?i)Deposit\s+Growth\s*:?\s*(\d+\.?\d*)%?", "deposit_growth"),
                    (r"(?i)Revenue\s+Growth\s*:?\s*(\d+\.?\d*)%?", "revenue_growth"),
                    (r"(?i)Profit\s+Growth\s*:?\s*(\d+\.?\d*)%?", "profit_growth")
                ]
            },
            "efficiency": {
                "patterns": [
                    (r"(?i)Cost\s+to\s+Income\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "cir"),
                    (r"(?i)Operating\s+Efficiency\s+Ratio\s*:?\s*(\d+\.?\d*)%?", "operating_efficiency"),
                    (r"(?i)Employee\s+Productivity\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "employee_productivity"),
                    (r"(?i)Branch\s+Productivity\s*:?\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", "branch_productivity"),
                    (r"(?i)Digital\s+Transactions\s*:?\s*(\d+(?:,\d{3})*)", "digital_transactions")
                ]
            }
        }
        
        # Table detection patterns
        self.table_patterns = {
            "start": r"(?i)(?:^|\n)(?:\s*[-=]+\s*){2,}",
            "end": r"(?i)(?:\s*[-=]+\s*){2,}(?:\n|$)",
            "row": r"(?i)(?:^|\n)(?:\s*\|?\s*[^|\n]+\s*\|?\s*)+"
        }
        
        # Data validation rules
        self.validation_rules = {
            "percentage": {
                "min": 0.0,
                "max": 100.0,
                "message": "Percentage value must be between 0 and 100"
            },
            "amount": {
                "min": -1e15,  # Allow negative values for losses
                "max": 1e15,
                "message": "Amount value is outside reasonable range"
            },
            "ratio": {
                "min": 0.0,
                "max": 1000.0,
                "message": "Ratio value is outside reasonable range"
            }
        }

    def _validate_value(self, value: float, value_type: str) -> bool:
        """Validate a numeric value against rules."""
        if value_type in self.validation_rules:
            rules = self.validation_rules[value_type]
            if not (rules["min"] <= value <= rules["max"]):
                logger.warning(f"{rules['message']}: {value}")
                return False
        return True

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,%₹$€£\-()]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize currency symbols
        text = re.sub(r'[₹$€£]', '₹', text)
        
        # Normalize percentage signs
        text = re.sub(r'%', ' %', text)
        
        return text.strip()

    def extract_structured_data(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract structured data from PDF."""
        try:
            # Read PDF
            text = self._read_pdf(pdf_path)
            
            # Create directory for chunks
            doc_name = pdf_path.stem
            chunks_dir = Path("data/chunks") / doc_name
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract sections
            sections = self._extract_sections(text)
            for section_name, section_chunks in sections.items():
                chunk_file = chunks_dir / f"{section_name}_chunks.json"
                with open(chunk_file, 'w') as f:
                    json.dump(section_chunks, f, indent=2)
            
            # Extract tables
            tables = self._extract_tables(text)
            for i, table_chunks in enumerate(tables):
                chunk_file = chunks_dir / f"table_{i}_chunks.json"
                with open(chunk_file, 'w') as f:
                    json.dump(table_chunks, f, indent=2)
            
            # Extract metrics
            metrics = self._extract_metrics(text)
            for category, metric_chunks in metrics.items():
                chunk_file = chunks_dir / f"{category}_chunks.json"
                with open(chunk_file, 'w') as f:
                    json.dump(metric_chunks, f, indent=2)
            
            # Extract metadata
            metadata = self._extract_metadata(text, pdf_path)
            
            return {
                "sections": sections,
                "tables": tables,
                "metrics": metrics,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            raise

    def _read_pdf(self, pdf_path: Path) -> str:
        """Read text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def _extract_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract sections from text and split into chunks."""
        sections = defaultdict(list)
        current_section = None
        current_text = []
        
        # Split text into lines and process each line
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section
            found_section = False
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Save previous section if exists
                        if current_section and current_text:
                            sections[current_section].extend(self._split_into_chunks('\n'.join(current_text)))
                            current_text = []
                        
                        current_section = section_name
                        current_text = [line]  # Start new section with the header
                        found_section = True
                        break
                if found_section:
                    break
            
            # If no new section found, add line to current section
            if not found_section and current_section:
                current_text.append(line)
        
        # Save the last section if exists
        if current_section and current_text:
            sections[current_section].extend(self._split_into_chunks('\n'.join(current_text)))
        
        return dict(sections)

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of approximately chunk_size characters."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in text.split('\n'):
            line_size = len(line)
            if current_size + line_size > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from text with validation."""
        tables = []
        current_table = []
        in_table = False
        
        for line in text.split('\n'):
            # Check if line is a table header
            if re.search(r'^\s*[-+]+\s*$', line):
                if in_table:
                    # End of table
                    if current_table:
                        table_text = '\n'.join(current_table)
                        chunks = self._split_into_chunks(table_text)
                        tables.append({
                            "text": table_text,
                            "chunks": chunks
                        })
                    current_table = []
                    in_table = False
                else:
                    in_table = True
            elif in_table:
                current_table.append(line)
        
        # Process last table if exists
        if in_table and current_table:
            table_text = '\n'.join(current_table)
            chunks = self._split_into_chunks(table_text)
            tables.append({
                "text": table_text,
                "chunks": chunks
            })
        
        return tables

    def _extract_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract financial metrics from text and split into chunks."""
        metrics = defaultdict(list)
        
        # Process each category of metrics
        for category, category_data in self.metric_patterns.items():
            patterns = category_data["patterns"]
            category_text = []
            
            # Process each line of text
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check each pattern in the category
                for pattern, metric_name in patterns:
                    match = re.search(pattern, line)
                    if match:
                        # Extract the value and context
                        value = match.group(1)
                        context = line.strip()
                        
                        # Create a metric entry with context
                        metric_entry = {
                            "metric": metric_name,
                            "value": value,
                            "context": context
                        }
                        
                        # Add to category text
                        category_text.append(json.dumps(metric_entry))
            
            # If we found any metrics in this category, split into chunks
            if category_text:
                chunks = self._split_into_chunks('\n'.join(category_text))
                metrics[category].extend(chunks)
        
        return dict(metrics)

    def _extract_metadata(self, text: str, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file and text."""
        try:
            # Get file metadata
            file_stats = pdf_path.stat()
            
            # Try to extract date from text
            date_patterns = [
                r"(?i)as\s+at\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                r"(?i)for\s+the\s+period\s+ended\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                r"(?i)year\s+ended\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
            ]
            
            date = None
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    date = match.group(1)
                    break
            
            return {
                "filename": pdf_path.name,
                "size": file_stats.st_size,
                "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "date": date
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}

    def save_structured_data(self, data: Dict[str, Any], output_path: Path):
        """Save extracted structured data to JSON file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved structured data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving structured data to {output_path}: {str(e)}")

    def load_structured_data(self, input_path: Path) -> Dict[str, Any]:
        """Load structured data from JSON file."""
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading structured data from {input_path}: {str(e)}")
            return {}

    def extract_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured data from PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                data = {
                    'text_content': [],
                    'tables': [],
                    'sections': {},
                    'metrics': {}
                }
                all_text = []
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Extract text content
                        data['text_content'].append({
                            'page': page_num,
                            'text': text
                        })
                        all_text.append(text)
                        # Try to detect tables in the text
                        tables = self._detect_tables(text)
                        if tables:
                            for table_num, table in enumerate(tables, 1):
                                data['tables'].append({
                                    'page': page_num,
                                    'table_number': table_num,
                                    'data': table
                                })
                # Use canonical methods on the full text
                joined_text = '\n'.join(all_text)
                data['sections'] = self._extract_sections(joined_text)
                data['metrics'] = self._extract_metrics(joined_text)
                return data
        except Exception as e:
            logger.error(f"Error extracting data from PDF: {str(e)}")
            raise

    def _detect_tables(self, text: str) -> List[List[str]]:
        """Detect tables in text using pattern matching."""
        tables = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Look for patterns that indicate a table
        table_start_patterns = [
            r'^\s*[A-Za-z\s]+\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*',  # Header with numbers
            r'^\s*[A-Za-z\s]+\s+[A-Za-z\s]+\s+\d+\.?\d*',  # Header with text and numbers
            r'^\s*[A-Za-z\s]+\s+\d+\.?\d*$'  # Single column with numbers
        ]
        
        current_table = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches table patterns
            is_table_line = any(re.match(pattern, line) for pattern in table_start_patterns)
            
            if is_table_line:
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            elif in_table:
                # End of table
                if current_table:
                    tables.append(current_table)
                in_table = False
                current_table = []
        
        # Add last table if exists
        if current_table:
            tables.append(current_table)
        
        return tables

    def save_chunks(self, data: Dict[str, Any], output_dir: str) -> None:
        """Save extracted data as chunks."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save text chunks
        for i, content in enumerate(data['text_content']):
            chunk_file = output_path / f"text_chunk_{i+1}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        
        # Save table chunks
        for i, table in enumerate(data['tables']):
            chunk_file = output_path / f"table_chunk_{i+1}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(table, f, ensure_ascii=False, indent=2)
        
        # Save sections
        sections_file = output_path / "sections.json"
        with open(sections_file, 'w', encoding='utf-8') as f:
            json.dump(data['sections'], f, ensure_ascii=False, indent=2)
        
        # Save metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(data['metrics'], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved chunks to {output_dir}") 