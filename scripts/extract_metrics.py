import re
import sqlite3
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

def setup_database():
    db_path = Path("data/finance.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create metrics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT,
        year INTEGER,
        quarter INTEGER,
        metric_name TEXT,
        value REAL,
        unit TEXT,
        source_file TEXT
    )
    """)
    
    conn.commit()
    return conn

def extract_metrics():
    # Setup database
    conn = setup_database()
    cursor = conn.cursor()
    
    # Regular expressions for common metrics
    patterns = {
        'revenue': r'(?:revenue|total income|total revenue)[^\d]*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|cr|million|billion)?',
        'profit': r'(?:net profit|profit after tax|PAT)[^\d]*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|cr|million|billion)?',
        'assets': r'(?:total assets|assets under management|AUM)[^\d]*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|cr|million|billion)?'
    }
    
    # Process each raw text file
    raw_dir = Path("data/raw")
    for text_file in raw_dir.glob("*.txt"):
        logger.info(f"Extracting metrics from {text_file.name}")
        
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Extract company name from filename
            company = text_file.stem.split("_")[0]
            
            # Extract year and quarter if present in filename
            year_match = re.search(r'(\d{4})', text_file.stem)
            quarter_match = re.search(r'Q([1-4])', text_file.stem)
            
            year = int(year_match.group(1)) if year_match else None
            quarter = int(quarter_match.group(1)) if quarter_match else None
            
            # Extract metrics
            for metric_name, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = float(match.group(1).replace(',', ''))
                    
                    # Insert into database
                    cursor.execute("""
                    INSERT INTO financial_metrics 
                    (company, year, quarter, metric_name, value, source_file)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (company, year, quarter, metric_name, value, text_file.name))
            
            conn.commit()
            logger.info(f"Processed metrics for {company}")
            
        except Exception as e:
            logger.error(f"Error processing {text_file.name}: {str(e)}")
    
    conn.close()

if __name__ == "__main__":
    extract_metrics() 