import os
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import base64
from io import BytesIO
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_utils.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_ascii_table(table_str: str) -> pd.DataFrame:
    """Parse an ASCII table (with | separators) into a pandas DataFrame."""
    lines = [line for line in table_str.split('\n') if '|' in line]
    if not lines:
        return pd.DataFrame()
    # Remove border/separator lines
    lines = [line for line in lines if not re.match(r'^[|\-\s]+$', line)]
    # Find the header row (the one with the most columns)
    split_rows = [line.split('|')[1:-1] for line in lines]
    header_idx = max(range(len(split_rows)), key=lambda i: len(split_rows[i]))
    header = [cell.strip() for cell in split_rows[header_idx]]
    # Collect data rows that match header length
    data = []
    for i, row in enumerate(split_rows):
        if i == header_idx:
            continue
        if len(row) == len(header):
            data.append([cell.strip() for cell in row])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=header)
    return df

def select_relevant_data(df: pd.DataFrame, query: str):
    """Select relevant columns based on the query."""
    query_lower = query.lower()
    # If years are in columns, use them as y, and the first column as x
    year_cols = [col for col in df.columns if re.search(r'20\d{2}', col)]
    if year_cols:
        x_col = df.columns[0]
        y_cols = year_cols
        return x_col, y_cols
    # Otherwise, try to find columns matching the query
    y_cols = [col for col in df.columns if any(word in col.lower() for word in query_lower.split())]
    if y_cols:
        x_col = df.columns[0]
        return x_col, y_cols
    # Fallback: all numeric columns except the first
    numeric_cols = []
    for col in df.columns[1:]:
        try:
            df[col] = df[col].str.replace(',', '').astype(float)
            numeric_cols.append(col)
        except Exception:
            continue
    if numeric_cols:
        return df.columns[0], numeric_cols
    return df.columns[0], []

def score_table_for_query(df: pd.DataFrame, query: str, main_phrase: str) -> int:
    """Score a table for relevance to the query, prioritizing exact phrase and header matches."""
    score = 0
    # Score exact phrase in headers
    for col in df.columns:
        if main_phrase in col.lower():
            score += 20
    # Score exact phrase in rows
    for _, row in df.iterrows():
        for cell in row:
            if main_phrase in str(cell).lower():
                score += 10
    return score

def extract_main_metric_phrase(query: str) -> str:
    # Try to extract a known metric phrase from the query
    known_metrics = [
        "fees and commission income", "total income", "total assets", "revenue from operations",
        "interest income", "profit before tax", "net profit", "operating expenses", "total liabilities"
    ]
    query_lower = query.lower()
    for metric in known_metrics:
        if metric in query_lower:
            return metric
    # Fallback: extract the longest noun phrase or quoted phrase
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    if quoted:
        return quoted[0][0] or quoted[0][1]
    # Otherwise, use the longest sequence of non-stopwords
    stopwords = set(['the', 'for', 'and', 'of', 'in', 'to', 'as', 'on', 'at', 'by', 'with', 'from', 'a', 'an', 'is', 'was', 'were', 'are', 'be', 'this', 'that', 'these', 'those', 'it', 'its'])
    words = [w for w in query_lower.split() if w not in stopwords]
    return ' '.join(words)

def select_best_row(df: pd.DataFrame, main_phrase: str) -> pd.DataFrame:
    first_col = df.columns[0]
    # Normalize row labels
    candidates = df[first_col].astype(str).str.lower().str.replace('&', 'and').str.replace(r'\s+', ' ', regex=True).str.strip().tolist()
    logger.info(f'Parsed DataFrame for matching:\n{df}')
    logger.info(f'Row candidates for matching: {candidates}')
    # Normalize the metric phrase
    norm_phrase = main_phrase.lower().replace('&', 'and').replace('  ', ' ').strip()
    matches = difflib.get_close_matches(norm_phrase, candidates, n=1, cutoff=0.5)
    if matches:
        idx = candidates.index(matches[0])
        logger.info(f'Fuzzy matched row: {matches[0]} (index {idx}) for metric: {main_phrase}')
        return df.iloc[[idx]]
    logger.info('No fuzzy match found for metric, returning full table.')
    return df

class GraphGenerator:
    def __init__(self):
        self.graphs_dir = "graphs"
        os.makedirs(self.graphs_dir, exist_ok=True)
        logger.info(f"GraphGenerator initialized with directory: {self.graphs_dir}")

    def _save_graph(self, fig: go.Figure, graph_type: str) -> Tuple[str, str]:
        """Save graph as PNG and return file path and embed code."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{graph_type}_{timestamp}.png"
            filepath = os.path.join(self.graphs_dir, filename)
            
            logger.info(f"Attempting to save graph to: {filepath}")
            
            # Convert figure to PNG bytes
            img_bytes = fig.to_image(format="png")
            
            # Save the bytes to file
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            
            logger.info(f"Graph saved successfully at {filepath}")
            
            # Generate base64 encoded image for embed code
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            embed_code = f'<img src="data:image/png;base64,{img_base64}" alt="{graph_type} graph">'
            
            return filepath, embed_code
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            # If saving fails, try to return just the base64 image
            try:
                img_bytes = fig.to_image(format="png")
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                embed_code = f'<img src="data:image/png;base64,{img_base64}" alt="{graph_type} graph">'
                return None, embed_code
            except Exception as e2:
                logger.error(f"Error generating base64 image: {str(e2)}")
                raise

    def get_all_table_chunks_from_docs(self, context_docs):
        # context_docs: List[Document]
        table_chunks = []
        for doc in context_docs:
            if '|' in doc.page_content:
                table_chunks.append(doc.page_content)
        return table_chunks

    def _extract_data_from_context(self, context: List[str], query: str) -> pd.DataFrame:
        """Extract the most relevant table from context for graphing."""
        try:
            logger.info("Extracting table data from context for graphing")
            main_phrase = extract_main_metric_phrase(query)
            logger.info(f"Main metric phrase for matching: '{main_phrase}'")
            best_score = -1
            best_df = pd.DataFrame()
            for chunk in context:
                if '|' in chunk:  # likely a table
                    df = parse_ascii_table(chunk)
                    # Only consider tables with at least 2 columns and 2 rows
                    if df.empty or len(df.columns) < 2 or len(df) < 2:
                        logger.info(f"Skipping table with shape {df.shape}")
                        continue
                    score = score_table_for_query(df, query, main_phrase)
                    logger.info(f"Table columns: {df.columns.tolist()} | Score: {score}")
                    if score > best_score:
                        best_score = score
                        best_df = df
            if not best_df.empty and best_score > 0:
                logger.info(f"Selected table with columns: {best_df.columns.tolist()} and score: {best_score}")
                return best_df
            logger.warning("No relevant table found in context for graphing")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error extracting table from context: {str(e)}")
            return pd.DataFrame()

    def generate_graph(self, query: str, context: List[str], context_docs=None) -> Dict[str, Any]:
        """Generate a graph dynamically from the most relevant table and row found in context. Fallback to all table chunks if needed."""
        try:
            logger.info(f"Generating graph for query: {query}")
            main_phrase = extract_main_metric_phrase(query)
            df = self._extract_data_from_context(context, query)
            if df.empty and context_docs:
                logger.info("No relevant table in top-k context, searching all table chunks in document.")
                all_table_chunks = self.get_all_table_chunks_from_docs(context_docs)
                df = self._extract_data_from_context(all_table_chunks, query)
            if df.empty:
                return {"error": "No relevant table data found in context for graphing.", "graph_data": None, "embed_code": None}

            # Stage 2: Select the best-matching row for the metric/label
            filtered_df = select_best_row(df, main_phrase)
            if filtered_df.empty:
                return {"error": "No relevant row found for graphing.", "graph_data": None, "embed_code": None}

            x_col, y_cols = select_relevant_data(filtered_df, query)
            if not y_cols:
                return {"error": "No relevant columns found for graphing.", "graph_data": None, "embed_code": None}

            logger.info(f"Plotting x_col: {x_col}, y_cols: {y_cols}")
            fig = go.Figure()
            for y_col in y_cols:
                fig.add_trace(go.Bar(x=filtered_df[x_col], y=pd.to_numeric(filtered_df[y_col], errors='coerce'), name=y_col))

            fig.update_layout(
                title=f"{query.title()} (from table)",
                xaxis_title=x_col,
                yaxis_title=", ".join(y_cols),
                template='plotly_white',
                height=500,
                width=800,
                showlegend=True
            )

            filepath, embed_code = self._save_graph(fig, "table_graph")
            logger.info(f"Graph generated and saved successfully at {filepath}")
            graph_json = json.loads(fig.to_json())
            return {
                "graph_type": "bar",
                "graph_data": graph_json,
                "embed_code": embed_code,
                "filepath": filepath
            }
        except Exception as e:
            logger.error(f"Error generating graph: {str(e)}")
            return {
                "error": str(e),
                "graph_data": None,
                "embed_code": None
            } 