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
from plotly.subplots import make_subplots

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

def parse_ascii_table(table_text: str) -> pd.DataFrame:
    """Parse an ASCII table into a pandas DataFrame."""
    try:
        # Split into lines and remove empty lines
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        # Find the separator line (contains | and -)
        separator_idx = next(i for i, line in enumerate(lines) if '|' in line and '-' in line)
        
        # Get headers
        headers = [h.strip() for h in lines[separator_idx-1].split('|') if h.strip()]
        
        # Get data rows
        data_rows = []
        for line in lines[separator_idx+1:]:
            if '|' in line:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(row) == len(headers):
                    data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='ignore')
        
        return df
    except Exception as e:
        logger.error(f"Error parsing ASCII table: {str(e)}")
        return pd.DataFrame()

def select_relevant_data(df: pd.DataFrame, query: str) -> Tuple[str, List[str]]:
    """Select relevant columns for plotting."""
    try:
        # Find year/date column
        date_cols = [col for col in df.columns if any(year in col.lower() for year in ['2024', '2025'])]
        if date_cols:
            x_col = date_cols[0]
        else:
            x_col = df.columns[0]
            
        # Find value columns
        value_cols = []
        for col in df.columns:
            if col != x_col and pd.api.types.is_numeric_dtype(df[col]):
                value_cols.append(col)
                
        return x_col, value_cols
    except Exception as e:
        logger.error(f"Error selecting relevant data: {str(e)}")
        return df.columns[0], []

def score_table_for_query(df: pd.DataFrame, query: str, main_phrase: str) -> float:
    """Score how well a table matches the query."""
    try:
        score = 0.0
        
        # Check column names
        for col in df.columns:
            col_lower = col.lower()
            # Exact match
            if main_phrase in col_lower:
                score += 1.0
            # Partial match
            elif any(word in col_lower for word in main_phrase.split()):
                score += 0.5
            # Check for year/date columns
            if any(year in col_lower for year in ['2024', '2025']):
                score += 0.3
                
        # Check for numerical data
        if df.select_dtypes(include=['float64', 'int64']).shape[1] > 0:
            score += 0.5
            
        return score
    except Exception as e:
        logger.error(f"Error scoring table: {str(e)}")
        return 0.0

def extract_main_metric_phrase(query: str) -> str:
    """Extract the main metric phrase from the query."""
    # Common financial metrics
    metrics = [
        "revenue", "income", "profit", "expense", "cost", "margin",
        "ratio", "growth", "trend", "fees", "commission", "interest"
    ]
    
    # Find the first metric mentioned in the query
    query_lower = query.lower()
    for metric in metrics:
        if metric in query_lower:
            return metric
    return ""

def select_best_row(df: pd.DataFrame, main_phrase: str) -> pd.DataFrame:
    """Select the best row from the DataFrame based on the query."""
    try:
        if df.empty:
            return df
            
        # If we have a metric column, filter by it
        if 'metric' in df.columns:
            return df[df['metric'].str.contains(main_phrase, case=False, na=False)]
            
        # Otherwise, try to find the best row
        best_score = -1
        best_row = None
        
        for idx, row in df.iterrows():
            score = 0
            for val in row:
                if isinstance(val, str):
                    if main_phrase in val.lower():
                        score += 1
                    elif any(word in val.lower() for word in main_phrase.split()):
                        score += 0.5
            if score > best_score:
                best_score = score
                best_row = row
                
        if best_row is not None:
            return pd.DataFrame([best_row], columns=df.columns)
            
        return df
    except Exception as e:
        logger.error(f"Error selecting best row: {str(e)}")
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
            
            # Save as HTML first
            html_path = filepath.replace('.png', '.html')
            fig.write_html(html_path)
            
            # Convert to PNG
            fig.write_image(filepath)
            
            logger.info(f"Graph saved successfully at {filepath}")
            
            # Generate embed code
            embed_code = f'<iframe src="/graphs/{filename}" width="800" height="500" frameborder="0"></iframe>'
            
            return filepath, embed_code
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
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

    def generate_graph(self, query: str, context: List[str], context_docs=None, separated_data=None) -> Dict[str, Any]:
        """Generate a graph dynamically from numerical data and tables."""
        try:
            logger.info(f"Generating graph for query: {query}")
            main_phrase = extract_main_metric_phrase(query)
            
            # Process numerical data first
            numerical_data = separated_data.get("numerical_data", []) if separated_data else []
            tables = separated_data.get("tables", []) if separated_data else []
            
            # Combine numerical data into a DataFrame
            numerical_df = None
            if numerical_data:
                numerical_rows = []
                for data in numerical_data:
                    text = data["text"]
                    numbers = data["numbers"]
                    # Try to extract labels from the text
                    labels = re.findall(r'[A-Za-z]+(?:\s+[A-Za-z]+)*', text)
                    if labels and len(labels) == len(numbers):
                        for label, number in zip(labels, numbers):
                            numerical_rows.append({
                                "metric": label.strip(),
                                "value": float(number)
                            })
                if numerical_rows:
                    numerical_df = pd.DataFrame(numerical_rows)
                    logger.info(f"Created numerical DataFrame with {len(numerical_rows)} rows")
            
            # Process tables
            best_table_df = None
            best_score = -1
            
            if tables:
                for table_data in tables:
                    df = table_data["dataframe"]
                    score = score_table_for_query(df, query, main_phrase)
                    logger.info(f"Table columns: {df.columns.tolist()} | Score: {score}")
                    if score > best_score:
                        best_score = score
                        best_table_df = df
            
            # Decide which data to use
            if best_table_df is not None and (numerical_df is None or best_score > 0.7):
                logger.info("Using table data for graph generation")
                df = best_table_df
            elif numerical_df is not None:
                logger.info("Using numerical data for graph generation")
                df = numerical_df
            else:
                # Fallback to original extraction method
                logger.info("No suitable data found, using original extraction method")
                df = self._extract_data_from_context(context, query)
                if df.empty and context_docs:
                    logger.info("No relevant table in top-k context, searching all table chunks in document.")
                    all_table_chunks = self.get_all_table_chunks_from_docs(context_docs)
                    df = self._extract_data_from_context(all_table_chunks, query)
            
            if df is None or df.empty:
                return {"error": "No relevant data found for graphing.", "graph_data": None, "embed_code": None}

            # Select relevant data for plotting
            if "metric" in df.columns and "value" in df.columns:
                # Numerical data format
                x_col = "metric"
                y_cols = ["value"]
            else:
                # Table format
                filtered_df = select_best_row(df, main_phrase)
                if filtered_df.empty:
                    return {"error": "No relevant row found for graphing.", "graph_data": None, "embed_code": None}
                x_col, y_cols = select_relevant_data(filtered_df, query)
                df = filtered_df

            if not y_cols:
                return {"error": "No relevant columns found for graphing.", "graph_data": None, "embed_code": None}

            logger.info(f"Plotting x_col: {x_col}, y_cols: {y_cols}")
            fig = go.Figure()
            
            # Determine the best chart type based on the data
            if len(y_cols) > 1:
                # Multiple metrics - use bar chart
                for y_col in y_cols:
                    fig.add_trace(go.Bar(
                        x=df[x_col],
                        y=pd.to_numeric(df[y_col], errors='coerce'),
                        name=y_col
                    ))
            else:
                # Single metric - use line chart for trends
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=pd.to_numeric(df[y_cols[0]], errors='coerce'),
                    mode='lines+markers',
                    name=y_cols[0]
                ))

            fig.update_layout(
                title=f"{query.title()} (from {'numerical data' if numerical_df is not None else 'table data'})",
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
                "graph_type": "bar" if len(y_cols) > 1 else "line",
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