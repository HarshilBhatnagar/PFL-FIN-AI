from typing import List, Dict, Any, Tuple, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from app.utils.logger import get_logger
from app.services.llm_setup import llm
import re
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
import heapq
from functools import lru_cache
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = get_logger(__name__)

class ValueNormalizer:
    """Handles normalization of financial values and units."""
    
    UNIT_MULTIPLIERS = {
        'Cr': 1e7,    # Crore
        'L': 1e5,     # Lakh
        'K': 1e3,     # Thousand
        'M': 1e6,     # Million
        'B': 1e9,     # Billion
        'T': 1e12     # Trillion
    }
    
    CURRENCY_SYMBOLS = {
        '₹': 'INR',
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP'
    }
    
    @staticmethod
    def normalize_number(value: str) -> float:
        """Normalize a number string to float."""
        try:
            # Remove currency symbols and commas
            value = re.sub(r'[₹$€£,]', '', value)
            
            # Extract number and unit
            match = re.match(r'([\d.]+)\s*([A-Za-z]*)', value)
            if not match:
                return float(value)
            
            number, unit = match.groups()
            number = float(number)
            
            # Apply unit multiplier if present
            if unit in ValueNormalizer.UNIT_MULTIPLIERS:
                number *= ValueNormalizer.UNIT_MULTIPLIERS[unit]
            
            return number
        except Exception:
            return 0.0
    
    @staticmethod
    def normalize_percentage(value: str) -> float:
        """Normalize a percentage string to float."""
        try:
            return float(re.sub(r'[%]', '', value)) / 100
        except Exception:
            return 0.0
    
    @staticmethod
    def format_value(value: float, unit: str = '') -> str:
        """Format a normalized value back to string."""
        try:
            if unit == '%':
                return f"{value * 100:.2f}%"
            elif unit in ValueNormalizer.UNIT_MULTIPLIERS:
                if value >= 1e12:
                    return f"₹{value/1e12:.2f}T"
                elif value >= 1e9:
                    return f"₹{value/1e9:.2f}B"
                elif value >= 1e7:
                    return f"₹{value/1e7:.2f}Cr"
                elif value >= 1e5:
                    return f"₹{value/1e5:.2f}L"
                else:
                    return f"₹{value:,.2f}"
            else:
                return f"{value:,.2f}"
        except Exception:
            return str(value)

class ContextManager:
    def __init__(self, max_context_items: int = 10):
        self.max_context_items = max_context_items
        self.context_weights = {
            "recent_queries": 0.3,
            "recent_responses": 0.2,
            "financial_metrics": 0.25,
            "temporal_context": 0.15,
            "user_preferences": 0.1
        }
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize value normalizer
        self.value_normalizer = ValueNormalizer()
        
        # Initialize embedding cache
        self._init_embedding_cache()
        
        # Comprehensive financial metric patterns
        self.metric_patterns = {
            "asset_quality": {
                "patterns": [
                    r"\b(GNPA|NNPA|NPA|Gross NPA|Net NPA)\b",
                    r"\b(Provision Coverage Ratio|PCR)\b",
                    r"\b(Asset Quality|Asset Quality Ratio)\b",
                    r"\b(Restructured Assets|SMA|Special Mention Account)\b",
                    r"\b(Substandard Assets|Doubtful Assets|Loss Assets)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "profitability": {
                "patterns": [
                    r"\b(ROE|ROA|Return on Equity|Return on Assets)\b",
                    r"\b(PAT|Profit After Tax|Net Profit)\b",
                    r"\b(NIM|Net Interest Margin)\b",
                    r"\b(Operating Profit|EBIT|EBITDA)\b",
                    r"\b(EPS|Earnings Per Share)\b",
                    r"\b(Dividend Yield|Dividend Payout)\b",
                    r"\b(Operating Margin|Net Margin)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "capital_adequacy": {
                "patterns": [
                    r"\b(CAR|CRAR|Capital Adequacy Ratio)\b",
                    r"\b(Tier 1|Tier 2|Capital)\b",
                    r"\b(Leverage Ratio|LR)\b",
                    r"\b(CET1|Common Equity Tier 1)\b",
                    r"\b(Risk Weighted Assets|RWA)\b",
                    r"\b(Total Capital|Regulatory Capital)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "liquidity": {
                "patterns": [
                    r"\b(LCR|Liquidity Coverage Ratio)\b",
                    r"\b(NSFR|Net Stable Funding Ratio)\b",
                    r"\b(Current Ratio|Quick Ratio)\b",
                    r"\b(Liquid Assets|Liquid Investments)\b",
                    r"\b(Cash Reserve Ratio|CRR)\b",
                    r"\b(Statutory Liquidity Ratio|SLR)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "growth_metrics": {
                "patterns": [
                    r"\b(YoY|Year over Year|Year on Year)\b",
                    r"\b(QoQ|Quarter over Quarter)\b",
                    r"\b(Growth Rate|CAGR|Compound Annual Growth Rate)\b",
                    r"\b(Asset Growth|Loan Growth|Deposit Growth)\b",
                    r"\b(Revenue Growth|Profit Growth)\b",
                    r"\b(Business Growth|Portfolio Growth)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "efficiency_metrics": {
                "patterns": [
                    r"\b(Cost to Income Ratio|CIR)\b",
                    r"\b(Operating Efficiency|Efficiency Ratio)\b",
                    r"\b(Employee Productivity|Branch Productivity)\b",
                    r"\b(Digital Transactions|Digital Banking)\b",
                    r"\b(Transaction Cost|Processing Cost)\b",
                    r"\b(Resource Utilization|Asset Utilization)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "market_metrics": {
                "patterns": [
                    r"\b(Market Share|Market Position)\b",
                    r"\b(Customer Base|Customer Acquisition)\b",
                    r"\b(Brand Value|Brand Equity)\b",
                    r"\b(Market Capitalization|Market Cap)\b",
                    r"\b(Price to Book|P/B Ratio)\b",
                    r"\b(Price to Earnings|P/E Ratio)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            },
            "risk_metrics": {
                "patterns": [
                    r"\b(Credit Risk|Market Risk|Operational Risk)\b",
                    r"\b(Risk Weighted Assets|RWA)\b",
                    r"\b(Value at Risk|VaR)\b",
                    r"\b(Stress Test|Scenario Analysis)\b",
                    r"\b(Risk Adjusted Return|RAROC)\b",
                    r"\b(Risk Management|Risk Controls)\b"
                ],
                "values": [
                    r"\d+\.?\d*\s*%",
                    r"₹\s*\d+(?:,\d{3})*(?:\.\d{2})?",
                    r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
                ]
            }
        }

    def _init_embedding_cache(self):
        """Initialize embedding cache."""
        self.embedding_cache = {}
        self.cache_size = 1000  # Maximum number of cached embeddings
        
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.sentence_transformer.encode([text])[0]
        
        # Update cache
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest item
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        
        self.embedding_cache[text] = embedding
        return embedding

    def extract_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract financial metrics and values from text with enhanced pattern matching."""
        metrics = defaultdict(list)
        
        # Extract metrics by category
        for category, patterns in self.metric_patterns.items():
            # Extract metric names
            for pattern in patterns["patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    metric_name = match.group()
                    # Look for associated values in the surrounding context
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    
                    # Find values near the metric
                    for value_pattern in patterns["values"]:
                        value_matches = re.finditer(value_pattern, context)
                        for value_match in value_matches:
                            value = value_match.group()
                            # Normalize value
                            if '%' in value:
                                normalized_value = self.value_normalizer.normalize_percentage(value)
                                formatted_value = self.value_normalizer.format_value(normalized_value, '%')
                            else:
                                normalized_value = self.value_normalizer.normalize_number(value)
                                formatted_value = self.value_normalizer.format_value(normalized_value)
                            
                            metrics[category].append(f"{metric_name}: {formatted_value}")
        
        return dict(metrics)

    def calculate_context_relevance(self, query: str, context: Dict[str, Any]) -> float:
        """Calculate relevance score for context items using semantic similarity."""
        relevance_score = 0.0
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Check recent queries for relevance
        if "recent_queries" in context:
            for prev_query in context["recent_queries"]:
                # Calculate semantic similarity using embeddings
                prev_embedding = self.get_embedding(prev_query)
                similarity = self._calculate_embedding_similarity(query_embedding, prev_embedding)
                relevance_score += similarity * self.context_weights["recent_queries"]
        
        # Check for metric continuity
        if "financial_metrics" in context:
            query_metrics = self.extract_metrics(query)
            context_metrics = context.get("financial_metrics", {})
            
            for category in query_metrics:
                if category in context_metrics:
                    # Calculate metric similarity
                    metric_similarity = self._calculate_metric_similarity(
                        query_metrics[category],
                        context_metrics[category]
                    )
                    relevance_score += metric_similarity * self.context_weights["financial_metrics"]
        
        # Consider temporal relevance
        if "temporal_context" in context:
            time_diff = datetime.now() - datetime.fromisoformat(context["temporal_context"])
            time_relevance = 1.0 / (1.0 + time_diff.total_seconds() / 3600)  # Decay over hours
            relevance_score += time_relevance * self.context_weights["temporal_context"]
        
        return min(relevance_score, 1.0)

    def _calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0

    def _calculate_metric_similarity(self, metrics1: List[str], metrics2: List[str]) -> float:
        """Calculate similarity between two sets of metrics with normalized values."""
        if not metrics1 or not metrics2:
            return 0.0
        
        # Extract metric names and values
        def parse_metric(metric: str) -> Tuple[str, str]:
            parts = metric.split(":", 1)
            return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
        
        # Parse metrics
        parsed1 = [parse_metric(m) for m in metrics1]
        parsed2 = [parse_metric(m) for m in metrics2]
        
        # Calculate name similarity
        name_similarities = []
        for name1, _ in parsed1:
            for name2, _ in parsed2:
                # Get embeddings for metric names
                emb1 = self.get_embedding(name1)
                emb2 = self.get_embedding(name2)
                similarity = self._calculate_embedding_similarity(emb1, emb2)
                name_similarities.append(similarity)
        
        # Calculate value similarity
        value_similarities = []
        for _, value1 in parsed1:
            for _, value2 in parsed2:
                if value1 and value2:
                    # Normalize values
                    if '%' in value1 and '%' in value2:
                        norm1 = self.value_normalizer.normalize_percentage(value1)
                        norm2 = self.value_normalizer.normalize_percentage(value2)
                    else:
                        norm1 = self.value_normalizer.normalize_number(value1)
                        norm2 = self.value_normalizer.normalize_number(value2)
                    
                    if norm1 != 0 and norm2 != 0:
                        # Calculate relative difference
                        diff = abs(norm1 - norm2) / max(abs(norm1), abs(norm2))
                        value_similarities.append(1 - diff)
        
        # Combine similarities
        name_sim = max(name_similarities) if name_similarities else 0.0
        value_sim = max(value_similarities) if value_similarities else 0.0
        
        # Weight name similarity more than value similarity
        return 0.7 * name_sim + 0.3 * value_sim

    def summarize_context(self, context: Dict[str, Any]) -> str:
        """Generate a concise summary of the context with enhanced metric grouping."""
        summary_parts = []
        
        if "recent_queries" in context and context["recent_queries"]:
            summary_parts.append(f"Previous queries: {', '.join(context['recent_queries'][-2:])}")
        
        if "financial_metrics" in context:
            metrics_by_category = defaultdict(list)
            for category, metrics in context["financial_metrics"].items():
                if metrics:
                    # Group similar metrics
                    unique_metrics = set()
                    for metric in metrics:
                        name = metric.split(":", 1)[0].strip()
                        if name not in unique_metrics:
                            unique_metrics.add(name)
                            metrics_by_category[category].append(metric)
            
            # Create summary for each category
            category_summaries = []
            for category, metrics in metrics_by_category.items():
                if metrics:
                    category_summaries.append(f"{category}: {', '.join(metrics[-2:])}")
            
            if category_summaries:
                summary_parts.append(f"Relevant metrics: {'; '.join(category_summaries)}")
        
        return " | ".join(summary_parts) if summary_parts else "No relevant context"

class Session:
    def __init__(self, session_id: str = None, max_history: int = 10):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.max_history = max_history
        self.history = []
        self.context = {}
        self.context_manager = ContextManager()

    def add_to_history(self, query: str, response: str, relevant_chunks: List[str] = None):
        """Add interaction to session history with enhanced context."""
        # Extract metrics from query and response
        query_metrics = self.context_manager.extract_metrics(query)
        response_metrics = self.context_manager.extract_metrics(response)
        
        # Update context with new information
        self.context.update({
            "temporal_context": datetime.now().isoformat(),
            "financial_metrics": {
                **query_metrics,
                **response_metrics
            }
        })
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "relevant_chunks": relevant_chunks,
            "context": self.context.copy()
        })
        
        # Keep only the last max_history interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.last_accessed = datetime.now()

    def get_relevant_context(self, current_query: str, n: int = 3) -> Dict[str, Any]:
        """Get most relevant context for the current query."""
        if not self.history:
            return {}
        
        # Calculate relevance scores for each history item
        scored_items = []
        for item in self.history:
            relevance = self.context_manager.calculate_context_relevance(
                current_query,
                item.get("context", {})
            )
            scored_items.append((relevance, item))
        
        # Get top n most relevant items
        top_items = heapq.nlargest(n, scored_items, key=lambda x: x[0])
        
        # Combine context from top items
        combined_context = {
            "recent_queries": [],
            "recent_responses": [],
            "financial_metrics": defaultdict(list),
            "temporal_context": datetime.now().isoformat()
        }
        
        for _, item in top_items:
            combined_context["recent_queries"].append(item["query"])
            combined_context["recent_responses"].append(item["response"])
            
            # Merge metrics
            if "context" in item and "financial_metrics" in item["context"]:
                for metric_type, values in item["context"]["financial_metrics"].items():
                    combined_context["financial_metrics"][metric_type].extend(values)
        
        # Convert defaultdict to regular dict
        combined_context["financial_metrics"] = dict(combined_context["financial_metrics"])
        
        return combined_context

    def is_expired(self, max_age: timedelta = timedelta(hours=1)) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_accessed > max_age

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.base_dir = Path("data")
        self.chunks_dir = self.base_dir / "chunks"
        self.index_dir = self.base_dir / "embeddings"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 384  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_metadata = []
        
        # Initialize keyword index
        self.keyword_index = {}
        
        # Session management
        self.sessions: Dict[str, Session] = {}
        self.session_cleanup_interval = timedelta(minutes=30)
        self.last_cleanup = datetime.now()
        
        # Search result cache
        self.search_cache = {}
        self.cache_size = 1000
        self.cache_ttl = timedelta(hours=1)
        
        # Load or create index
        self._load_or_create_index()

    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = datetime.now()
        if current_time - self.last_cleanup > self.session_cleanup_interval:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if session.is_expired()
            ]
            for session_id in expired_sessions:
                del self.sessions[session_id]
            self.last_cleanup = current_time

    def get_or_create_session(self, session_id: str = None) -> Session:
        """Get existing session or create new one."""
        self._cleanup_expired_sessions()
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_accessed = datetime.now()
            return session
        
        session = Session(session_id)
        self.sessions[session.session_id] = session
        return session

    def _get_session_context(self, session: Session) -> Dict[str, Any]:
        """Get context from session history."""
        recent_interactions = session.get_relevant_context(session.history[-1]["query"])
        if not recent_interactions:
            return {}
        
        return {
            "recent_queries": recent_interactions["recent_queries"],
            "recent_responses": recent_interactions["recent_responses"],
            "session_id": session.session_id,
            "session_age": (datetime.now() - session.created_at).total_seconds()
        }

    def _build_keyword_index(self):
        """Build keyword index for text search."""
        self.keyword_index = {}
        
        for i, metadata in enumerate(self.chunk_metadata):
            # Tokenize text
            words = set(re.findall(r'\w+', metadata["text"].lower()))
            
            # Add to index
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = set()
                self.keyword_index[word].add(i)

    def _hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search using both semantic and keyword matching with enhanced scoring."""
        # Check cache
        cache_key = f"{query}_{k}"
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if datetime.now() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["results"]
        
        # Semantic search
        query_embedding = self.model.encode([query])[0]
        semantic_scores, semantic_indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k * 2  # Get more results for better scoring
        )
        
        # Keyword search with enhanced matching
        query_terms = set(re.findall(r'\w+', query.lower()))
        keyword_matches = defaultdict(lambda: {"count": 0, "positions": [], "exact_matches": 0})
        
        # Process each term
        for term in query_terms:
            if term in self.keyword_index:
                for doc_id in self.keyword_index[term]:
                    # Get document text
                    doc_text = self.chunk_metadata[doc_id]["text"].lower()
                    
                    # Find all occurrences
                    positions = [m.start() for m in re.finditer(r'\b' + term + r'\b', doc_text)]
                    
                    if positions:
                        keyword_matches[doc_id]["count"] += 1
                        keyword_matches[doc_id]["positions"].extend(positions)
                        
                        # Check for exact matches (case-insensitive)
                        exact_matches = len(re.findall(r'\b' + term + r'\b', doc_text, re.IGNORECASE))
                        keyword_matches[doc_id]["exact_matches"] += exact_matches
        
        # Calculate keyword scores with position weighting
        keyword_scores = {}
        for doc_id, match_info in keyword_matches.items():
            # Base score from term frequency
            base_score = match_info["count"] / len(query_terms)
            
            # Position score (terms closer together get higher score)
            position_score = 0.0
            if len(match_info["positions"]) > 1:
                positions = sorted(match_info["positions"])
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    position_score = 1.0 / (1.0 + avg_gap/100)  # Normalize gap impact
            
            # Exact match bonus
            exact_match_score = match_info["exact_matches"] / len(query_terms)
            
            # Combine scores
            keyword_scores[doc_id] = 0.4 * base_score + 0.3 * position_score + 0.3 * exact_match_score
        
        # Normalize keyword scores
        max_keyword_score = max(keyword_scores.values()) if keyword_scores else 1
        keyword_scores = {
            doc_id: score / max_keyword_score
            for doc_id, score in keyword_scores.items()
        }
        
        # Combine results with sophisticated scoring
        combined_scores = {}
        
        # Add semantic search results with position weighting
        for score, idx in zip(semantic_scores[0], semantic_indices[0]):
            if idx < len(self.chunk_metadata):
                # Convert distance to similarity score
                semantic_score = 1.0 - (score / max(semantic_scores[0]))
                
                # Position in results affects score
                position_weight = 1.0 - (len(combined_scores) / (k * 2))
                
                combined_scores[idx] = semantic_score * (0.7 + 0.3 * position_weight)
        
        # Add keyword search results with dynamic weighting
        for doc_id, score in keyword_scores.items():
            if doc_id in combined_scores:
                # Adjust weights based on query characteristics
                if len(query_terms) <= 2:  # Short queries favor keyword matching
                    semantic_weight = 0.4
                    keyword_weight = 0.6
                else:  # Longer queries favor semantic matching
                    semantic_weight = 0.7
                    keyword_weight = 0.3
                
                combined_scores[doc_id] = (
                    semantic_weight * combined_scores[doc_id] +
                    keyword_weight * score
                )
            else:
                combined_scores[doc_id] = 0.3 * score  # Lower weight for keyword-only matches
        
        # Apply diversity penalty to avoid similar results
        final_scores = {}
        used_embeddings = []
        
        for doc_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if len(final_scores) >= k:
                break
                
            # Get document embedding
            doc_embedding = self.model.encode([self.chunk_metadata[doc_id]["text"]])[0]
            
            # Calculate diversity penalty
            diversity_penalty = 0.0
            for used_embedding in used_embeddings:
                similarity = self._calculate_embedding_similarity(doc_embedding, used_embedding)
                diversity_penalty += similarity
            
            # Apply penalty
            final_score = score * (1.0 - 0.3 * diversity_penalty)
            final_scores[doc_id] = final_score
            
            # Add to used embeddings
            used_embeddings.append(doc_embedding)
        
        # Sort by final scores
        sorted_results = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Prepare results
        results = []
        for doc_id, score in sorted_results:
            results.append({
                "text": self.chunk_metadata[doc_id]["text"],
                "metadata": {
                    "pdf": self.chunk_metadata[doc_id]["pdf"],
                    "section": self.chunk_metadata[doc_id]["section"],
                    "chunk_index": self.chunk_metadata[doc_id]["chunk_index"],
                    "score": score,
                    "semantic_score": combined_scores[doc_id] if doc_id in combined_scores else 0.0,
                    "keyword_score": keyword_scores[doc_id] if doc_id in keyword_scores else 0.0
                }
            })
        
        # Update cache
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.search_cache.keys(),
                key=lambda k: self.search_cache[k]["timestamp"]
            )
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now()
        }
        
        return results

    def _create_index(self):
        """Create FAISS index from texts in the texts folder."""
        all_chunks = []
        all_metadata = []
        
        # Process all text files in the texts folder
        texts_dir = self.base_dir / "texts"
        if not texts_dir.exists():
            logger.error("Texts directory does not exist.")
            return
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        for text_file in texts_dir.glob("*.txt"):
            try:
                logger.info(f"Processing {text_file.name}")
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Split text into chunks
                chunks = text_splitter.split_text(text)
                logger.info(f"Created {len(chunks)} chunks from {text_file.name}")
                
                # Add chunks and metadata
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "pdf": text_file.stem,
                        "section": "full_text",
                        "chunk_index": i,
                        "text": chunk
                    })
            except Exception as e:
                logger.error(f"Error processing {text_file}: {str(e)}")
                continue
        
        if all_chunks:
            logger.info(f"Creating index with {len(all_chunks)} chunks")
            # Create embeddings
            embeddings = self.model.encode(all_chunks)
            
            # Add to FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)  # Reset index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Save metadata
            self.chunk_metadata = all_metadata
            
            # Build keyword index
            self._build_keyword_index()
            
            # Save indices
            self._save_index()
            logger.info("Index created and saved successfully")
        else:
            logger.warning("No texts found to create index")

    def _save_index(self):
        """Save FAISS index, metadata, and keyword index."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "faiss_index.bin"))
        
        # Save metadata
        with open(self.index_dir / "chunk_metadata.json", 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)
        
        # Save keyword index
        with open(self.index_dir / "keyword_index.json", 'w') as f:
            # Convert sets to lists for JSON serialization
            keyword_index_serializable = {
                k: list(v) for k, v in self.keyword_index.items()
            }
            json.dump(keyword_index_serializable, f, indent=2)

    def _load_or_create_index(self):
        """Load existing indices or create new ones."""
        index_file = self.index_dir / "faiss_index.bin"
        metadata_file = self.index_dir / "chunk_metadata.json"
        keyword_index_file = self.index_dir / "keyword_index.json"
        
        if all(f.exists() for f in [index_file, metadata_file, keyword_index_file]):
            # Load existing indices
            self.index = faiss.read_index(str(index_file))
            with open(metadata_file, 'r') as f:
                self.chunk_metadata = json.load(f)
            with open(keyword_index_file, 'r') as f:
                # Convert lists back to sets
                self.keyword_index = {
                    k: set(v) for k, v in json.load(f).items()
                }
        else:
            # Create new indices
            self._create_index()

    def query_vectorstore(self, query: str) -> List[str]:
        """Query the vector store for relevant chunks."""
        try:
            logger.info(f"Querying vector store with query: {query}")
            # Perform hybrid search
            results = self._hybrid_search(query, k=3)  # Get top 3 results
            logger.info(f"Hybrid search returned {len(results)} results")
            if not results:
                logger.warning("No results found for query")
                return []
            
            # Extract text from results
            relevant_chunks = []
            for result in results:
                if result["text"].strip():
                    relevant_chunks.append(result["text"])
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found after filtering")
                return []
            
            logger.info(f"Returning {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []

    def _extract_temporal_context(self, query: str) -> Dict[str, Any]:
        """Extract temporal context from query."""
        temporal_context = {
            "time_period": None,
            "comparison_period": None,
            "is_comparison": False
        }
        
        # Patterns for time periods
        time_patterns = {
            "quarter": r"(?:Q[1-4]|quarter)\s*(?:of\s*)?(\d{4})",
            "year": r"(?:year|FY|fiscal year)\s*(?:of\s*)?(\d{4})",
            "month": r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(\d{4})",
            "date": r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
        }
        
        # Patterns for comparisons
        comparison_patterns = {
            "vs": r"vs\.?\s*(?:Q[1-4]|quarter|year|FY|fiscal year)\s*(?:of\s*)?(\d{4})",
            "compared_to": r"compared\s+to\s*(?:Q[1-4]|quarter|year|FY|fiscal year)\s*(?:of\s*)?(\d{4})",
            "versus": r"versus\s*(?:Q[1-4]|quarter|year|FY|fiscal year)\s*(?:of\s*)?(\d{4})"
        }
        
        # Extract time period
        for period_type, pattern in time_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if period_type == "date":
                    day, month, year = match.groups()
                    temporal_context["time_period"] = {
                        "type": "date",
                        "value": f"{year}-{month}-{day}"
                    }
                else:
                    temporal_context["time_period"] = {
                        "type": period_type,
                        "value": match.group(1)
                    }
                break
        
        # Extract comparison period
        for comp_type, pattern in comparison_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                temporal_context["comparison_period"] = {
                    "type": "year",
                    "value": match.group(1)
                }
                temporal_context["is_comparison"] = True
                break
        
        return temporal_context

    def _filter_by_temporal_context(self, results: List[Dict[str, Any]], 
                                  temporal_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter search results based on temporal context."""
        if not temporal_context["time_period"]:
            return results
        
        filtered_results = []
        for result in results:
            metadata = result["metadata"]
            
            # Extract date from metadata
            date = None
            if "date" in metadata:
                date = metadata["date"]
            elif "quarter" in metadata:
                date = metadata["quarter"]
            elif "year" in metadata:
                date = metadata["year"]
            
            if date:
                # Convert dates to comparable format
                if isinstance(date, str):
                    try:
                        date = datetime.strptime(date, "%Y-%m-%d")
                    except ValueError:
                        try:
                            date = datetime.strptime(date, "%Y")
                        except ValueError:
                            continue
                
                # Check if date matches temporal context
                if temporal_context["time_period"]["type"] == "date":
                    target_date = datetime.strptime(
                        temporal_context["time_period"]["value"],
                        "%Y-%m-%d"
                    )
                    if date == target_date:
                        filtered_results.append(result)
                elif temporal_context["time_period"]["type"] == "year":
                    if date.year == int(temporal_context["time_period"]["value"]):
                        filtered_results.append(result)
                elif temporal_context["time_period"]["type"] == "quarter":
                    year = int(temporal_context["time_period"]["value"])
                    quarter = int(temporal_context["time_period"]["type"][1])
                    if date.year == year and (date.month - 1) // 3 + 1 == quarter:
                        filtered_results.append(result)
        
        return filtered_results if filtered_results else results

    def _enhance_context(self, query: str, results: List[Dict[str, Any]], 
                        session: Session) -> Dict[str, Any]:
        """Enhance context with additional information."""
        enhanced_context = {
            "temporal_context": self._extract_temporal_context(query),
            "financial_context": self._extract_financial_context(query),
            "session_context": session.get_relevant_context(query),
            "metadata_context": self._extract_metadata_context(results)
        }
        
        return enhanced_context

    def _extract_financial_context(self, query: str) -> Dict[str, Any]:
        """Extract financial context from query."""
        financial_context = {
            "metrics": [],
            "categories": [],
            "comparisons": []
        }
        
        # Extract metric names
        for category, patterns in self.context_manager.metric_patterns.items():
            for pattern in patterns["patterns"]:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    metric_name = match.group()
                    financial_context["metrics"].append(metric_name)
                    financial_context["categories"].append(category)
        
        # Extract comparisons
        comparison_patterns = [
            r"(\d+\.?\d*)\s*(?:times|%)\s+(?:higher|lower|better|worse)",
            r"(?:increase|decrease|growth|decline)\s+of\s+(\d+\.?\d*)%",
            r"(\d+\.?\d*)\s*(?:times|%)\s+(?:more|less)"
        ]
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                financial_context["comparisons"].append({
                    "value": float(match.group(1)),
                    "type": "percentage" if "%" in match.group() else "ratio"
                })
        
        return financial_context

    def _extract_metadata_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata context from search results."""
        metadata_context = {
            "sources": set(),
            "sections": set(),
            "dates": set(),
            "metrics": set()
        }
        
        for result in results:
            metadata = result["metadata"]
            
            if "pdf" in metadata:
                metadata_context["sources"].add(metadata["pdf"])
            if "section" in metadata:
                metadata_context["sections"].add(metadata["section"])
            if "date" in metadata:
                metadata_context["dates"].add(metadata["date"])
            if "metrics" in metadata:
                metadata_context["metrics"].update(metadata["metrics"])
        
        # Convert sets to lists for JSON serialization
        return {
            k: list(v) for k, v in metadata_context.items()
        }

    def generate_answer(self, query: str, relevant_chunks: List[str], 
                       session_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate answer using LLM with enhanced context and temporal support."""
        try:
            # Get or create session
            session = self.get_or_create_session(session_id)
            
            # Extract temporal context
            temporal_context = self._extract_temporal_context(query)
            
            # Filter results by temporal context
            filtered_chunks = self._filter_by_temporal_context(
                [{"text": chunk, "metadata": {}} for chunk in relevant_chunks],
                temporal_context
            )
            
            # Get relevant context
            session_context = session.get_relevant_context(query)
            
            # Enhance context
            enhanced_context = self._enhance_context(query, filtered_chunks, session)
            
            # Merge contexts
            merged_context = {
                **session_context,
                **(context or {}),
                **enhanced_context
            }
            
            # Prepare context
            context_str = "\n\n".join(chunk["text"] for chunk in filtered_chunks)
            
            # Generate context summary
            context_summary = session.context_manager.summarize_context(merged_context)
            
            # Prepare prompt with enhanced context
            prompt = f"""Based on the following financial information and conversation history, answer the query.
            
Query: {query}

Context:
{context_str}

Temporal Context:
{json.dumps(temporal_context, indent=2)}

Financial Context:
{json.dumps(enhanced_context["financial_context"], indent=2)}

Conversation History:
{context_summary}

Additional Context:
{json.dumps(merged_context, indent=2) if merged_context else 'None'}

Provide a clear, concise answer focusing on the financial aspects. If the information is not available in the context, say so."""

            # Generate response
            response = llm(
                prompt,
                max_tokens=500,
                temperature=0.3,
                stop=["\n\n"],
                echo=False
            )
            
            answer = response["choices"][0]["text"].strip()
            
            # Add to session history
            session.add_to_history(query, answer, [chunk["text"] for chunk in filtered_chunks])
            
            return {
                "answer": answer,
                "session_id": session.session_id,
                "relevant_chunks": [chunk["text"] for chunk in filtered_chunks],
                "context_summary": context_summary,
                "temporal_context": temporal_context,
                "financial_context": enhanced_context["financial_context"]
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query.",
                "session_id": session_id,
                "error": str(e)
            }

    def update_index(self):
        """Update the FAISS index with new chunks."""
        self._create_index()
        logger.info("Index updated successfully")
