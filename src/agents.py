from typing import Dict, List, Any
import logging
from datetime import datetime
from graph_utils import GraphGenerator
import pandas as pd
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class DataSeparatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("data_separator")
        self.agent_type = "data_processing"
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Separate text and numerical data from the input context."""
        context = input_data.get("context", [])
        if not context:
            return None
            
        separated_data = {
            "text_data": [],
            "numerical_data": [],
            "tables": []
        }
        
        for doc in context:
            content = doc.page_content
            
            # Check if content is a table
            if '|' in content:
                try:
                    df = pd.read_csv(pd.StringIO(content), sep='|', skipinitialspace=True)
                    separated_data["tables"].append({
                        "content": content,
                        "dataframe": df
                    })
                except:
                    # If table parsing fails, treat as text
                    separated_data["text_data"].append(content)
            else:
                # Extract numerical data using regex
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
                if numbers:
                    separated_data["numerical_data"].append({
                        "text": content,
                        "numbers": numbers
                    })
                else:
                    separated_data["text_data"].append(content)
        
        return {
            "agent": self.name,
            "agent_type": self.agent_type,
            "confidence": 0.9,
            "separated_data": separated_data
        }

class TableFormatterAgent(BaseAgent):
    def __init__(self):
        super().__init__("table_formatter")
        self.agent_type = "formatting"
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format query results into a table structure."""
        query = input_data.get("query", "")
        context = input_data.get("context", [])
        
        if not context:
            return None
            
        # Extract relevant data
        relevant_data = []
        for doc in context:
            content = doc.page_content
            if '|' in content:
                try:
                    df = pd.read_csv(pd.StringIO(content), sep='|', skipinitialspace=True)
                    relevant_data.append(df)
                except:
                    continue
        
        if not relevant_data:
            return None
            
        # Combine relevant data
        combined_df = pd.concat(relevant_data, ignore_index=True)
        
        # Format as markdown table
        markdown_table = combined_df.to_markdown(index=False)
        
        return {
            "agent": self.name,
            "agent_type": self.agent_type,
            "confidence": 0.9,
            "formatted_table": markdown_table,
            "dataframe": combined_df
        }

class QueryAgent(BaseAgent):
    def __init__(self, rag_engine):
        super().__init__("query")
        self.rag_engine = rag_engine
        self.agent_type = "text"
        self.supported_queries = [
            "what", "how", "when", "where", "why", "who",
            "show", "tell", "explain", "describe", "list"
        ]
        self.data_separator = DataSeparatorAgent()
        self.table_formatter = TableFormatterAgent()

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "").lower()
        
        # Check if query is text-based
        if any(term in query for term in self.supported_queries):
            self.logger.info(f"Query agent processing: {query}")
            
            # Get context and generate response using RAG engine
            context = self.rag_engine._get_relevant_context(query)
            response = self.rag_engine._generate_response(query, context)
            
            # Separate data using DataSeparatorAgent
            separated_data = self.data_separator.process({"context": context})
            
            # Check if table formatting is requested
            if "table" in query or "format" in query:
                formatted_table = self.table_formatter.process({
                    "query": query,
                    "context": context
                })
                if formatted_table:
                    response += "\n\nFormatted Table:\n" + formatted_table["formatted_table"]
            
            return {
                "agent": self.name,
                "agent_type": self.agent_type,
                "confidence": 0.9,
                "response": response,
                "context_used": context,
                "separated_data": separated_data.get("separated_data") if separated_data else None
            }
        return None

class GraphAgent(BaseAgent):
    def __init__(self, rag_engine):
        super().__init__("graph")
        self.rag_engine = rag_engine
        self.agent_type = "visualization"
        self.graph_generator = GraphGenerator()
        self.supported_queries = [
            "graph", "chart", "plot", "visualize", "trend",
            "compare", "relationship", "correlation", "analysis",
            "show me", "display", "illustrate", "line", "bar",
            "pie", "scatter", "histogram", "time series"
        ]
        self.data_separator = DataSeparatorAgent()

    def should_handle_query(self, query: str) -> bool:
        """Enhanced query detection for graph-related requests."""
        query_lower = query.lower()
        
        # Check for explicit graph-related terms
        if any(term in query_lower for term in self.supported_queries):
            return True
            
        # Check for implicit graph indicators
        graph_indicators = [
            "over time", "across", "between", "versus", "vs",
            "for the year", "for the period", "trend", "comparison",
            "how has", "how did", "show the", "show me the"
        ]
        if any(indicator in query_lower for indicator in graph_indicators):
            return True
            
        # Check for time-based queries that would benefit from visualization
        time_indicators = ["year", "month", "quarter", "period", "2024", "2025"]
        if any(indicator in query_lower for indicator in time_indicators):
            # If query contains time indicators and numerical metrics, it's likely a graph request
            metric_indicators = ["profit", "revenue", "income", "expense", "ratio", "percentage"]
            if any(metric in query_lower for metric in metric_indicators):
                return True
                
        return False

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "").lower()
        
        # Use enhanced query detection
        if self.should_handle_query(query):
            self.logger.info(f"Graph agent processing: {query}")
            
            try:
                # Get context from RAG engine
                context_docs = self.rag_engine._get_relevant_context(query)
                
                # First, separate data using DataSeparatorAgent
                separated_data = self.data_separator.process({"context": context_docs})
                
                if not separated_data:
                    return {
                        "agent": self.name,
                        "agent_type": self.agent_type,
                        "confidence": 0.0,
                        "response": "No data found to process.",
                        "graph_data": None,
                        "embed_code": None
                    }
                
                # Extract numerical data and tables
                numerical_data = separated_data.get("separated_data", {}).get("numerical_data", [])
                tables = separated_data.get("separated_data", {}).get("tables", [])
                
                if not numerical_data and not tables:
                    return {
                        "agent": self.name,
                        "agent_type": self.agent_type,
                        "confidence": 0.0,
                        "response": "No numerical data found for graph generation.",
                        "graph_data": None,
                        "embed_code": None
                    }
                
                # Convert Document objects to strings for logging
                context = []
                for doc in context_docs:
                    content = doc.page_content
                    self.logger.info(f"Context document: {content[:200]}...")
                    context.append(content)
                
                # Generate graph using only the numerical data
                graph_result = self.graph_generator.generate_graph(
                    query, 
                    context, 
                    context_docs=context_docs,
                    separated_data={
                        "numerical_data": numerical_data,
                        "tables": tables
                    }
                )
                
                if "error" in graph_result:
                    self.logger.error(f"Graph generation error: {graph_result['error']}")
                    return {
                        "agent": self.name,
                        "agent_type": self.agent_type,
                        "confidence": 0.0,
                        "response": f"Error generating graph: {graph_result['error']}",
                        "graph_data": None,
                        "embed_code": None
                    }
                
                self.logger.info(f"Graph generated successfully: {graph_result['graph_type']}")
                
                # Include information about the data used
                data_summary = {
                    "numerical_data_points": len(numerical_data),
                    "tables_processed": len(tables),
                    "data_types": ["numerical", "tabular"] if numerical_data and tables else 
                                 ["numerical"] if numerical_data else ["tabular"]
                }
                
                return {
                    "agent": self.name,
                    "agent_type": self.agent_type,
                    "confidence": 0.9,
                    "response": f"Generated {graph_result['graph_type']} graph for your query using {', '.join(data_summary['data_types'])} data",
                    "graph_data": graph_result["graph_data"],
                    "embed_code": graph_result["embed_code"],
                    "filepath": graph_result["filepath"],
                    "context_used": context,
                    "data_summary": data_summary
                }
            except Exception as e:
                self.logger.error(f"Error in graph generation: {str(e)}")
                return {
                    "agent": self.name,
                    "agent_type": self.agent_type,
                    "confidence": 0.0,
                    "response": f"Error processing graph request: {str(e)}",
                    "graph_data": None,
                    "embed_code": None
                }
        return None

class MasterAgent:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.agents = [
            QueryAgent(rag_engine),
            GraphAgent(rag_engine)
        ]
        self.logger = logging.getLogger("agent.master")
        # Store last context used by QueryAgent
        self._last_query_context = None
        self._last_query_text = None

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the master agent and specialized agents.
        Returns a response with agent attribution and confidence scores.
        """
        start_time = datetime.now()
        input_data = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        agent_responses = []
        
        # First, determine which agent should handle the query
        graph_agent = next((agent for agent in self.agents if agent.name == "graph"), None)
        query_agent = next((agent for agent in self.agents if agent.name == "query"), None)
        
        # Check if it's a graph query first
        if graph_agent and graph_agent.should_handle_query(query):
            self.logger.info("Query detected as graph request")
            response = graph_agent.process(input_data)
            if response:
                agent_responses.append(response)
        # If not a graph query, use query agent
        elif query_agent:
            self.logger.info("Query detected as text request")
            response = query_agent.process(input_data)
            if response:
                agent_responses.append(response)
                # Store context for potential future use
                self._last_query_context = response.get("context_used", [])
                self._last_query_text = query
        
        final_response = self._select_best_response(agent_responses)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": final_response["response"],
            "agent_used": final_response["agent"],
            "agent_type": final_response["agent_type"],
            "confidence": final_response["confidence"],
            "context_used": final_response.get("context_used", []),
            "graph_data": final_response.get("graph_data", {}),
            "embed_code": final_response.get("embed_code"),
            "filepath": final_response.get("filepath"),
            "processing_time": processing_time,
            "status": "success"
        }

    def _select_best_response(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best response based on confidence scores.
        In the future, this could be more sophisticated with ensemble methods.
        """
        if not responses:
            return {
                "agent": "master",
                "agent_type": "fallback",
                "confidence": 0.0,
                "response": "I couldn't find a suitable response for your query."
            }
        
        # Sort by confidence and return the highest
        return sorted(responses, key=lambda x: x["confidence"], reverse=True)[0] 