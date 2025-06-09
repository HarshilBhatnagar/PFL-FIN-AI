from typing import Dict, List, Any
import logging
from datetime import datetime
from graph_utils import GraphGenerator
import pandas as pd
import re
import json
from langchain.schema import Document

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
        """Initialize the master agent with specialized agents."""
        self.rag_engine = rag_engine
        self.logger = logging.getLogger("agent.master")
        self.query_history = []
        
        # Initialize specialized agents
        self.agents = {
            "text": QueryAgent(rag_engine),
            "graph": GraphAgent(rag_engine)
        }
        
        self.logger.info(f"MasterAgent initialized with {len(self.agents)} specialized agents")

    def _get_llm_agent_selection(self, query: str, context: List[Document]) -> Dict[str, float]:
        """
        Use LLM to determine which agent should handle the query.
        Returns a dictionary of agent types and their confidence scores.
        """
        self.logger.info("Getting LLM agent selection for query: %s", query)
        
        # Prepare context for LLM
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Create prompt for LLM
        prompt = f"""You are an expert at determining whether a query requires a graph visualization or text response.
        Analyze the following query and context to decide which type of agent should handle it.

        Query: {query}

        Context:
        {context_text}

        IMPORTANT RULES:
        1. If the query contains ANY of these words/phrases, ALWAYS choose graph agent:
           - "show the graph"
           - "plot"
           - "visualize"
           - "chart"
           - "trend"
           - "compare"
           - "over time"
           - "for the years"
           - "between"
           - "versus"
           - "vs"
           - "across"

        2. If the query asks about:
           - Trends over time
           - Comparisons between values
           - Visual representation of data
           - Changes over periods
           -> Choose graph agent

        3. If the query asks for:
           - Detailed explanations
           - Specific values
           - Text descriptions
           - Individual data points
           -> Choose text agent

        Return a JSON object with confidence scores (0.0 to 1.0) for each agent type.
        Example: {{"graph": 0.9, "text": 0.1}}

        Your response:"""

        try:
            # Get response from LLM
            response = self.rag_engine.llm.invoke(prompt)
            self.logger.debug("LLM response for agent selection: %s", response)
            
            # Parse the response to get scores
            try:
                scores = json.loads(response)
                # Ensure we have both scores
                if "graph" not in scores:
                    scores["graph"] = 0.0
                if "text" not in scores:
                    scores["text"] = 0.0
                    
                # Normalize scores
                total = sum(scores.values())
                if total > 0:
                    scores = {k: v/total for k, v in scores.items()}
                    
                self.logger.info("LLM agent selection scores: %s", scores)
                return scores
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM response as JSON: %s", response)
                # Fallback to equal distribution
                return {"graph": 0.5, "text": 0.5}
                
        except Exception as e:
            self.logger.error("Error getting LLM agent selection: %s", str(e))
            # Fallback to equal distribution
            return {"graph": 0.5, "text": 0.5}

    def analyze_query(self, query: str) -> Dict[str, float]:
        """
        Analyze the query using LLM to determine the most suitable agent.
        """
        self.logger.info("Starting query analysis for: %s", query)
        
        # Get context from RAG engine
        self.logger.debug("Retrieving context from RAG engine")
        context = self.rag_engine._get_relevant_context(query)
        self.logger.info("Retrieved %d context documents", len(context))
        
        # Use LLM to determine agent type
        scores = self._get_llm_agent_selection(query, context)
        
        self.logger.info("Final query analysis scores: %s", scores)
        return scores

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the master agent and specialized agents.
        Uses LLM to select the most appropriate agent.
        """
        self.logger.info("Starting query processing: %s", query)
        start_time = datetime.now()
        input_data = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store query in history
        self.query_history.append(query)
        self.logger.debug("Added query to history. History size: %d", len(self.query_history))
        
        # Analyze query to determine best agent using LLM
        query_scores = self.analyze_query(query)
        best_agent_type = max(query_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info("Selected agent type: %s with confidence %.2f", 
                        best_agent_type, query_scores[best_agent_type])
        
        # Get the appropriate agent
        selected_agent = self.agents.get(best_agent_type)
        
        if not selected_agent:
            self.logger.warning("No agent found for type %s, falling back to QueryAgent", best_agent_type)
            selected_agent = self.agents["text"]
        
        self.logger.info("Processing with agent: %s", selected_agent.name)
        
        # Process with selected agent
        response = selected_agent.process(input_data)
        
        if not response:
            self.logger.warning("Selected agent returned no response, trying fallback agents")
            # Try other agents as fallback
            for agent_type, agent in self.agents.items():
                if agent_type != best_agent_type:
                    self.logger.debug("Trying fallback agent: %s", agent.name)
                    response = agent.process(input_data)
                    if response:
                        self.logger.info("Fallback agent %s succeeded", agent.name)
                        break
        
        if not response:
            self.logger.error("No response generated from any agent")
            return {
                "response": "I couldn't find a suitable response for your query.",
                "agent_used": "master",
                "agent_type": "fallback",
                "confidence": 0.0,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "status": "error"
            }
        
        # Add processing metadata
        response.update({
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "query_analysis": query_scores,
            "status": "success"
        })
        
        self.logger.info("Query processing completed in %.2f seconds", 
                        (datetime.now() - start_time).total_seconds())
        return response

    def _select_best_response(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best response based on confidence scores and query analysis.
        """
        self.logger.debug("Selecting best response from %d responses", len(responses))
        
        if not responses:
            self.logger.warning("No responses available for selection")
            return {
                "agent": "master",
                "agent_used": "master",
                "agent_type": "fallback",
                "confidence": 0.0,
                "response": "I couldn't find a suitable response for your query."
            }
        
        # Sort by confidence and return the highest
        best_response = sorted(responses, key=lambda x: x["confidence"], reverse=True)[0]
        
        # Ensure agent_used field is present
        if "agent_used" not in best_response:
            best_response["agent_used"] = best_response.get("agent", "unknown")
            
        self.logger.info("Selected response with confidence %.2f from agent %s",
                        best_response["confidence"], best_response["agent_used"])
        return best_response 