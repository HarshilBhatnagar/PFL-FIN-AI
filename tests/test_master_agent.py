import pytest
from app.services.master_agent import MasterAgent
from unittest.mock import patch, MagicMock
import json

@pytest.fixture
def master_agent():
    return MasterAgent()

def test_intent_analysis(master_agent):
    # Test query intent
    query = "What is the current NPA ratio?"
    intent = master_agent._analyze_intent(query)
    assert intent["intent"] == "query"
    assert "needs_visualization" in intent
    assert "time_period" in intent

    # Test graph intent
    query = "Show me the trend of NPA ratio over the last 5 years"
    intent = master_agent._analyze_intent(query)
    assert intent["intent"] == "graph"
    assert intent["needs_visualization"] == True
    assert "visualization_type" in intent

    # Test report intent
    query = "Generate a quarterly financial report for HDFC Bank"
    intent = master_agent._analyze_intent(query)
    assert intent["intent"] == "report"
    assert "time_period" in intent
    assert intent["time_period"] == "quarterly"

    # Test analysis intent
    query = "Analyze the profitability trends of ICICI Bank"
    intent = master_agent._analyze_intent(query)
    assert intent["intent"] == "analysis"
    assert "analysis_type" in intent
    assert intent["analysis_type"] in ["trend", "comparison", "forecast"]

def test_metric_extraction(master_agent):
    # Test NPA extraction
    text = "The Gross NPA ratio increased to 5.2%"
    metrics = master_agent._extract_metrics(text)
    assert "npa" in metrics

    # Test profit extraction
    text = "Net Profit After Tax was Rs. 1000 crores"
    metrics = master_agent._extract_metrics(text)
    assert "profit" in metrics

    # Test multiple metrics
    text = "Total Assets under Management and Capital Adequacy Ratio"
    metrics = master_agent._extract_metrics(text)
    assert "assets" in metrics
    assert "capital" in metrics

    # Test no metrics
    text = "The bank performed well in Q3"
    metrics = master_agent._extract_metrics(text)
    assert len(metrics) == 0

def test_query_handling(master_agent):
    query = "What is the current NPA ratio?"
    result = master_agent._handle_query(query, {})
    assert "response" in result
    assert result["graph_data"] is None

def test_report_handling(master_agent):
    query = "Generate a quarterly financial report"
    result = master_agent._handle_report(query, {
        "time_period": "quarterly",
        "data_points": ["npa", "profit"]
    })
    assert "response" in result
    assert isinstance(result["response"], str)

def test_graph_handling(master_agent):
    query = "Show NPA trend"
    result = master_agent._handle_graph(query, {
        "visualization_type": "line",
        "time_period": "annual"
    })
    assert "response" in result
    assert "graph_data" in result
    if result["graph_data"]:
        assert "type" in result["graph_data"]
        assert "datasets" in result["graph_data"]

def test_analysis_handling(master_agent):
    query = "Analyze profitability trends"
    result = master_agent._handle_analysis(query, {
        "analysis_type": "trend",
        "time_period": "annual"
    })
    assert "response" in result
    assert isinstance(result["response"], str)

def test_process_query_chain_of_thought(master_agent):
    query = "Show me the NPA trend and analyze its impact on profitability"
    
    # Mock the LLM responses
    with patch('app.services.master_agent.llm') as mock_llm:
        # Mock intent analysis
        mock_llm.return_value = {
            "choices": [{
                "text": json.dumps({
                    "intent": "analysis",
                    "data_points": ["npa", "profit"],
                    "time_period": "annual",
                    "needs_visualization": True,
                    "visualization_type": "line",
                    "analysis_type": "trend"
                })
            }]
        }
        
        result = master_agent.process_query(query)
        assert "response" in result
        assert isinstance(result["response"], str)

def test_error_handling(master_agent):
    # Test with invalid query
    query = ""
    result = master_agent.process_query(query)
    assert "response" in result
    assert result["response"] is not None

    # Test with invalid intent
    with patch('app.services.master_agent.llm') as mock_llm:
        mock_llm.return_value = {
            "choices": [{
                "text": "invalid json"
            }]
        }
        result = master_agent.process_query("test query")
        assert "response" in result
        assert result["response"] is not None 