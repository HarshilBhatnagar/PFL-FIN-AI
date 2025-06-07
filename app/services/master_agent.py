from typing import List, Dict, Any
from app.utils.logger import get_logger
from services.rag_engine import query_vectorstore, generate_answer
from services.llm_setup import llm
import json
import re

logger = get_logger(__name__)

class MasterAgent:
    def __init__(self):
        self.agents = {
            "query": self._handle_query,
            "report": self._handle_report,
            "graph": self._handle_graph,
            "analysis": self._handle_analysis
        }
        
        # Define financial metrics patterns
        self.metric_patterns = {
            # Asset Quality Metrics
            "npa": r"NPA|Non-Performing Assets|Gross NPA|Net NPA|Non-Performing Loan|NPL",
            "provision": r"Provision|Provisioning Coverage|PCR|Provision Coverage Ratio",
            "restructured": r"Restructured Assets|Restructured Loans|SMA|Special Mention Account",
            
            # Profitability Metrics
            "profit": r"Profit|Net Income|PAT|Profit After Tax|PBT|Profit Before Tax|ROA|Return on Assets|ROE|Return on Equity",
            "revenue": r"Revenue|Income|Total Income|Interest Income|Non-Interest Income|Fee Income",
            "margin": r"Margin|NIM|Net Interest Margin|Spread|Cost to Income Ratio",
            
            # Capital Metrics
            "capital": r"Capital|Capital Adequacy|CAR|Capital to Risk|Tier 1|Tier 2|Common Equity|CET1",
            "leverage": r"Leverage|Leverage Ratio|Debt to Equity|Gearing Ratio",
            
            # Asset Metrics
            "assets": r"Assets|Total Assets|Assets Under Management|AUM|Loan Book|Advances|Deposits",
            "investment": r"Investment|Securities|Bonds|Equity|Mutual Funds",
            
            # Liability Metrics
            "liability": r"Liabilities|Deposits|Borrowings|Debt|Subordinated Debt",
            "deposit": r"Deposits|CASA|Current Account|Savings Account|Term Deposits",
            
            # Growth Metrics
            "growth": r"Growth|YoY|Year on Year|QoQ|Quarter on Quarter|CAGR|Compound Annual Growth",
            "market_share": r"Market Share|Market Position|Ranking",
            
            # Efficiency Metrics
            "efficiency": r"Efficiency|Cost to Income|Operating Efficiency|Productivity",
            "employee": r"Employee|Staff|Headcount|Productivity per Employee",
            
            # Risk Metrics
            "risk": r"Risk|Risk Weighted Assets|RWA|Credit Risk|Market Risk|Operational Risk",
            "liquidity": r"Liquidity|LCR|Liquidity Coverage Ratio|NSFR|Net Stable Funding Ratio",
            
            # Valuation Metrics
            "valuation": r"Valuation|P/E|Price to Earnings|P/B|Price to Book|Market Cap",
            "dividend": r"Dividend|Dividend Yield|Payout Ratio|Dividend per Share",
            
            # Business Metrics
            "business": r"Business|Customer|Client|Account|Branch|Network|Digital",
            "product": r"Product|Service|Offering|Portfolio|Mix"
        }

    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine intent and required actions."""
        prompt = f"""You are an expert-level financial assistant designed to interpret user queries related to business and finance. Your task is to analyze the following financial query and return a clear, structured breakdown of the requirements.
 
Please extract and return the following information **accurately and in detail**:
 
1. **Primary Intent**:  
   - Determine the main goal of the query.  
   - Categorize the intent clearly as one of the following: 
     - "Information Query" (the user is asking for specific data or values)
     - "Report Request" (the user is requesting a formatted summary or detailed report)
     - "Graph or Visualization" (the user wants the data presented visually)
     - "Financial Analysis" (the user wants patterns, insights, or conclusions drawn from data)
     - "Other" (if none of the above apply—explain clearly)
 
2. **Required Data Points and Financial Metrics**:  
   - List **all** the metrics, dimensions, or KPIs mentioned or implied (e.g., revenue, net profit margin, YoY growth, cost center, ROI, department, region, etc.)  
   - Be specific—do not generalize. If a user asks for “sales trends in South India,” list “sales,” “region (South India),” and “trend over time” as separate items.
 
3. **Time Period Referenced**:  
   - Identify the **exact time period** requested or implied in the query (e.g., Q1 FY24, last 3 months, since Jan 2020, year-over-year, etc.)  
   - If no time period is mentioned, write “Not specified.”
 
4. **Visualization Requirement**:  
   - Determine whether the user expects or implies a **visual output**, such as a chart, graph, dashboard, or table.  
   - Return “Yes,” “No,” or “Possibly,” and explain briefly.
 
5. **Type of Financial Analysis Required** (if applicable):  
   - Identify what type of analysis is expected:  
     - Descriptive (e.g., summary stats)
     - Comparative (e.g., comparing regions or time periods)
     - Trend/Time Series (e.g., over months or years)
     - Variance (e.g., actual vs. target or budget)
     - Forecasting
     - Ratio Analysis
     - Root Cause Analysis
     - Any other — specify clearly
   - If no analysis is required, state “None.”
 
Now, analyze the following user query as per the above structure, and format your output under each numbered section:

Query: {query}

Respond in JSON format:
{{
    "intent": "query/report/graph/analysis",
    "data_points": ["metric1", "metric2"],
    "time_period": "quarterly/annual/ytd",
    "needs_visualization": true/false,
    "visualization_type": "bar/line/pie/table",
    "analysis_type": "trend/comparison/forecast"
}}"""

        response = llm(
            prompt,
            max_tokens=300,
            temperature=0.1,
            stop=["\n\n"],
            echo=False
        )
        
        try:
            return json.loads(response["choices"][0]["text"].strip())
        except:
            logger.error("Failed to parse intent analysis")
            return {
                "intent": "query",
                "data_points": [],
                "time_period": "annual",
                "needs_visualization": False,
                "analysis_type": "trend"
            }

    def _extract_metrics(self, text: str) -> List[str]:
        """Extract financial metrics from text."""
        metrics = []
        for metric_name, pattern in self.metric_patterns.items():
            if re.search(pattern, text, re.I):
                metrics.append(metric_name)
        return metrics

    def _handle_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simple query resolution with enhanced context."""
        relevant_chunks = query_vectorstore(query)
        if not relevant_chunks:
            return {
                "response": "No relevant information found.",
                "graph_data": None
            }

        # Extract metrics from query
        metrics = self._extract_metrics(query)
        
        # Generate answer with metric context
        answer = generate_answer(
            query,
            relevant_chunks,
            context={"metrics": metrics, "time_period": context.get("time_period", "annual")}
        )
        
        return {
            "response": answer,
            "graph_data": None
        }

    def _handle_report(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation with structured data."""
        # Extract required metrics
        metrics = self._extract_metrics(query)
        
        # Get relevant data chunks
        relevant_chunks = query_vectorstore(query)
        
        # Generate report structure
        report_prompt = f"""Generate a financial report based on the following:
Query: {query}
Metrics: {metrics}
Time Period: {context.get('time_period', 'annual')}

Use the following data:
{relevant_chunks}

Format the response as a structured report with sections for each metric."""

        report = llm(
            report_prompt,
            max_tokens=500,
            temperature=0.2,
            stop=["\n\n"],
            echo=False
        )
        
        return {
            "response": report["choices"][0]["text"].strip(),
            "graph_data": None
        }

    def _handle_graph(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph generation with data visualization."""
        # Extract metrics and time period
        metrics = self._extract_metrics(query)
        time_period = context.get("time_period", "annual")
        
        # Get relevant data
        relevant_chunks = query_vectorstore(query)
        
        # Generate graph data
        graph_prompt = f"""Generate graph data for the following:
Query: {query}
Metrics: {metrics}
Time Period: {time_period}
Visualization Type: {context.get('visualization_type', 'line')}

Use the following data:
{relevant_chunks}

Respond in JSON format:
{{
    "type": "line/bar/pie",
    "title": "Graph Title",
    "labels": ["label1", "label2"],
    "data": [value1, value2],
    "datasets": [
        {{
            "label": "Metric Name",
            "data": [value1, value2]
        }}
    ]
}}"""

        try:
            graph_data = json.loads(llm(
                graph_prompt,
                max_tokens=300,
                temperature=0.1,
                stop=["\n\n"],
                echo=False
            )["choices"][0]["text"].strip())
            
            return {
                "response": f"Generated {context.get('visualization_type', 'line')} graph for {', '.join(metrics)}",
                "graph_data": graph_data
            }
        except:
            logger.error("Failed to generate graph data")
            return {
                "response": "Failed to generate graph data.",
                "graph_data": None
            }

    def _handle_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle financial analysis with insights."""
        metrics = self._extract_metrics(query)
        relevant_chunks = query_vectorstore(query)
        
        analysis_prompt = f"""Perform financial analysis based on:
Query: {query}
Metrics: {metrics}
Analysis Type: {context.get('analysis_type', 'trend')}
Time Period: {context.get('time_period', 'annual')}

Data:
{relevant_chunks}

Provide analysis with:
1. Key findings
2. Trends
3. Insights
4. Recommendations"""

        analysis = llm(
            analysis_prompt,
            max_tokens=500,
            temperature=0.2,
            stop=["\n\n"],
            echo=False
        )
        
        return {
            "response": analysis["choices"][0]["text"].strip(),
            "graph_data": None
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the appropriate agent with enhanced chain-of-thought."""
        logger.info(f"Processing query: {query}")
        
        # Step 1: Analyze intent
        intent_analysis = self._analyze_intent(query)
        logger.info(f"Intent analysis: {intent_analysis}")
        
        # Step 2: Extract metrics
        metrics = self._extract_metrics(query)
        logger.info(f"Extracted metrics: {metrics}")
        
        # Step 3: Route to appropriate agent
        agent = self.agents.get(intent_analysis["intent"], self._handle_query)
        response = agent(query, {**intent_analysis, "metrics": metrics})
        
        logger.info("Query processed successfully")
        return response 