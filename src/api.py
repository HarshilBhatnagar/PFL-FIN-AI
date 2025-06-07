from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import logging
import sys
from rag_engine import RAGEngine
from agents import MasterAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount the graphs directory
app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")

# Session management
class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.conversation_history = []

# In-memory session storage
sessions: Dict[str, Session] = {}

# Initialize RAG engine and master agent
rag_engine = RAGEngine()
master_agent = MasterAgent(rag_engine)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    agent_used: str
    agent_type: str
    confidence: float
    context_used: list
    graph_data: Optional[Dict[str, Any]] = None
    embed_code: Optional[str] = None
    filepath: Optional[str] = None
    processing_time: float
    status: str

def get_session(session_id: Optional[str] = None) -> Session:
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.last_activity = datetime.now()
        return session
    
    # Create new session if none exists
    session = Session()
    sessions[session.session_id] = session
    return session

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query and return the response."""
    try:
        # Get or create session
        session = get_session(request.session_id)
        
        # Process the query using master agent
        result = master_agent.process_query(request.query)
        
        # Update session with conversation history
        session.conversation_history.append({
            "query": request.query,
            "response": result["response"],
            "agent_used": result["agent_used"],
            "agent_type": result["agent_type"],
            "confidence": result["confidence"],
            "graph_data": result.get("graph_data"),
            "embed_code": result.get("embed_code"),
            "filepath": result.get("filepath"),
            "timestamp": datetime.now().isoformat()
        })
        session.last_activity = datetime.now()
        
        # Return response with session ID
        return {
            "response": result["response"],
            "session_id": session.session_id,
            "agent_used": result["agent_used"],
            "agent_type": result["agent_type"],
            "confidence": result["confidence"],
            "context_used": result["context_used"],
            "graph_data": result.get("graph_data"),
            "embed_code": result.get("embed_code"),
            "filepath": result.get("filepath"),
            "processing_time": result["processing_time"],
            "status": result["status"]
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "conversation_history": session.conversation_history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 