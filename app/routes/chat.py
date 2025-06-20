from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.rag_engine import RAGEngine
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Initialize RAG engine
rag_engine = RAGEngine()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    relevant_chunks: Optional[List[str]] = None
    error: Optional[str] = None
    agent_used: str
    agent_type: str
    confidence: float
    context_used: List[str]
    processing_time: float
    status: str

@router.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests and return responses based on the RAG engine."""
    logger.info(f"Received chat request: {request.query}")
    try:
        # Query the RAG engine
        response = rag_engine.query_vectorstore(request.query)
        logger.debug(f"Found {len(response)} relevant chunks")
        if not response:
            logger.warning("No relevant chunks found for query")
            return ChatResponse(
                answer="No relevant information found.",
                session_id=request.session_id or "",
                relevant_chunks=[],
                agent_used="query",
                agent_type="text",
                confidence=0.0,
                context_used=[],
                processing_time=0.0,
                status="ok"
            )
        answer = rag_engine.generate_answer(
            query=request.query,
            relevant_chunks=response,
            session_id=request.session_id,
            context=request.context
        )
        logger.info("Generated response successfully")
        return ChatResponse(
            answer=answer,
            session_id=request.session_id or "",
            relevant_chunks=response,
            agent_used="query",
            agent_type="text",
            confidence=0.9,
            context_used=response,
            processing_time=0.0,
            status="ok"
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with session management."""
    try:
        # Get relevant chunks
        relevant_chunks = rag_engine.query_vectorstore(request.query)
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I couldn't find any relevant information to answer your query.",
                session_id=request.session_id or "",
                relevant_chunks=[],
                agent_used="query",
                agent_type="text",
                confidence=0.0,
                context_used=[],
                processing_time=0.0,
                status="error"
            )
        
        # Generate answer with session management
        response = rag_engine.generate_answer(
            query=request.query,
            relevant_chunks=relevant_chunks,
            session_id=request.session_id,
            context=request.context
        )
        
        return ChatResponse(
            answer=response,
            session_id=request.session_id or "",
            relevant_chunks=relevant_chunks,
            agent_used="query",
            agent_type="text",
            confidence=0.9,
            context_used=relevant_chunks,
            processing_time=0.0,
            status="ok"
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        ) 