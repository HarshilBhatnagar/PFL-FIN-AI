from fastapi import APIRouter, Request
from pydantic import BaseModel
from services.rag_engine import query_vectorstore, generate_answer
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

class ChatRequest(BaseModel):
    query: str

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    query = req.query
    logger.info(f"Received chat request: {query}")
    
    relevant_chunks = query_vectorstore(query)
    logger.debug(f"Found {len(relevant_chunks)} relevant chunks")

    if not relevant_chunks:
        logger.warning("No relevant chunks found for query")
        return {"query": query, "response": "No relevant information found."}

    answer = generate_answer(query, relevant_chunks)
    logger.info("Generated response successfully")

    return {
        "query": query,
        "context": relevant_chunks,
        "response": answer
    } 