from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from services.llm_setup import llm
from app.utils.logger import get_logger

import os

logger = get_logger(__name__)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup vectorstore (FAISS)
def create_vectorstore_from_text(raw_text):
    logger.info("Creating vectorstore from text")
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(raw_text)
    logger.debug(f"Split text into {len(docs)} chunks")

    # Convert to LangChain Document format
    documents = [Document(page_content=chunk) for chunk in docs]

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embedding_model)
    logger.info("Created FAISS vectorstore")

    # Save index
    vectorstore.save_local("models/faiss_index")
    logger.info("Saved vectorstore to models/faiss_index")

    return vectorstore

def query_vectorstore(query, k=3):
    logger.info(f"Querying vectorstore with: {query}")
    # Load existing FAISS index
    vectorstore = FAISS.load_local("models/faiss_index", embedding_model)

    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=k)
    logger.debug(f"Found {len(docs)} relevant chunks")

    # Return top results' content
    return [doc.page_content for doc in docs]

def generate_answer(query, context_chunks):
    logger.info(f"Generating answer for query: {query}")
    # Create prompt with context
    context = "\n".join(context_chunks)
    prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information, please answer the following question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Question: {query}
Answer:"""

    # Generate response using LLM
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["Question:", "\n\n"],
        echo=False
    )
    logger.debug("Generated response from LLM")

    return response["choices"][0]["text"].strip() 