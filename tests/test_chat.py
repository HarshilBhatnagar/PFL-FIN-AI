from services.rag_engine import query_vectorstore, generate_answer

def test_rag_response():
    query = "What is the total revenue for FY 2023?"
    chunks = query_vectorstore(query)
    assert chunks, "No chunks found for the query"

    response = generate_answer(query, chunks)
    assert isinstance(response, str) and len(response) > 0 