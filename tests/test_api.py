from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
 
def test_chat_api():
    response = client.post("/api/chat", json={"query": "What is net profit in Q1?"})
    assert response.status_code == 200
    assert "response" in response.json() 