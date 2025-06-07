import requests
from app.utils.logger import get_logger

logger = get_logger(__name__)

class OllamaLLM:
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def __call__(self, prompt, max_tokens=500, temperature=0.7, stop=None, echo=False):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        if stop:
            payload["options"]["stop"] = stop
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return {"choices": [{"text": result["response"]}]}
        except Exception as e:
            logger.error(f"Ollama LLM error: {e}")
            return {"choices": [{"text": "Error: LLM not available."}]}

llm = OllamaLLM(model="llama3") 