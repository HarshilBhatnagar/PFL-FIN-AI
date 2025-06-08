# RAG Chatbot with Graph Generation

A sophisticated chatbot that combines Retrieval-Augmented Generation (RAG) with graph visualization capabilities. The system can process queries, generate text responses, and create visualizations of data when appropriate.

## Features

- **RAG-based Query Processing**: Uses LangChain and ChromaDB for efficient document retrieval and response generation
- **Graph Generation**: Automatically generates visualizations for data-related queries
- **Multi-Agent Architecture**: 
  - Master Agent for query routing
  - Graph Agent for visualization
  - Query Agent for text responses
- **Interactive API**: FastAPI-based REST API for easy integration

## Prerequisites

- Python 3.8+
- Ollama (for LLM)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
- Download from https://ollama.ai/download
- Install and start the Ollama service
- Pull the required model:
```bash
ollama pull llama2
```

## Usage

1. Start the API server:
```bash
python src/api.py
```

2. The API will be available at `http://localhost:8000`

3. Example API endpoints:
- POST `/query`: Process a query
- GET `/sessions/{session_id}`: Get conversation history

## Project Structure

```
.
├── src/
│   ├── api.py           # FastAPI server
│   ├── agents.py        # Agent implementations
│   ├── rag_engine.py    # RAG engine
│   └── graph_utils.py   # Graph generation utilities
├── data/               # Document storage
├── graphs/            # Generated graph storage
└── requirements.txt   # Project dependencies
```

## Configuration

The system uses several configuration parameters that can be adjusted:
- LLM settings in `rag_engine.py`
- Graph generation parameters in `graph_utils.py`
- API settings in `api.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
