# Financial RAG Chatbot

A sophisticated RAG (Retrieval-Augmented Generation) based chatbot specialized in financial document analysis, capable of both text-based responses and dynamic graph generation.

## Features

- **Document Processing**: Handles various financial documents including ASCII tables
- **Smart Chunking**: Preserves table structures and maintains context
- **Dual-Mode Querying**: 
  - Text-based responses with source citations
  - Dynamic graph generation for data visualization
- **Context-Aware**: Maintains conversation context and document relevance
- **Robust Table Handling**: Specialized processing for financial tables and metrics

## Project Structure

```
financial-rag-chatbot/
├── src/
│   ├── rag_engine.py      # Core RAG implementation
│   ├── agents.py          # Query and Graph agents
│   ├── graph_utils.py     # Graph generation utilities
│   └── __init__.py
├── scripts/
│   ├── ingest_texts.py    # Document ingestion script
│   └── query.py          # Query testing script
├── requirements.txt       # Project dependencies
└── run.py                # Main application entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-rag-chatbot.git
cd financial-rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ingest documents:
```bash
python scripts/ingest_texts.py --file path/to/your/document.txt
```

2. Run the chatbot:
```bash
python run.py
```

3. Query examples:
- Text query: "What were the total fees for 2024?"
- Graph query: "Plot fees and commission income for 2024 and 2025"

## Features in Detail

### Document Processing
- Handles various document formats
- Preserves table structures
- Smart chunking for optimal context retrieval

### Query Processing
- Natural language understanding
- Context-aware responses
- Source citation and transparency

### Graph Generation
- Dynamic graph creation based on query intent
- Support for various chart types
- Accurate data extraction from tables

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Python and modern NLP technologies
- Utilizes advanced RAG techniques for accurate financial analysis
- Implements best practices in document processing and visualization
