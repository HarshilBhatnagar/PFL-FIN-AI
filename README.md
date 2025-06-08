#Directory format:
.
├── .git/
├── .gitignore
├── README.md
├── rag_engine.log
├── graph_utils.log
├── requirements.txt
├── agents.log
├── api.log
├── query.log
├── ingestion.log
├── run.py
├── data/
├── txtfiles/
├── src/
│   ├── pycache/
│   ├── agents.py
│   ├── graph_utils.py
│   ├── rag_engine.py
│   ├── init.py
│   └── api.py
├── graphs/
├── scripts/
│   ├── ingest_texts.py
│   ├── load_pdf_and_index.py
│   ├── query.py
│   ├── ingest_text.py
│   ├── process_pdfs.py
│   ├── reindex.py
│   ├── batch_ingest.py
│   ├── download_pdfs.py
│   ├── extract_metrics.py
│   ├── create_embeddings.py
│   ├── chunk_text.py
│   └── extract_pdf_text.py
├── pycache/
├── services/
│   ├── pycache/
│   ├── llm_setup.py
│   ├── rag_engine.py
│   └── pdf_ingestor.py
├── routes/
│   ├── ingest.py
│   ├── pycache/
│   └── chat.py
├── tests/
│   ├── test_api.py
│   ├── test_rag_engine.py
│   ├── test_master_agent.py
│   ├── test_pdf_downloader.py
│   ├── test_chat.py
│   ├── test_chatbot.py
│   └── init.py
├── app/
│   ├── init.py
│   ├── main.py
│   ├── utils/
│   │   ├── pycache/
│   │   ├── text_processor.py
│   │   ├── init.py
│   │   └── logger.py
│   ├── services/
│   │   ├── master_agent.py
│   │   ├── pycache/
│   │   ├── rag_engine.py
│   │   ├── data_extractor.py
│   │   ├── llm_setup.py
│   │   ├── data_validator.py
│   │   ├── text_processor.py
│   │   ├── graph_generator.py
│   │   ├── pdf_ingestor.py
│   │   └── init.py
│   ├── routes/
│   │   ├── pycache/
│   │   ├── init.py
│   │   ├── chat.py
│   │   ├── ingest.py
│   │   └── chat_routes.py
│   ├── templates/
│   │   └── index.html
│   └── db/
│       ├── database.py
│       └── init.py
├── models/
├── .cache/
