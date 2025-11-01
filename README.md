# Clinical RAG System

Retrieval-Augmented Generation system for clinical data analysis. Combines structured EHR data with semantic search over clinical notes for intelligent query answering.

## Architecture

- **MySQL 8.0**: Structured patient data, diagnoses, medications, labs
- **ChromaDB**: Vector embeddings (text-embedding-3-large, 3072 dims)
- **FastAPI**: REST API with tool-use agent
- **GPT-4o**: Reasoning with citations
- **Cross-encoder**: ms-marco-MiniLM-L-6-v2 for reranking

## Features

- Patient-specific and global queries
- Tool-use agent with structured data access
- Evidence citations with relevance scores
- Latency and token metrics

## Setup

```bash
docker compose up -d
docker compose exec backend python scripts/generate_data.py
docker compose exec backend python scripts/ingest.py
```
