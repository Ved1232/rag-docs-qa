"""
setup_structure.py — Run this ONCE to create the full project folder structure.

Usage:
    python setup_structure.py

What it does:
    - Creates all folders
    - Creates all empty placeholder files with descriptive headers
    - Does NOT overwrite files that already exist
    - Safe to run multiple times
"""

import os

# ── FOLDER STRUCTURE ──────────────────────────────────────────────
FOLDERS = [
    "core",
    "tests",
    "docs",
    ".github/workflows",
    "chroma_db",
]

# ── FILES WITH STARTER CONTENT ────────────────────────────────────
# Format: { "filepath": "file content" }
FILES = {
    # ── CORE PACKAGE ──
    "core/__init__.py": '''# core/__init__.py
# Makes core/ a Python package so imports work:
#   from core.ingest import ingest_text
#   from core.query import query_pipeline
''',

    "core/ingest.py": '''# core/ingest.py
# LangChain-powered document ingestion (Phase 1)
# TODO: Already built — paste content from Claude here
''',

    "core/query.py": '''# core/query.py
# LangChain LCEL retrieval + generation pipeline (Phase 2+3)
# TODO: Already built — paste content from Claude here
''',

    "core/cache.py": '''# core/cache.py
# Redis semantic query cache with TTL
# TODO: Already built — paste content from Claude here
''',

    "core/auth.py": '''# core/auth.py
# streamlit-authenticator login system
# Per-user document isolation in Pinecone namespaces
# TODO: To be built in next phase
''',

    "core/tasks.py": '''# core/tasks.py
# Celery async task queue for background document ingestion
# Broker: Redis
# TODO: To be built in next phase
''',

    # ── TESTS ──
    "tests/__init__.py": '''# tests/__init__.py
# Makes tests/ a Python package for pytest discovery
''',

    "tests/test_ingest.py": '''# tests/test_ingest.py
# Tests for document ingestion pipeline
# Covers: chunking, embedding, deduplication, file loading
# TODO: To be built in next phase
''',

    "tests/test_query.py": '''# tests/test_query.py
# Tests for retrieval + generation pipeline
# Covers: retrieval accuracy, source citations, hallucination guard
# TODO: To be built in next phase
''',

    "tests/test_cache.py": '''# tests/test_cache.py
# Tests for Redis cache layer
# Covers: cache hit/miss, TTL expiry, fallback without Redis
# TODO: To be built in next phase
''',

    # ── DOCKER ──
    "Dockerfile": '''# Dockerfile
# Python 3.11 base image for the RAG pipeline
# TODO: To be built in Docker phase
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
''',

    "docker-compose.yml": '''# docker-compose.yml
# Spins up three services with one command: docker-compose up
# Services: app (Streamlit) + redis + celery worker
# TODO: To be completed in Docker phase
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8501:8501"
    env_file: .env
    depends_on:
      - redis
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./docs:/app/docs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    command: celery -A core.tasks worker --loglevel=info
    env_file: .env
    depends_on:
      - redis
''',

    # ── GITHUB ACTIONS ──
    ".github/workflows/ci.yml": '''# .github/workflows/ci.yml
# GitHub Actions CI/CD pipeline
# Runs on every push to main:
#   1. Install dependencies
#   2. Run pytest
#   3. Build Docker image
# TODO: To be completed in CI/CD phase
name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
''',

    # ── ROOT FILES ──
    "app.py": '''# app.py
# Streamlit UI entry point
# TODO: Already built — paste content from Claude here
# Run with: streamlit run app.py
''',

    "requirements.txt": '''# requirements.txt
# Pinned dependencies for the RAG pipeline
streamlit==1.35.0
langchain==0.2.6
langchain-openai==0.1.14
langchain-community==0.2.6
langchain-chroma==0.1.2
chromadb==0.4.24
pinecone-client==3.2.2
openai==1.55.3
httpx==0.27.2
redis==5.0.7
celery==5.4.0
streamlit-authenticator==0.3.2
bcrypt==4.1.3
pypdf==4.2.0
python-docx==1.1.2
ragas==0.1.9
python-dotenv==1.0.0
numpy==1.26.4
requests==2.31.0
beautifulsoup4==4.12.3
pytest==8.2.0
''',

    ".env": '''# .env — NEVER commit this file to GitHub
# Add your real keys here

OPENAI_API_KEY=your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=rag-docs-qa
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600
ANONYMIZED_TELEMETRY=False
''',

    ".gitignore": '''# .gitignore
# Files that should NEVER be pushed to GitHub

# API keys and secrets
.env

# Virtual environment
venv/
.venv/

# ChromaDB vector store (large binary files)
chroma_db/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Pytest cache
.pytest_cache/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Temporary files
*.tmp
*.log
''',

    "README.md": '''# RAG Document Q&A Pipeline

> Production-grade Retrieval-Augmented Generation system built with LangChain, ChromaDB/Pinecone, Redis, Celery, and Streamlit.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Vector DB (dev) | ChromaDB |
| Vector DB (prod) | Pinecone |
| Caching | Redis |
| Queue | Celery + Redis |
| LLM | GPT-4o-mini + Claude |
| Auth | streamlit-authenticator |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Evaluation | RAGAS |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/rag-docs-qa.git
cd rag-docs-qa

# 2. Create virtual environment
py -3.11 -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys to .env
cp .env.example .env
# Edit .env with your keys

# 5. Run the app
streamlit run app.py
```

## Run with Docker

```bash
docker-compose up
```

Opens at http://localhost:8501

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
rag-docs-qa/
├── core/                  # Pipeline logic (decoupled from UI)
│   ├── ingest.py          # Document ingestion
│   ├── query.py           # Retrieval + generation
│   ├── cache.py           # Redis cache
│   ├── auth.py            # User authentication
│   └── tasks.py           # Celery async queue
├── tests/                 # Pytest test suite
├── docs/                  # Source documents
├── .github/workflows/     # GitHub Actions CI/CD
├── app.py                 # Streamlit UI
├── Dockerfile
└── docker-compose.yml
```

## Author

Vedant Chavan — MSc Cloud Computing, NCI Dublin
[LinkedIn](https://linkedin.com/in/vedantrchavan) | [GitHub](https://github.com/Ved1232)
''',

    "scrape_bentley.py": '''# scrape_bentley.py
# Fetches Bentley iTwin public documentation and saves as .txt files
# Run once to populate the docs/ folder
# Usage: python scrape_bentley.py
# TODO: Already built — paste content from Claude here
''',
}


def create_structure():
    print("=" * 55)
    print("RAG Pipeline — Project Structure Setup")
    print("=" * 55)

    # Create folders
    print("\n📁 Creating folders...")
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ {folder}/")

    # Create files
    print("\n📄 Creating files...")
    created = 0
    skipped = 0

    for filepath, content in FILES.items():
        if os.path.exists(filepath):
            print(f"  ⏭  Skipped (exists): {filepath}")
            skipped += 1
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✓ Created: {filepath}")
            created += 1

    # Summary
    print("\n" + "=" * 55)
    print(f"Done! {created} files created, {skipped} skipped.")
    print("\nNext steps:")
    print("  1. Add your API keys to .env")
    print("  2. Paste your core/ files from Claude into each file")
    print("  3. Run: streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    create_structure()