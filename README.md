# 🛡️ Enterprise RAG Knowledge Engine

> A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, Pinecone, Redis, Celery, and Streamlit. Answers questions grounded exclusively in your documents — no hallucination, full source citations, multi-user with role-based access.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple)](https://pinecone.io)
[![Redis](https://img.shields.io/badge/Redis-7.x-red)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![CI](https://github.com/Ved1232/rag-docs-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Ved1232/rag-docs-qa/actions/workflows/ci.yml)

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Why This Solution Works](#why-this-solution-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Real Data Integration](#real-data-integration)
- [Authentication and Roles](#authentication-and-roles)
- [Known Issues and Fixes](#known-issues-and-fixes)
- [Testing](#testing)
- [CICD Pipeline](#cicd-pipeline)
- [Interview QA](#interview-qa)
- [Author](#author)

---

## Problem Statement

Large Language Models like GPT-4 are trained on general internet data up to a certain date. They cannot answer questions about private documents, access real-time information, provide source-cited answers, scale to enterprise document volumes, or serve multiple users with isolated data.

Traditional search returns documents. LLMs hallucinate answers. Neither solves the problem of getting accurate, grounded answers from a specific set of documents.

---

## Why This Solution Works

| Problem | Solution |
|---|---|
| LLM does not know your docs | Documents chunked, embedded, stored in vector DB |
| LLM hallucination | System prompt enforces answer only from provided context |
| No source attribution | Every answer cites the exact document it came from |
| Slow repeated queries | Redis cache serves repeat questions in under 5ms |
| Single user limitation | JWT auth plus Pinecone namespaces for isolated storage |
| Scaling to enterprise | ChromaDB for dev, Pinecone for prod, one env var switch |
| UI freezes on large uploads | Celery async queue processes PDFs in background |
| Duplicate document ingestion | SHA256 deterministic chunk IDs prevent duplicates |

---

## Architecture

```
User (Browser)
      |
Streamlit UI — Login, Register, Chat, Admin Panel
      |
core/auth    core/ingest         core/query
bcrypt+JWT   LangChain           LangChain
Roles        SHA256 IDs          LCEL Chain
             Embeddings          Retriever
                  |                   |
          Vector Store (factory pattern)
          DEV:  ChromaDB (local disk)
          PROD: Pinecone (cloud serverless)
                  |                   |
           Redis Cache           OpenAI API
           Celery Queue          GPT-4o-mini
```

Three phases of the RAG pipeline:

1. **Ingestion** — Documents split into 500-char chunks with 80-char overlap, embedded via text-embedding-3-small, stored with SHA256 deterministic IDs
2. **Retrieval** — User question embedded, cosine similarity search returns top-4 most relevant chunks
3. **Generation** — Chunks plus question plus chat history sent to GPT-4o-mini via LangChain LCEL chain

---

## Tech Stack

| Layer | Technology | Why Chosen |
|---|---|---|
| Orchestration | LangChain 0.3.x | Swap any component without rewriting logic |
| Vector DB dev | ChromaDB | Local, zero config, zero cost |
| Vector DB prod | Pinecone Serverless | Scales to billions of vectors, auto-scales to zero |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim, $0.00002 per 1K tokens |
| LLM | GPT-4o-mini | $0.00015 per 1K input tokens, fast, accurate |
| Caching | Redis 7 | In-memory, sub-millisecond reads, built-in TTL |
| Task Queue | Celery + Redis | Async ingestion, UI never freezes |
| Auth | streamlit-authenticator + bcrypt | bcrypt hashing, JWT sessions, role-based access |
| UI | Streamlit 1.45 | Python-only web interface |
| Containerisation | Docker + docker-compose | One command spins up full stack |
| CI/CD | GitHub Actions | Auto-runs pytest and builds Docker on every push |
| Evaluation | RAGAS | Measures faithfulness, answer relevance, context precision |

---

## Project Structure

```
rag-docs-qa/
├── core/
│   ├── __init__.py
│   ├── ingest.py          SHA256 chunk IDs, ChromaDB/Pinecone factory
│   ├── query.py           LCEL chain: retriever | prompt | LLM | parser
│   ├── cache.py           Redis query cache with TTL
│   ├── auth.py            bcrypt + JWT + roles + admin panel
│   └── tasks.py           Celery async task definitions
├── tests/
│   ├── test_ingest.py     In-memory ChromaDB, chunking, deduplication
│   ├── test_query.py      Retrieval accuracy, hallucination guard
│   └── test_cache.py      Cache hit/miss, Redis fallback
├── docs/                  Source documents populated by fetch_data.py
├── .github/workflows/
│   └── ci.yml             pytest then Docker build
├── app.py                 Streamlit UI entry point
├── fetch_data.py          Fetches Bentley + Wikipedia + NewsAPI
├── setup_pinecone.py      One-time Pinecone index creation and ingestion
├── Dockerfile             Multi-stage Python 3.11 image
├── docker-compose.yml     app + redis + worker
├── requirements.txt
└── .env                   API keys, never committed
```

---

## Quick Start

Prerequisites: Python 3.11, Docker Desktop, OpenAI key, Pinecone key, NewsAPI key

```bash
git clone https://github.com/Ved1232/rag-docs-qa.git
cd rag-docs-qa

py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create `.env`:

```
OPENAI_API_KEY=sk-proj-your-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=rag-docs-qa
VECTOR_STORE=chroma
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600
NEWS_API_KEY=your-newsapi-key-here
JWT_SECRET_KEY=change-this-in-production
ANONYMIZED_TELEMETRY=False
```

```bash
python fetch_data.py
python setup_pinecone.py
streamlit run app.py
```

Open http://localhost:8501

Default credentials: admin_user / Admin@123

---

## Running with Docker

```bash
docker-compose up --build
```

| Service | Container | Port |
|---|---|---|
| Streamlit | rag_app | 8501 |
| Redis | rag_redis | 6379 |
| Celery worker | rag_worker | internal |

Open http://localhost:8501

---

## Real Data Integration

| Source | Method | Content | Files |
|---|---|---|---|
| Bentley iTwin docs | BeautifulSoup scraper | iTwin API, Auth, Tutorials | bentley_*.txt |
| Wikipedia API | Action API action=parse | RAG, LLMs, Vector DBs, Digital twins | wikipedia_*.txt |
| NewsAPI | REST API free dev key | AI, cloud, Bentley news | news_*.txt |

Total indexed: 439 vectors across 14 documents

Wikipedia note: Requires bot User-Agent. Browser User-Agents return 403 from Wikimedia CDN. Format: AppName/Version (contact) library/version

---

## Authentication and Roles

Self-service registration with enforced password policy: 8+ characters, uppercase, lowercase, number, special character, cannot contain username.

| Feature | Admin | Viewer |
|---|---|---|
| Ask questions | yes | yes |
| Upload documents | yes | no |
| Admin panel | yes | no |
| Delete users | yes | no |

Security: bcrypt 12 rounds, JWT cookies, Pinecone namespace isolation at DB level, password policy enforced at registration.

---

## Known Issues and Fixes

| Issue | Cause | Fix |
|---|---|---|
| Python 3.13 + numpy failure | numpy does not support 3.13 on Windows | Install Python 3.11 side-by-side |
| .env UTF-16 encoding error | Windows echo saves as UTF-16 | Create in VS Code, save as UTF-8 |
| OpenAI proxies TypeError | openai and httpx version mismatch | Pin openai==1.68.2 httpx==0.27.2 |
| Wikipedia 403 error | Browser User-Agent blocked by Wikimedia CDN | Use bot-identified User-Agent format |
| streamlit-authenticator 0.4.x breaks | Four breaking API changes in one version | Systematic fixes per changelog |
| ChromaDB readonly in CI | PersistentClient needs disk write permissions | Switch tests to EphemeralClient |
| UUID chunk IDs break deduplication | uuid4 generates new ID every run, same doc treated as new | Replace with SHA256 of source plus content |
| Docker version attribute warning | Obsolete in Compose v2+ | Remove version line from docker-compose.yml |

---

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=core --cov-report=term-missing
```

Key design decisions:
- `chromadb.EphemeralClient()` — in-memory, no disk writes, works in any CI environment
- `mock_embeddings` injected into `patch_vectorstore` — Chroma requires embedding_function for add_documents
- SHA256 chunk IDs — deterministic deduplication verifiable in tests
- Fake LLM — predictable responses, no OpenAI API calls in tests

---

## CICD Pipeline

```
git push main
  Install Python 3.11 + dependencies
  Start Redis service container
  Run pytest
  Build Docker image
```

Add OPENAI_API_KEY to GitHub Secrets: Settings > Secrets > Actions > New repository secret

---

## Interview QA

**Q: Walk me through your architecture.**
Three phases: ingestion converts documents to vectors stored in Pinecone using SHA256 deterministic IDs to prevent duplicates; retrieval embeds the question and finds the top-4 chunks via cosine similarity; generation builds a grounded prompt and calls GPT-4o-mini via a LangChain LCEL chain. Redis sits in front — repeat questions are served in under 5ms.

**Q: Why SHA256 for chunk IDs instead of UUIDs?**
UUID generates a random ID every run so the same chunk gets a different ID each time and the duplicate check never matches. SHA256 of source name plus content always produces the same ID for the same content — re-ingesting a document correctly detects and skips existing chunks.

**Q: How do you prevent hallucination?**
The system prompt explicitly says answer ONLY from the provided context. I tested with "What is the capital of Ireland?" — the system returns "I don't have enough information" even though GPT knows the answer.

**Q: Why Redis for caching?**
Redis is in-memory — reads happen in microseconds. On a system with 40% repeat queries, caching cuts OpenAI costs by 40% and response time from 2-3 seconds to under 5ms.

**Q: How does per-user isolation work?**
Pinecone namespaces partition vectors within one index. User A's queries filter by namespace user_a — they literally cannot reach user B's vectors. Isolation is enforced at the database level not just the UI.

**Q: Why Docker?**
docker-compose orchestrates three containers. They communicate over a private bridge network using container names as hostnames. One command spins up the entire production stack identically on any machine.

---

## Author

**Vedant Chavan**
MSc Cloud Computing — National College of Ireland, Dublin (2026)
Microsoft AZ-104 Certified · Oracle AI Foundations Certified

[GitHub](https://github.com/Ved1232) · [Portfolio](https://vedantchavan01.vip) · [LinkedIn](https://linkedin.com/in/vedantrchavan)

---

