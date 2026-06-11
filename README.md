# 🛡️ Enterprise RAG Knowledge Engine

> A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, Pinecone, Redis, Celery, and Streamlit. Answers questions grounded exclusively in your documents — no hallucination, full source citations, multi-user with role-based access.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple)](https://pinecone.io)
[![Redis](https://img.shields.io/badge/Redis-7.x-red)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github)](https://github.com/features/actions)

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
- [Authentication & Roles](#authentication--roles)
- [API Integrations](#api-integrations)
- [Known Issues & Fixes](#known-issues--fixes)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Interview Q&A](#interview-qa)
- [Author](#author)

---

## 🎯 Problem Statement

Large Language Models like GPT-4 are trained on general internet data up to a certain date. They cannot:

- Answer questions about **private or internal documents**
- Access **real-time or recent information**
- Provide **verifiable, source-cited answers**
- Scale to **enterprise document volumes**
- Serve **multiple users with isolated data**

Traditional search returns documents. LLMs hallucinate answers. Neither solves the problem of getting accurate, grounded answers from a specific set of documents.

---

## ✅ Why This Solution Works

This RAG pipeline solves every limitation above:

| Problem | Our Solution |
|---|---|
| LLM doesn't know your docs | Documents are chunked, embedded, and stored in a vector database |
| LLM hallucination | System prompt enforces "answer only from provided context" |
| No source attribution | Every answer cites exactly which document it came from |
| Slow repeated queries | Redis cache serves repeat questions in <5ms, cutting API costs ~40% |
| Single user limitation | JWT auth + Pinecone namespaces give each user isolated document storage |
| Scaling to enterprise volume | ChromaDB for dev → Pinecone cloud for prod, swapped with one env var |
| UI freezes on large uploads | Celery async queue processes large PDFs in background |
| No verification of quality | RAGAS evaluation framework measures answer faithfulness and precision |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User (Browser)                        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│              Streamlit UI (app.py)                       │
│         Login · Register · Chat · Admin Panel            │
└──────┬───────────────┬────────────────┬─────────────────┘
       │               │                │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  core/auth  │ │core/ingest  │ │ core/query  │
│  bcrypt+JWT │ │  LangChain  │ │  LangChain  │
│  Roles:     │ │  Splitter   │ │  LCEL Chain │
│  admin/view │ │  Embeddings │ │  Retriever  │
└─────────────┘ └──────┬──────┘ └──────┬──────┘
                       │               │
              ┌────────▼───────────────▼────────┐
              │         Vector Store             │
              │  DEV:  ChromaDB (local disk)     │
              │  PROD: Pinecone (cloud serverless)│
              └─────────────────────────────────┘
                       │               │
              ┌────────▼──────┐ ┌──────▼──────┐
              │     Redis     │ │   OpenAI    │
              │  Query Cache  │ │  Embeddings │
              │  Celery Queue │ │  GPT-4o-mini│
              └───────────────┘ └─────────────┘
```

**Three phases of the RAG pipeline:**

1. **Ingestion** — Documents are loaded → split into 500-char chunks with 80-char overlap → embedded via `text-embedding-3-small` → stored in vector DB with source metadata
2. **Retrieval** — User question is embedded → cosine similarity search returns top-4 most relevant chunks → chunks formatted with source labels
3. **Generation** — Retrieved chunks + question + chat history → GPT-4o-mini via LangChain LCEL chain → grounded answer with citations

---

## 🛠️ Tech Stack

| Layer | Technology | Why Chosen |
|---|---|---|
| **Orchestration** | LangChain 0.3.x | Standard interface for all components — swap any part without rewriting logic |
| **Vector DB (dev)** | ChromaDB | Local, zero config, no cost — perfect for development and testing |
| **Vector DB (prod)** | Pinecone Serverless | Scales to billions of vectors, auto-scales to zero at idle, sub-10ms queries |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim, $0.00002/1K tokens, outperforms ada-002 |
| **LLM** | GPT-4o-mini | $0.00015/1K input tokens, fast, accurate for document Q&A |
| **Caching** | Redis 7 | In-memory, <1ms reads, built-in TTL — serves repeat queries instantly |
| **Task Queue** | Celery + Redis | Async document ingestion — large PDFs process in background, UI stays responsive |
| **Auth** | streamlit-authenticator + bcrypt | bcrypt password hashing, JWT cookie sessions, role-based access |
| **UI** | Streamlit 1.45 | Python-only web UI — no HTML/CSS/JS needed |
| **Containerisation** | Docker + docker-compose | One command spins up app + Redis + Celery worker |
| **CI/CD** | GitHub Actions | Auto-runs pytest + builds Docker image on every push to main |
| **Evaluation** | RAGAS | Measures faithfulness, answer relevance, context precision |
| **Data Sources** | Bentley iTwin + Wikipedia + NewsAPI | Three content types: technical docs, encyclopedic, live news |

---

## 📁 Project Structure

```
rag-docs-qa/
├── core/                          # Pipeline logic — decoupled from UI
│   ├── __init__.py                # Makes core/ a Python package
│   ├── ingest.py                  # Document ingestion (LangChain + ChromaDB/Pinecone)
│   ├── query.py                   # Retrieval + generation (LCEL chain)
│   ├── cache.py                   # Redis semantic query cache
│   ├── auth.py                    # Authentication, registration, roles
│   └── tasks.py                   # Celery async task definitions
│
├── tests/                         # Pytest test suite
│   ├── __init__.py
│   ├── test_ingest.py             # Tests: chunking, deduplication, file loading
│   ├── test_query.py              # Tests: retrieval accuracy, hallucination guard
│   └── test_cache.py              # Tests: cache hit/miss, Redis fallback
│
├── docs/                          # Source documents (populated by fetch_data.py)
│   ├── bentley_*.txt              # Bentley iTwin documentation
│   ├── wikipedia_*.txt            # Wikipedia articles
│   └── news_*.txt                 # NewsAPI articles
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions: test → build Docker
│
├── app.py                         # Streamlit UI entry point
├── fetch_data.py                  # Fetches real data from 3 APIs
├── setup_pinecone.py              # One-time Pinecone index creation + ingestion
├── setup_structure.py             # Creates full project folder structure
├── scrape_bentley.py              # Bentley docs scraper
├── Dockerfile                     # Multi-stage Python 3.11 image
├── docker-compose.yml             # app + redis + worker services
├── .dockerignore                  # Excludes venv, .env, chroma_db from image
├── requirements.txt               # All pinned dependencies
├── .env                           # API keys (never committed)
└── .gitignore                     # Excludes .env, venv, chroma_db
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Docker Desktop
- OpenAI API key — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Pinecone API key — [app.pinecone.io](https://app.pinecone.io)
- NewsAPI key — [newsapi.org/register](https://newsapi.org/register)

### 1. Clone and set up environment

```bash
git clone https://github.com/Ved1232/rag-docs-qa.git
cd rag-docs-qa

# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-proj-your-key-here

# Pinecone
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=rag-docs-qa

# Vector store: 'chroma' for local dev, 'pinecone' for production
VECTOR_STORE=chroma

# Redis (local)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600

# NewsAPI
NEWS_API_KEY=your-newsapi-key-here

# Auth
JWT_SECRET_KEY=change-this-in-production

# ChromaDB telemetry (disable)
ANONYMIZED_TELEMETRY=False
```

### 3. Fetch real data

```bash
# Fetch from all 3 sources: Bentley + Wikipedia + NewsAPI
python fetch_data.py

# Or fetch individual sources
python fetch_data.py --source wikipedia
python fetch_data.py --source news
python fetch_data.py --source bentley
```

### 4. Set up Pinecone (production mode)

```bash
# Creates the index and ingests all docs into Pinecone
python setup_pinecone.py
```

### 5. Run the app

```bash
# Local development (ChromaDB)
streamlit run app.py

# Production mode (Pinecone)
# Set VECTOR_STORE=pinecone in .env, then:
streamlit run app.py
```

Open **http://localhost:8501**

**Default credentials:**
```
Username: admin_user
Password: Admin@123
```

---

## 🐳 Running with Docker

The entire stack — app, Redis, and Celery worker — runs with one command:

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

**Services:**

| Service | Container | Port | Purpose |
|---|---|---|---|
| Streamlit app | `rag_app` | 8501 | Web UI + RAG pipeline |
| Redis | `rag_redis` | 6379 | Cache + Celery broker |
| Celery worker | `rag_worker` | — | Async document processing |

Open **http://localhost:8501** after startup.

> **Note:** `docker-compose up` uses `VECTOR_STORE=pinecone` by default. Make sure your `.env` has valid Pinecone credentials.

---

## 📡 Real Data Integration

The pipeline is tested with three real-world data sources:

### Source 1 — Bentley iTwin Documentation
- **Method:** Web scraper (BeautifulSoup + requests)
- **Content:** iTwin Platform overview, Auth scopes, Tutorials
- **Why:** Directly relevant to Bentley Systems data engineering roles
- **Files:** `bentley_itwin_overview.txt`, `bentley_auth_scope.txt`, `bentley_tutorials.txt`

### Source 2 — Wikipedia REST API
- **Method:** Wikipedia Action API (`action=parse`)
- **Content:** RAG, LLMs, Vector databases, Cloud computing, Digital twins
- **Why:** Free, no key required, structured factual content — easy to verify answers
- **Note:** Requires bot-identified User-Agent: `AppName/Version (contact) library/version`
- **Files:** `wikipedia_rag.txt`, `wikipedia_llm.txt`, `wikipedia_vector_db.txt`, etc.

### Source 3 — NewsAPI
- **Method:** REST API (free developer key, 100 req/day)
- **Content:** Latest AI, cloud, Bentley, and vector DB news articles
- **Why:** Tests pipeline on unstructured, time-sensitive, varied-length content
- **Files:** `news_ai_cloud.txt`, `news_langchain_rag.txt`, etc.

**Total indexed:** 439 vectors across 14 documents

---

## 🔐 Authentication & Roles

### Registration
Anyone can create an account via the Register tab:
- Username (3-20 chars, alphanumeric + underscore)
- Full name + email
- Role selection: **admin** or **viewer**
- Password with enforced policy (8+ chars, uppercase, lowercase, number, special char)

### Roles

| Feature | Admin | Viewer |
|---|---|---|
| Ask questions | ✅ | ✅ |
| Upload documents | ✅ | ❌ |
| Paste text | ✅ | ❌ |
| Admin panel | ✅ | ❌ |
| Delete users | ✅ | ❌ |

### Security
- Passwords hashed with **bcrypt** (12 rounds, random salt per user)
- Sessions managed via **JWT cookies** (1-day expiry)
- Per-user **Pinecone namespaces** — user A cannot access user B's documents even if they bypass the UI
- Password policy enforced at registration — no weak passwords accepted

---

## ⚠️ Known Issues & Fixes Encountered

These are real issues hit during development and how they were resolved:

### 1. Python 3.13 + numpy build failure
**Error:** `metadata-generation-failed` for numpy  
**Cause:** numpy doesn't support Python 3.13 on Windows yet  
**Fix:** Install Python 3.11 side-by-side, recreate venv with `py -3.11 -m venv venv`

### 2. `.env` file encoding (UTF-16)
**Error:** `dotenv parser decode error`  
**Cause:** Windows `echo` command saves files as UTF-16, dotenv expects UTF-8  
**Fix:** Create `.env` in VS Code and explicitly save as UTF-8, or use Python:
```python
with open('.env', 'w', encoding='utf-8') as f:
    f.write('OPENAI_API_KEY=your-key')
```

### 3. OpenAI `proxies` TypeError
**Error:** `Client.__init__() got unexpected keyword argument 'proxies'`  
**Cause:** openai and httpx version mismatch  
**Fix:** `pip install openai==1.68.2 httpx==0.27.2`

### 4. ChromaDB telemetry warnings
**Error:** `capture() takes 1 positional argument but 3 were given`  
**Cause:** ChromaDB telemetry bug in version 1.x  
**Fix:** Add `ANONYMIZED_TELEMETRY=False` to `.env` — harmless, just noise

### 5. Uploaded file shows temp name as source
**Error:** Source badge shows `tmpsijvun2z.txt` instead of original filename  
**Cause:** `ingest_file()` used the temp file path as the source label  
**Fix:** Read TXT files directly from the uploaded file object; pass original filename to `ingest_text()` separately

### 6. Wikipedia API returns 403
**Error:** All Wikipedia requests return `Wikimedia Error` 403  
**Cause:** Browser-style User-Agents are blocked by Wikimedia's CDN  
**Fix:** Use bot-identified User-Agent format required by Wikimedia API policy:
```python
"User-Agent": "RAGPipeline/1.0 (github.com/Ved1232; educational project) python-requests/2.31.0"
```

### 7. streamlit-authenticator 0.4.x breaking changes
**Errors (multiple):**
- `Hasher.__init__() takes 1 positional argument`
- `pre_authorized parameter removed`
- `login() got multiple values for location`
- `cannot unpack non-iterable NoneType`

**Cause:** All breaking API changes in streamlit-authenticator 0.4.x vs 0.3.x  
**Fixes:**
- Replace `stauth.Hasher(['pass']).generate()` → use `bcrypt` directly
- Remove `pre_authorized` from `Authenticate()` constructor
- Remove positional title arg from `login()` and `logout()`
- Read auth state from `st.session_state` instead of return values

### 8. Docker `version` attribute warning
**Error:** `the attribute version is obsolete`  
**Fix:** Remove `version: "3.9"` from `docker-compose.yml` — not needed in Compose v2+

### 9. PowerShell `rmdir /s /q` not working
**Error:** `PositionalParameterNotFound`  
**Cause:** `/s /q` flags are CMD syntax, not PowerShell  
**Fix:** Use `Remove-Item -Recurse -Force chroma_db`

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_cache.py -v

# Run with coverage report
pytest tests/ -v --cov=core --cov-report=term-missing
```

**Test coverage:**

| File | What it tests |
|---|---|
| `test_ingest.py` | Chunking logic, metadata inheritance, deduplication, file loading |
| `test_query.py` | Retrieval accuracy, source citations, hallucination guard, chat history |
| `test_cache.py` | Cache hit/miss, key generation, Redis fallback when unavailable |

**Key test — hallucination guard:**
```python
def test_empty_db_returns_helpful_message(self):
    answer, sources = query_pipeline("What is cloud computing?")
    assert "no documents" in answer.lower()  # must NOT hallucinate
```

---

## 🔄 CI/CD Pipeline

GitHub Actions runs on every push to `main`:

```
Push to main
    │
    ▼
Install Python 3.11 + dependencies
    │
    ▼
Start Redis service container
    │
    ▼
Run pytest (with Redis available)
    │
    ▼ (only if tests pass)
Build Docker image
    │
    ▼
✅ Pipeline complete
```

View results: GitHub repo → **Actions** tab

---

## 💬 Interview Q&A

**Q: Walk me through your architecture.**
> "The pipeline has three phases: ingestion converts documents to vectors stored in Pinecone; retrieval embeds the user's question and finds the 4 most similar chunks via cosine similarity; generation builds a grounded prompt and calls GPT-4o-mini via a LangChain LCEL chain. Redis sits in front of the LLM call — repeat questions are served in under 5ms without touching OpenAI."

**Q: Why LangChain instead of building it yourself?**
> "LangChain gives standard interfaces for every component. The `get_vectorstore()` factory function returns ChromaDB or Pinecone based on an env var — the rest of the code doesn't know which one it's talking to. Swapping databases is one line. Building that abstraction manually would take weeks."

**Q: How do you prevent hallucination?**
> "The system prompt explicitly says 'answer ONLY from the provided context — if the answer isn't there, say so.' The model is reading the documents like a human would, not relying on training memory. I tested this with 'What is the capital of Ireland?' — the system correctly says it doesn't have that information even though GPT knows the answer."

**Q: Why Redis for caching?**
> "Redis is in-memory — reads happen in microseconds, not milliseconds. A database would be 10-100x slower. Redis also has built-in TTL support so stale cache entries expire automatically. On a system with 40% repeat queries, caching cuts OpenAI API costs by 40% and dramatically improves response time."

**Q: How does per-user isolation work?**
> "Pinecone namespaces are logical partitions within one index. When user A ingests a document, it's stored under namespace 'user_a'. Queries from user A filter by namespace='user_a' — they literally cannot reach user B's vectors. The isolation is enforced at the database level, not just the UI."

**Q: Why Docker?**
> "Docker solves the 'works on my machine' problem. docker-compose orchestrates three containers — app, Redis, and Celery worker. They communicate over a private bridge network using container names as hostnames. One command spins up the entire production stack identically on any machine."

---

## 👤 Author

**Vedant Chavan**  
MSc Cloud Computing — National College of Ireland, Dublin (2026)  
Microsoft AZ-104 Certified · Oracle AI Foundations Certified

🔗 [GitHub](https://github.com/Ved1232) · [Portfolio](https://vedantchavan01.vip) · [LinkedIn](https://linkedin.com/in/vedantrchavan)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
