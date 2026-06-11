# ─────────────────────────────────────────────────────────────────
# fetch_data.py — Real data fetcher for RAG pipeline testing
#
# SOURCES:
#   1. Bentley iTwin docs  — web scraper (no key needed)
#   2. Wikipedia API       — REST API (no key needed)
#   3. NewsAPI             — REST API (free dev key)
#
# HOW TO RUN:
#   python fetch_data.py                    # fetch all sources
#   python fetch_data.py --source bentley   # fetch only Bentley
#   python fetch_data.py --source wikipedia # fetch only Wikipedia
#   python fetch_data.py --source news      # fetch only NewsAPI
#
# WHAT IT DOES:
#   Fetches content → cleans text → saves to docs/ folder →
#   ingests directly into ChromaDB via core.ingest
#
# INTERVIEW ANSWER — "How did you test with real data?":
#   "I built a data fetcher that pulls from three different sources:
#    Bentley's own documentation (directly relevant to the role),
#    Wikipedia for factual reference content, and NewsAPI for
#    current unstructured news articles. This tests the pipeline
#    across all three major content types — technical docs,
#    encyclopedic text, and live news."
# ─────────────────────────────────────────────────────────────────

import os
import sys
import time
import argparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────
DOCS_DIR = "docs"
REQUEST_DELAY = 1.5   # seconds between requests — be polite to servers
REQUEST_TIMEOUT = 15  # seconds before giving up on a request

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Wikimedia API requires a bot-identified User-Agent — browser User-Agents
# get 403 from their CDN even though the main site loads fine.
# Format: AppName/Version (contact) library/version
# Reference: https://www.mediawiki.org/wiki/API:Etiquette
WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "RAGPipeline/1.0 "
        "(github.com/Ved1232; educational RAG project) "
        "python-requests/2.31.0"
    ),
    "Accept": "application/json",
}

os.makedirs(DOCS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# SOURCE 1 — BENTLEY iTWIN DOCUMENTATION
#
# WHY THIS SOURCE:
#   Directly relevant to the target role. Tests the pipeline on
#   real technical API documentation — the exact content type
#   you'd use in a production RAG system for a developer portal.
#
# APPROACH:
#   Simple requests + BeautifulSoup scraper. No API key needed.
#   Bentley's docs are publicly accessible HTML pages.
#   We extract only the main content area, stripping nav/footer.
# ─────────────────────────────────────────────────────────────────

BENTLEY_PAGES = [
    (
        "bentley_itwin_overview.txt",
        "https://developer.bentley.com/itwinplatform/",
        "Bentley iTwin Platform overview and core concepts"
    ),
    (
        "bentley_auth_scope.txt",
        "https://developer.bentley.com/itwin-platform-scope-introduction/",
        "Bentley iTwin authentication scopes and OAuth configuration"
    ),
    (
        "bentley_tutorials.txt",
        "https://developer.bentley.com/tutorials/",
        "Bentley iTwin Platform tutorials and getting started guides"
    ),
]


def fetch_bentley():
    """
    Fetches Bentley iTwin documentation pages and saves as .txt files.

    WHY BEAUTIFULSOUP FOR BENTLEY?
    Bentley's docs are server-rendered HTML — we can get the content
    directly from the HTML response without a browser. BeautifulSoup
    strips all the nav menus, headers, footers, and cookie banners
    leaving only the documentation text.

    Returns:
        list of (filename, word_count) tuples for successfully fetched pages
    """
    print("\n── Source 1: Bentley iTwin Docs ──────────────────────")
    results = []

    for filename, url, description in BENTLEY_PAGES:
        print(f"  Fetching: {description}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "header",
                             "footer", "aside", "form", "button"]):
                tag.decompose()

            # Find main content
            content = (
                soup.find("article") or
                soup.find("main") or
                soup.find("div", class_=lambda c: c and "content" in c.lower()) or
                soup.find("body")
            )

            if not content:
                print(f"  ✗ Could not find content area")
                continue

            text = content.get_text(separator=" ")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            clean_text = "\n".join(lines)

            if len(clean_text.strip()) < 100:
                print(f"  ⚠ Too little text ({len(clean_text)} chars) — page may be JS-rendered")
                continue

            # Save to docs/ folder
            filepath = os.path.join(DOCS_DIR, filename)
            header = f"SOURCE: {description}\nURL: {url}\n{'─'*60}\n\n"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + clean_text)

            word_count = len(clean_text.split())
            print(f"  ✓ Saved {filename} ({word_count} words)")
            results.append((filename, word_count))

        except Exception as e:
            print(f"  ✗ Failed: {e}")

        time.sleep(REQUEST_DELAY)

    return results


# ─────────────────────────────────────────────────────────────────
# SOURCE 2 — WIKIPEDIA REST API
#
# WHY WIKIPEDIA:
#   Free, no API key, structured factual content.
#   Wikipedia articles are well-written, dense with information,
#   and easy to verify — if the pipeline answers "What is RAG?"
#   correctly from the Wikipedia article, retrieval is working.
#
# API ENDPOINT:
#   https://en.wikipedia.org/api/rest_v1/page/summary/{title}
#   Returns: title, extract (plain text summary), description
#
# WHY THE SUMMARY ENDPOINT?
#   The full article endpoint returns MediaWiki markup which needs
#   heavy cleaning. The summary endpoint returns clean plain text
#   directly — perfect for our use case.
#   For longer articles, we also fetch sections via the mobile API.
# ─────────────────────────────────────────────────────────────────

WIKIPEDIA_TOPICS = [
    # Cloud and AI topics — relevant to tech roles
    ("Retrieval-augmented generation", "wikipedia_rag.txt"),
    ("Large language model", "wikipedia_llm.txt"),
    ("Vector database", "wikipedia_vector_db.txt"),
    ("Cloud computing", "wikipedia_cloud_computing.txt"),
    ("Digital twin", "wikipedia_digital_twin.txt"),
    ("LangChain", "wikipedia_langchain.txt"),
    ("Redis", "wikipedia_redis.txt"),
]


def fetch_wikipedia_article(title):
    """
    Fetches a Wikipedia article using two fallback strategies:

    Strategy 1 — Action API (action=parse):
      Full article HTML, cleaned with BeautifulSoup.
      Most reliable, used by thousands of apps since 2004.

    Strategy 2 — REST v1 extract API:
      Plain text extract — less content but always clean.
      Used as fallback if Strategy 1 fails.

    Both use the same Wikipedia domain — if one fails due to
    network issues, both will fail. The function gracefully
    returns None in that case and the caller skips the topic.

    Args:
        title: Wikipedia article title

    Returns:
        str: clean article text, or None if fetch failed
    """
    # ── Strategy 1: Action API (full article) ──────────────────
    try:
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "redirects": 1,
            "disableeditsection": 1,
        }

        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=WIKIPEDIA_HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        # Guard against non-JSON responses (403, empty body etc.)
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type or not response.text.strip():
            raise ValueError("Non-JSON response")

        data = response.json()

        if "error" in data:
            raise ValueError(f"API error: {data['error']}")

        html_content = data.get("parse", {}).get("text", {}).get("*", "")
        canonical_title = data.get("parse", {}).get("title", title)

        if not html_content:
            raise ValueError("Empty HTML content")

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["sup", "table", "div.navbox",
                         "div.reflist", "div.mw-references-wrap",
                         "div.hatnote", "div.toc"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        all_text = [f"# {canonical_title}\n"]

        for p in paragraphs:
            text = p.get_text(separator=" ").strip()
            if len(text) > 50:
                all_text.append(text)

        result = "\n\n".join(all_text)
        if len(result) > 200:
            return result

    except Exception:
        pass

    # ── Strategy 2: REST v1 extract (plain text fallback) ──────
    try:
        slug = requests.utils.quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"

        response = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "json" not in content_type or not response.text.strip():
            return None

        data = response.json()
        extract = data.get("extract", "")
        canonical_title = data.get("title", title)

        if len(extract) > 200:
            return f"# {canonical_title}\n\n{extract}"

    except Exception:
        pass

    return None


def fetch_wikipedia():
    """
    Fetches all configured Wikipedia topics and saves as .txt files.

    Returns:
        list of (filename, word_count) tuples
    """
    print("\n── Source 2: Wikipedia API ───────────────────────────")
    results = []

    for topic, filename in WIKIPEDIA_TOPICS:
        print(f"  Fetching: {topic}")

        text = fetch_wikipedia_article(topic)

        if not text or len(text.strip()) < 200:
            print(f"  ✗ Failed or too short for: {topic}")
            time.sleep(REQUEST_DELAY)
            continue

        filepath = os.path.join(DOCS_DIR, filename)
        header = f"SOURCE: Wikipedia — {topic}\n{'─'*60}\n\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + text)

        word_count = len(text.split())
        print(f"  ✓ Saved {filename} ({word_count} words)")
        results.append((filename, word_count))

        time.sleep(REQUEST_DELAY)

    return results


# ─────────────────────────────────────────────────────────────────
# SOURCE 3 — NEWSAPI
#
# WHY NEWSAPI:
#   Tests the pipeline on real, current, unstructured content.
#   News articles are short, varied in style, and time-sensitive —
#   completely different from technical docs or encyclopedia articles.
#   This proves the pipeline handles diverse content types.
#
# API ENDPOINT:
#   https://newsapi.org/v2/everything?q={query}&apiKey={key}
#
# FREE TIER LIMITS:
#   100 requests/day — more than enough for testing
#   Articles from last 30 days
#   Full article content (not just headlines)
#
# WHY NOT JUST USE HEADLINES?
#   "Headlines don't contain enough information for meaningful RAG.
#    We fetch the full article description and content so there's
#    actually something for the embedding model to work with."
# ─────────────────────────────────────────────────────────────────

NEWS_TOPICS = [
    ("artificial intelligence cloud computing", "news_ai_cloud.txt"),
    ("LangChain RAG pipeline", "news_langchain_rag.txt"),
    ("Bentley Systems digital twin", "news_bentley_digital_twin.txt"),
    ("vector database pinecone chromadb", "news_vector_db.txt"),
]

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_news_articles(query, max_articles=5):
    """
    Fetches news articles for a given search query via NewsAPI.

    WHY max_articles=5?
    Each article is ~200-500 words. 5 articles = ~1500 words per topic
    which is enough for meaningful retrieval without over-indexing.

    WHAT WE EXTRACT:
    - Article title
    - Source name
    - Published date
    - Description (lead paragraph)
    - Content (article body, truncated by NewsAPI free tier)

    Args:
        query: search query string
        max_articles: maximum number of articles to fetch

    Returns:
        str: combined article text, or None if fetch failed
    """
    if not NEWS_API_KEY:
        print("  ✗ NEWS_API_KEY not found in .env — skipping NewsAPI")
        return None

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",     # most relevant articles first
        "pageSize": max_articles,
    }

    try:
        response = requests.get(
            NEWS_API_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"  ✗ API error: {data.get('message', 'unknown error')}")
            return None

        articles = data.get("articles", [])
        if not articles:
            print(f"  ✗ No articles found for: {query}")
            return None

        # Build combined text from all articles
        all_text = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "").strip()
            source = article.get("source", {}).get("name", "Unknown")
            published = article.get("publishedAt", "")[:10]  # just the date
            description = article.get("description", "").strip()
            content = article.get("content", "").strip()

            # Clean up truncation marker that NewsAPI adds
            # "[+XXXX chars]" appears when content is truncated
            if content and "[+" in content:
                content = content[:content.rfind("[+")].strip()

            article_text = f"## Article {i}: {title}\n"
            article_text += f"Source: {source} | Published: {published}\n\n"
            if description:
                article_text += f"{description}\n\n"
            if content and content != description:
                article_text += f"{content}\n"

            all_text.append(article_text)

        return "\n\n---\n\n".join(all_text)

    except requests.exceptions.HTTPError as e:
        if "401" in str(e):
            print(f"  ✗ Invalid API key — check NEWS_API_KEY in .env")
        elif "429" in str(e):
            print(f"  ✗ Rate limit hit — wait and try again")
        else:
            print(f"  ✗ HTTP error: {e}")
        return None

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def fetch_news():
    """
    Fetches all configured news topics and saves as .txt files.

    Returns:
        list of (filename, word_count) tuples
    """
    print("\n── Source 3: NewsAPI ─────────────────────────────────")

    if not NEWS_API_KEY:
        print("  ✗ NEWS_API_KEY not set in .env")
        print("  Get a free key at: https://newsapi.org/register")
        return []

    results = []

    for query, filename in NEWS_TOPICS:
        print(f"  Fetching: '{query}'")

        text = fetch_news_articles(query, max_articles=5)

        if not text:
            time.sleep(REQUEST_DELAY)
            continue

        filepath = os.path.join(DOCS_DIR, filename)
        header = (
            f"SOURCE: NewsAPI — Topic: {query}\n"
            f"{'─'*60}\n\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + text)

        word_count = len(text.split())
        print(f"  ✓ Saved {filename} ({word_count} words)")
        results.append((filename, word_count))

        # NewsAPI free tier: be conservative with rate limiting
        time.sleep(REQUEST_DELAY)

    return results


# ─────────────────────────────────────────────────────────────────
# INGESTION — Feed all fetched docs into the RAG pipeline
# ─────────────────────────────────────────────────────────────────

def ingest_all_docs(doc_files):
    """
    Ingests all fetched documents into ChromaDB via core.ingest.

    This is the bridge between the data fetcher and the RAG pipeline.
    After fetching, we don't need to manually upload files through
    the UI — we call ingest_file() directly.

    Args:
        doc_files: list of filenames that were successfully fetched
    """
    if not doc_files:
        print("\n  No documents to ingest.")
        return

    print(f"\n── Ingesting {len(doc_files)} documents into ChromaDB ──")

    try:
        from core.ingest import ingest_file
    except ImportError:
        print("  ✗ Could not import core.ingest")
        print("  Make sure you're running from the project root directory")
        return

    total_chunks = 0
    for filename in doc_files:
        filepath = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(filepath):
            continue

        try:
            chunks_added, total = ingest_file(filepath)
            print(f"  ✓ {filename}: {chunks_added} chunks indexed")
            total_chunks += chunks_added
        except Exception as e:
            print(f"  ✗ Failed to ingest {filename}: {e}")

    print(f"\n  Total chunks indexed: {total_chunks}")


# ─────────────────────────────────────────────────────────────────
# TEST QUESTIONS — Run these after fetching to verify the pipeline
# ─────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    # Bentley-specific
    ("What is the iTwin Platform?", "bentley"),
    ("What is the itwin-platform scope used for?", "bentley"),

    # Wikipedia-specific
    ("What is Retrieval-Augmented Generation?", "wikipedia"),
    ("What is a vector database?", "wikipedia"),
    ("What is a digital twin?", "wikipedia"),

    # News-specific
    ("What are the latest developments in AI and cloud computing?", "news"),
    ("What is happening with Bentley Systems and digital twins?", "news"),

    # Cross-source (tests that retrieval picks the right source)
    ("How does LangChain relate to RAG pipelines?", "cross"),

    # Out-of-scope (hallucination guard test)
    ("What is the capital of Ireland?", "out_of_scope"),
]


def run_test_questions():
    """
    Runs test questions through the pipeline and prints results.
    This verifies that ingestion worked and retrieval is accurate.
    """
    print("\n── Running test questions ────────────────────────────")

    try:
        from core.query import query_pipeline
    except ImportError:
        print("  ✗ Could not import core.query")
        return

    for question, category in TEST_QUESTIONS:
        print(f"\n  [{category.upper()}] {question}")
        try:
            answer, sources = query_pipeline(question)
            # Truncate long answers for readability
            short_answer = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"  Answer: {short_answer}")
            print(f"  Sources: {', '.join(sources) if sources else 'none'}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


# ─────────────────────────────────────────────────────────────────
# MAIN — Entry point with argument parsing
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch real data and ingest into RAG pipeline"
    )
    parser.add_argument(
        "--source",
        choices=["bentley", "wikipedia", "news", "all"],
        default="all",
        help="Which source to fetch (default: all)"
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Fetch files but skip ingestion into ChromaDB"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test questions after ingestion"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("RAG Pipeline — Real Data Fetcher")
    print("=" * 55)
    print(f"Output folder : {DOCS_DIR}/")
    print(f"Source        : {args.source}")
    print(f"Auto-ingest   : {'no' if args.no_ingest else 'yes'}")

    all_fetched = []

    # Fetch selected sources
    if args.source in ("bentley", "all"):
        results = fetch_bentley()
        all_fetched.extend([f for f, _ in results])

    if args.source in ("wikipedia", "all"):
        results = fetch_wikipedia()
        all_fetched.extend([f for f, _ in results])

    if args.source in ("news", "all"):
        results = fetch_news()
        all_fetched.extend([f for f, _ in results])

    # Summary
    print(f"\n── Summary ───────────────────────────────────────────")
    print(f"  Files fetched: {len(all_fetched)}")
    for f in all_fetched:
        print(f"  ✓ docs/{f}")

    # Ingest into ChromaDB
    if not args.no_ingest and all_fetched:
        ingest_all_docs(all_fetched)

    # Run test questions
    if args.test:
        run_test_questions()

    print("\n" + "=" * 55)
    print("Done! Open the app and test your questions:")
    print("  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()