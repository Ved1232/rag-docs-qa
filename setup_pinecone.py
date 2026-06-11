# setup_pinecone.py — Creates the Pinecone index and ingests all docs
# Run once: python setup_pinecone.py

import os
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-docs-qa")

print("=" * 55)
print("Pinecone Setup")
print("=" * 55)

# ── STEP 1: Verify API key ────────────────────────────────────────
if not PINECONE_API_KEY:
    print("✗ PINECONE_API_KEY not found in .env")
    exit(1)

print(f"✓ API key found: {PINECONE_API_KEY[:8]}...")
print(f"  Index name   : {INDEX_NAME}")

# ── STEP 2: Connect and create index ─────────────────────────────
try:
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✓ Connected to Pinecone")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Check existing indexes
existing = [idx.name for idx in pc.list_indexes()]
print(f"  Existing indexes: {existing if existing else 'none'}")

if INDEX_NAME in existing:
    print(f"✓ Index '{INDEX_NAME}' already exists — skipping creation")
else:
    print(f"\nCreating index '{INDEX_NAME}'...")
    print("  dimension : 1536 (text-embedding-3-small)")
    print("  metric    : cosine")
    print("  cloud     : aws us-east-1 (free tier)")

    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait for index to be ready
        print("\n  Waiting for index to be ready", end="", flush=True)
        for _ in range(30):
            status = pc.describe_index(INDEX_NAME).status
            if status.get("ready"):
                break
            print(".", end="", flush=True)
            time.sleep(2)
        print(" done!")
        print(f"✓ Index '{INDEX_NAME}' created successfully")

    except Exception as e:
        print(f"✗ Index creation failed: {e}")
        exit(1)

# ── STEP 3: Verify index stats ────────────────────────────────────
try:
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    print(f"\n── Index Stats ───────────────────────────────────────")
    print(f"  Total vectors : {total}")
    print(f"  Namespaces    : {stats.get('namespaces', {})}")
except Exception as e:
    print(f"✗ Could not get stats: {e}")

# ── STEP 4: Ingest docs into Pinecone ────────────────────────────
print(f"\n── Ingesting docs into Pinecone ──────────────────────")

# Set VECTOR_STORE=pinecone for the import
os.environ["VECTOR_STORE"] = "pinecone"

docs_dir = "docs"
if not os.path.exists(docs_dir):
    print("  ✗ docs/ folder not found — run fetch_data.py first")
else:
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    print(f"  Found {len(txt_files)} documents to ingest")

    try:
        from core.ingest import ingest_file

        total_chunks = 0
        for filename in txt_files:
            filepath = os.path.join(docs_dir, filename)
            try:
                chunks_added, total = ingest_file(filepath)
                print(f"  ✓ {filename}: {chunks_added} chunks")
                total_chunks += chunks_added
            except Exception as e:
                print(f"  ✗ {filename}: {e}")

        print(f"\n  Total chunks ingested: {total_chunks}")

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("  Make sure you're running from the project root")

# ── STEP 5: Final verification ────────────────────────────────────
print(f"\n── Final Verification ────────────────────────────────")
try:
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    total = stats.get("total_vector_count", 0)
    namespaces = stats.get("namespaces", {})
    print(f"  Total vectors in Pinecone : {total}")
    print(f"  Namespaces                : {namespaces}")

    if total > 0:
        print("\n✅ Pinecone is set up and ready!")
        print("\nNext steps:")
        print("  1. Make sure VECTOR_STORE=pinecone is in your .env")
        print("  2. Run: streamlit run app.py")
        print("  3. Ask questions — answers now come from Pinecone!")
    else:
        print("\n⚠ Index exists but no vectors yet")
        print("  Run fetch_data.py first to populate docs/")
        print("  Then run this script again")

except Exception as e:
    print(f"  ✗ Verification failed: {e}")

print("=" * 55)
