# ─────────────────────────────────────────────────────────────────
# app.py — Enterprise RAG UI with full auth, roles, admin panel
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import os
import tempfile

from core.ingest import ingest_text, ingest_file
from core.query import query_pipeline
from core.cache import get_cached_answer, set_cached_answer, get_cache_stats, clear_cache
from core.auth import (
    render_login_page, render_logout, render_admin_panel,
    load_config, get_user_namespace, get_user_role
)

st.set_page_config(
    page_title="Enterprise Knowledge Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
        .source-badge {
            background-color: #f0f2f6; color: #31333F;
            padding: 0.2rem 0.5rem; border-radius: 4px;
            font-size: 0.85rem; font-family: monospace;
        }
        .cached-badge {
            background-color: #d4edda; color: #155724;
            padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# ── AUTHENTICATION GATE ───────────────────────────────────────────
name, authentication_status, username, authenticator = render_login_page()

if not authentication_status:
    st.stop()

# ── LOAD USER CONTEXT ─────────────────────────────────────────────
config = load_config()
user_namespace = get_user_namespace(username, config)
user_role = get_user_role(username, config)
is_admin = user_role == "admin"

# ── SESSION STATE ─────────────────────────────────────────────────
messages_key = f"messages_{username}"
sources_key = f"ingested_sources_{username}"

if messages_key not in st.session_state:
    st.session_state[messages_key] = []
if sources_key not in st.session_state:
    st.session_state[sources_key] = []

messages = st.session_state[messages_key]
ingested_sources = st.session_state[sources_key]

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    render_logout(authenticator, name, user_role)
    st.divider()
    st.markdown("## ⚙️ Control Center")
    st.caption("LangChain · Pinecone · Redis")
    st.divider()

    # ── DOCUMENT INGESTION (admin only) ──────────────────────────
    if is_admin:
        with st.expander("📥 Ingest Documents", expanded=True):
            tab_upload, tab_paste = st.tabs(["📁 Upload File", "📝 Paste Text"])

            with tab_upload:
                st.markdown("<small>Supported: PDF, TXT</small>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    label="Choose a file", type=["pdf", "txt"],
                    label_visibility="collapsed"
                )
                if uploaded_file is not None:
                    if st.button("⚡ Process Document", key="btn_upload", use_container_width=True):
                        with st.spinner("Processing..."):
                            ext = uploaded_file.name.lower().split(".")[-1]
                            try:
                                if ext == "txt":
                                    text = uploaded_file.read().decode("utf-8")
                                    ingested, total = ingest_text(
                                        text, source_name=uploaded_file.name,
                                        namespace=user_namespace
                                    )
                                elif ext == "pdf":
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                        tmp.write(uploaded_file.read())
                                        tmp_path = tmp.name
                                    ingested, total = ingest_file(tmp_path, namespace=user_namespace)
                                    os.unlink(tmp_path)
                                else:
                                    st.error("Unsupported format.")
                                    ingested, total = 0, 0

                                if ingested > 0:
                                    if uploaded_file.name not in ingested_sources:
                                        ingested_sources.append(uploaded_file.name)
                                    clear_cache()
                                    st.success(f"✅ {ingested} chunks from **{uploaded_file.name}**")
                            except Exception as e:
                                st.error(f"Failed: {str(e)}")

            with tab_paste:
                pasted_text = st.text_area(
                    label="Paste text", height=150,
                    placeholder="Paste documentation...",
                    label_visibility="collapsed"
                )
                source_name = st.text_input("Source name", placeholder="e.g. policy_doc")
                if st.button("⚡ Index Text", key="btn_paste", use_container_width=True):
                    if not pasted_text.strip():
                        st.warning("Please paste some text first.")
                    else:
                        with st.spinner("Indexing..."):
                            name_val = source_name.strip() if source_name.strip() else "pasted_text"
                            ingested, total = ingest_text(
                                pasted_text, source_name=name_val,
                                namespace=user_namespace
                            )
                        if name_val not in ingested_sources:
                            ingested_sources.append(name_val)
                        clear_cache()
                        st.success(f"✅ {ingested} chunks under **{name_val}**")
    else:
        # Viewer role — show info instead of ingestion controls
        st.info("📖 **Viewer access** — You can query documents but cannot upload new ones. Contact an admin to upload documents.")

    # ── KNOWLEDGE BASE ────────────────────────────────────────────
    with st.expander("📚 Knowledge Base", expanded=True):
        if ingested_sources:
            for source in ingested_sources:
                st.markdown(f"🔒 <code style='font-size:0.82rem;'>{source}</code>", unsafe_allow_html=True)
        else:
            st.info("No documents indexed this session.")

    # ── CACHE STATS ───────────────────────────────────────────────
    with st.expander("⚡ Cache Stats", expanded=False):
        stats = get_cache_stats()
        if stats:
            st.metric("Cached Queries", stats["cached_queries"])
            st.metric("Cache Hits", stats["total_hits"])
            st.metric("Cache Misses", stats["total_misses"])
            if st.button("🗑️ Clear Cache", key="clear_cache_btn"):
                clear_cache()
                st.success("Cache cleared")
        else:
            st.caption("⚠️ Redis unavailable")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state[messages_key] = []
        st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────

# Admin panel tab (only visible to admins)
if is_admin:
    tab_chat, tab_admin = st.tabs(["💬 Chat", "👑 Admin Panel"])
else:
    tab_chat = st.tabs(["💬 Chat"])[0]

with tab_chat:
    st.title("🛡️ Enterprise Knowledge Engine")
    st.markdown(
        f"**LangChain · Pinecone · Redis** — "
        f"Welcome **{name}** · Role: **{user_role}**"
    )
    st.divider()

    # Render chat history
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if message.get("from_cache"):
                    st.markdown("<span class='cached-badge'>⚡ Cached</span>", unsafe_allow_html=True)
                if message["sources"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    cols = st.columns(len(message["sources"]) + 1)
                    with cols[0]:
                        st.caption("⚓ **Sources:**")
                    for i, src in enumerate(message["sources"]):
                        with cols[i + 1]:
                            st.markdown(f"<span class='source-badge'>{src}</span>", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        chat_history = ""
        for msg in messages[:-1]:
            role = "Human" if msg["role"] == "user" else "Assistant"
            chat_history += f"{role}: {msg['content']}\n"

        with st.chat_message("assistant"):
            cached = get_cached_answer(prompt)
            if cached:
                answer = cached["answer"]
                sources = cached["sources"]
                from_cache = True
                st.markdown(answer)
                st.markdown("<span class='cached-badge'>⚡ Cached</span>", unsafe_allow_html=True)
            else:
                with st.spinner("Querying knowledge base..."):
                    answer, sources = query_pipeline(
                        prompt, chat_history=chat_history,
                        namespace=user_namespace
                    )
                st.markdown(answer)
                from_cache = False
                set_cached_answer(prompt, answer, sources)

            if sources:
                st.markdown("<br>", unsafe_allow_html=True)
                cols = st.columns(len(sources) + 1)
                with cols[0]:
                    st.caption("⚓ **Sources:**")
                for i, src in enumerate(sources):
                    with cols[i + 1]:
                        st.markdown(f"<span class='source-badge'>{src}</span>", unsafe_allow_html=True)

        messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "from_cache": from_cache
        })

# Admin panel tab
if is_admin:
    with tab_admin:
        render_admin_panel(username)