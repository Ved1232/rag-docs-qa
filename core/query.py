# ─────────────────────────────────────────────────────────────────
# core/query.py — LangChain retrieval + generation pipeline
#
# WHAT CHANGED FOR PINECONE:
#   get_retriever() now passes namespace to get_vectorstore()
#   so each user's query only searches their own documents.
#   Everything else stays identical — LCEL chain, prompt,
#   output parser, chat history — none of that changes.
#
# This proves the abstraction worked: adding Pinecone support
# required zero changes to the query logic itself.
# ─────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from core.ingest import get_vectorstore, VECTOR_STORE

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────
# temperature=0.2: factual, grounded answers over creative ones
# gpt-4o-mini: best cost/quality ratio for document Q&A
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ── PROMPT TEMPLATE ───────────────────────────────────────────────
# System prompt enforces grounding — model must answer from context only.
# {context} = retrieved chunks
# {question} = user question
# {chat_history} = previous turns for follow-up questions
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a precise, helpful assistant that answers questions \
based strictly on the provided document context.

Rules:
1. Answer ONLY using information from the context below.
2. If the answer is not in the context, say exactly: \
"I don't have enough information in the provided documents to answer that."
3. Never use outside knowledge or make assumptions beyond the context.
4. Keep answers concise and well structured.
5. When possible, reference which document supports your answer.

Context:
{context}

Previous conversation:
{chat_history}"""
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

output_parser = StrOutputParser()


def format_docs(docs):
    """
    Formats retrieved Document objects into a context string.
    Each chunk is separated by --- and labelled with its source.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"{doc.page_content}\n[Source: {source}]")
    return "\n\n---\n\n".join(formatted)


def get_retriever(k=4, namespace=None):
    """
    Returns a configured retriever from the active vector store.

    namespace is passed through to Pinecone for per-user isolation.
    In ChromaDB mode namespace is ignored.

    WHY k=4?
    "4 chunks gives enough context for most questions without
     exceeding the LLM context window or driving up token costs.
     For complex multi-part questions, increase to 6-8."

    Args:
        k: number of chunks to retrieve
        namespace: Pinecone namespace (user's namespace)
    """
    vectorstore = get_vectorstore(namespace=namespace)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def query_pipeline(question, chat_history="", namespace=None):
    """
    Full RAG query: embed question → retrieve chunks → generate answer.

    LCEL CHAIN (interview answer):
    "The | pipe operator chains: retriever formats context,
     the prompt assembles the full message, the LLM generates
     a response, and StrOutputParser extracts the text string.
     Each component is independently testable and swappable."

    Args:
        question: user's question string
        chat_history: formatted string of previous conversation turns
        namespace: Pinecone namespace for per-user document isolation

    Returns:
        tuple: (answer string, list of source document names)
    """
    # Safety check — nothing to query if store is empty
    vectorstore = get_vectorstore(namespace=namespace)

    # Count differs between ChromaDB and Pinecone
    try:
        if VECTOR_STORE == "pinecone":
            # Pinecone: check by doing a dummy query
            # index.describe_index_stats() would require direct client
            index_stats = vectorstore._index.describe_index_stats()
            ns = namespace or "default"
            total = index_stats.get("namespaces", {}).get(
                ns, {}
            ).get("vector_count", 0)
        else:
            total = vectorstore._collection.count()
    except Exception:
        total = 1  # assume not empty if we can't check

    if total == 0:
        return (
            "No documents have been ingested yet. "
            "Please upload or paste some text first.",
            []
        )

    # Retrieve relevant chunks
    retriever = get_retriever(k=4, namespace=namespace)
    retrieved_docs = retriever.invoke(question)

    # Extract unique source names for citation badges
    sources = list(set([
        doc.metadata.get("source", "unknown")
        for doc in retrieved_docs
    ]))

    # Format chunks into context string
    context = format_docs(retrieved_docs)

    # Build and invoke the LCEL chain
    # prompt | llm | output_parser is the full pipeline
    chain = prompt | llm | output_parser

    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    return answer, sources