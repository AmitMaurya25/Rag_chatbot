"""
RAG Chatbot — Streamlit UI

Features:
- Upload PDF / DOCX / TXT / MD docs, ingest into a persisted FAISS index
- Ask questions grounded in the uploaded documents (RAG), streamed live
- Falls back to plain chat (Groq model) when no documents have been uploaded
- Multiple chat threads, in-memory for the current session (per your choice)
- Follow-up questions are history-aware (condensed into standalone queries)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src import memory
from src.config import settings, validate_settings
from src.document_loader import UnsupportedFileType, ingest_file, save_uploaded_file
from src.llm import condense_question, get_llm, stream_answer
from src.logger import get_logger
from src.vectorstore import (
    build_or_extend_vectorstore,
    delete_vectorstore,
    get_retriever,
    load_vectorstore,
)
from langchain_core.output_parsers import StrOutputParser

log = get_logger("app")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
memory.init_state()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []  # names of files ingested this session


# ---------------------------------------------------------------------------
# Sidebar: config check, document upload, thread management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💬 RAG Chatbot")

    problems = validate_settings()
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    st.caption(f"Model: `{settings.chat_model}`")

    st.divider()
    st.subheader("📄 Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process & index documents", type="primary", use_container_width=True):
            with st.spinner("Reading, chunking, and embedding documents…"):
                all_chunks = []
                for uf in uploaded_files:
                    try:
                        max_bytes = settings.max_upload_mb * 1024 * 1024
                        file_bytes = uf.getvalue()
                        if len(file_bytes) > max_bytes:
                            st.warning(f"Skipped {uf.name}: exceeds {settings.max_upload_mb}MB limit.")
                            continue
                        path = save_uploaded_file(file_bytes, uf.name)
                        chunks = ingest_file(path)
                        all_chunks.extend(chunks)
                        st.session_state.ingested_files.append(uf.name)
                    except UnsupportedFileType as e:
                        st.warning(str(e))
                    except Exception:
                        log.exception("Failed to ingest %s", uf.name)
                        st.error(f"Failed to process {uf.name}. Check logs/app.log for details.")

                if all_chunks:
                    st.session_state.vectorstore = build_or_extend_vectorstore(
                        all_chunks, st.session_state.vectorstore
                    )
                    st.success(f"Indexed {len(all_chunks)} chunk(s) from {len(uploaded_files)} file(s).")
                    st.rerun()

    if st.session_state.ingested_files:
        st.caption("Indexed this session:")
        for fname in st.session_state.ingested_files:
            st.caption(f"• {fname}")

    if st.session_state.vectorstore is not None:
        if st.button("🗑️ Clear knowledge base", use_container_width=True):
            delete_vectorstore()
            st.session_state.vectorstore = None
            st.session_state.ingested_files = []
            st.rerun()

    st.divider()
    st.subheader("🧵 Chat threads")

    if st.button("➕ New chat", use_container_width=True):
        memory.create_thread()
        st.rerun()

    active_id = st.session_state["active_thread_id"]
    for thread in memory.list_threads():
        col1, col2 = st.columns([5, 1])
        with col1:
            label = ("**" + thread["title"] + "**") if thread["id"] == active_id else thread["title"]
            if st.button(label, key=f"switch_{thread['id']}", use_container_width=True):
                memory.switch_thread(thread["id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{thread['id']}"):
                memory.delete_thread(thread["id"])
                st.rerun()

    st.caption("Chat history is kept in memory for this session only and clears on restart.")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
active_thread = memory.get_active_thread()

for msg in active_thread["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask a question…")

if user_question:
    memory.add_message("user", user_question)
    with st.chat_message("user"):
        st.markdown(user_question)

    history = memory.as_lc_history(active_thread, exclude_last=True)

    with st.chat_message("assistant"):
        try:
            vs = st.session_state.vectorstore

            if vs is not None:
                # --- RAG path ---
                standalone_question = condense_question(user_question, history)
                retriever = get_retriever(vs)
                context_docs = retriever.invoke(standalone_question)

                with st.expander("🔍 Retrieved context", expanded=False):
                    if context_docs:
                        for d in context_docs:
                            st.caption(f"**{d.metadata.get('source_file', 'unknown')}**")
                            st.text(d.page_content[:500])
                    else:
                        st.caption("No relevant chunks found for this question.")

                answer_stream = stream_answer(user_question, history, context_docs)
            else:
                # --- Plain chat fallback (no documents indexed yet) ---
                st.info("No documents indexed yet — answering from general knowledge. Upload a file in the sidebar to enable document Q&A.", icon="ℹ️")
                from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

                plain_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful, concise assistant."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{question}"),
                    ]
                )
                chain = plain_prompt | get_llm(streaming=True) | StrOutputParser()
                answer_stream = chain.stream({"chat_history": history, "question": user_question})

            full_answer = st.write_stream(answer_stream)

        except Exception as e:
            log.exception("Error generating answer")
            full_answer = (
                "Sorry, something went wrong while generating a response. "
                f"({type(e).__name__}: {e})"
            )
            st.error(full_answer)

    memory.add_message("assistant", full_answer)
