"""
Conversation memory.

Per your choice, this is in-memory only (resets when the Streamlit
process restarts) — no database. It lives entirely in
st.session_state, but is structured as multiple named "threads" (like
ChatGPT's chat list) so you can start a new conversation without losing
older ones during the same running session.

Each message is stored as {"role": "user"|"assistant", "content": str}.
For feeding into the LLM chain, use as_lc_history() which returns the
(role, content) tuples LangChain's MessagesPlaceholder expects.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import streamlit as st

_THREADS_KEY = "chat_threads"          # dict[thread_id -> thread dict]
_ACTIVE_THREAD_KEY = "active_thread_id"


def _new_thread(title: str = "New chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],  # list of {"role": ..., "content": ...}
    }


def init_state() -> None:
    if _THREADS_KEY not in st.session_state:
        st.session_state[_THREADS_KEY] = {}
    if _ACTIVE_THREAD_KEY not in st.session_state or (
        st.session_state[_ACTIVE_THREAD_KEY] not in st.session_state[_THREADS_KEY]
    ):
        thread = _new_thread()
        st.session_state[_THREADS_KEY][thread["id"]] = thread
        st.session_state[_ACTIVE_THREAD_KEY] = thread["id"]


def create_thread() -> str:
    thread = _new_thread()
    st.session_state[_THREADS_KEY][thread["id"]] = thread
    st.session_state[_ACTIVE_THREAD_KEY] = thread["id"]
    return thread["id"]


def switch_thread(thread_id: str) -> None:
    if thread_id in st.session_state[_THREADS_KEY]:
        st.session_state[_ACTIVE_THREAD_KEY] = thread_id


def delete_thread(thread_id: str) -> None:
    threads = st.session_state[_THREADS_KEY]
    if thread_id in threads:
        del threads[thread_id]
    if not threads:
        init_state()
    elif st.session_state[_ACTIVE_THREAD_KEY] == thread_id:
        st.session_state[_ACTIVE_THREAD_KEY] = next(iter(threads))


def get_active_thread() -> dict:
    return st.session_state[_THREADS_KEY][st.session_state[_ACTIVE_THREAD_KEY]]


def list_threads() -> list[dict]:
    """Most recently created first."""
    return sorted(
        st.session_state[_THREADS_KEY].values(),
        key=lambda t: t["created_at"],
        reverse=True,
    )


def add_message(role: str, content: str) -> None:
    thread = get_active_thread()
    thread["messages"].append({"role": role, "content": content})
    # Auto-title the thread from the first user message.
    if role == "user" and thread["title"] == "New chat":
        thread["title"] = (content[:40] + "…") if len(content) > 40 else content


def as_lc_history(thread: dict | None = None, exclude_last: bool = False) -> list[tuple[str, str]]:
    """Return history as (role, content) tuples for LangChain's MessagesPlaceholder.
    Set exclude_last=True when the latest user message hasn't been answered yet
    and shouldn't be duplicated into the prompt's {question} slot."""
    thread = thread or get_active_thread()
    msgs = thread["messages"][:-1] if exclude_last and thread["messages"] else thread["messages"]
    return [(m["role"], m["content"]) for m in msgs]
