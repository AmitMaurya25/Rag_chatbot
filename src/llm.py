"""
LLM + RAG chain.

This extends the notebook's "Augmentation" and "Generation" steps:
    prompt = PromptTemplate(...)
    llm = ChatOpenAI(...)
    main_chain = parallel_chain | prompt | llm | parser

Two differences from the notebook:
1. The model is served via Groq's API (langchain-groq) instead of OpenAI.
2. Follow-up questions are first "condensed" into a standalone question
   using the chat history, so pronouns like "what about that?" resolve
   correctly against the retriever (a bare RAG chain like the notebook's
   has no concept of conversation history).
3. The final answer is streamed token-by-token for use with
   st.write_stream in the UI.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rephrase the "
            "follow-up question into a standalone question that makes sense without "
            "the history. If the follow-up question is already standalone, return it "
            "unchanged. Return ONLY the rewritten question, nothing else.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about documents the user "
            "has uploaded. Answer ONLY using the provided context below. If the context "
            "is insufficient to answer, say you don't know rather than guessing. Be "
            "concise and cite which source file you're drawing from when relevant.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


def get_llm(streaming: bool = True) -> ChatGroq:
    return ChatGroq(
        model=settings.chat_model,
        api_key=settings.groq_api_key,
        temperature=settings.temperature,
        streaming=streaming,
        max_tokens=1024,
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def condense_question(question: str, chat_history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up question into a standalone one, using chat history.
    Skips the LLM call entirely if there's no history yet (saves a round trip)."""
    if not chat_history:
        return question

    llm = get_llm(streaming=False)
    chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    standalone = chain.invoke({"chat_history": chat_history, "question": question})
    log.info("Condensed question: %r -> %r", question, standalone)
    return standalone.strip()


def retrieve_context(retriever, question: str) -> list[Document]:
    return retriever.invoke(question)


def stream_answer(
    question: str,
    chat_history: list[tuple[str, str]],
    context_docs: list[Document],
) -> Iterator[str]:
    """Yields the answer token-by-token for use with st.write_stream."""
    llm = get_llm(streaming=True)
    chain = ANSWER_PROMPT | llm | StrOutputParser()

    inputs = {
        "context": _format_docs(context_docs),
        "chat_history": chat_history,
        "question": question,
    }

    for chunk in chain.stream(inputs):
        yield chunk
