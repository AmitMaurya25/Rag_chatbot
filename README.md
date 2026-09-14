# RAG Chatbot (Streamlit + Gemini + FAISS)

A document-grounded chatbot: upload PDFs/DOCX/TXT, ask questions, get
streamed answers grounded in your documents, with multi-thread chat
history for the session.

## Architecture

```
rag-chatbot/
├── app.py                  # Streamlit UI — entry point
├── src/
│   ├── config.py            # env/config loading (.env)
│   ├── logger.py            # rotating file + console logging
│   ├── document_loader.py   # PDF/DOCX/TXT loading + chunking
│   ├── vectorstore.py       # FAISS build/load/persist (Gemini embeddings)
│   ├── llm.py                # Gemini chat model + RAG chain + streaming
│   └── memory.py             # in-memory multi-thread chat history
├── data/
│   ├── uploads/              # saved copies of uploaded files
│   └── vectorstore/          # persisted FAISS index (index.faiss/.pkl)
├── logs/                     # rotating app.log
├── requirements.txt
├── .env.example               # copy to .env and fill in your key
└── .gitignore
```

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key**:
   - Copy `.env.example` to `.env`
   - Get a Gemini API key from https://aistudio.google.com/apikey
   - Set `GOOGLE_API_KEY=your_actual_key` in `.env`
   - Never commit `.env` — it's already in `.gitignore`

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   This opens the app at `http://localhost:8501`.

## How it works

1. **Upload & index**: Files are chunked with `RecursiveCharacterTextSplitter`
   (configurable via `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env`), embedded with
   Gemini's embedding model, and stored in a FAISS index persisted to
   `data/vectorstore/`. Uploading more files later adds to the same index.
2. **Ask a question**: If chat history exists, the follow-up question is
   first rewritten into a standalone question (so "what about the second
   one?" resolves correctly) before retrieval.
3. **Retrieve**: Top-k similar chunks are pulled from FAISS
   (`RETRIEVER_TOP_K` in `.env`).
4. **Generate**: Gemini answers using only the retrieved context, streamed
   token-by-token into the chat UI via `st.write_stream`.
5. **No documents yet?** The app falls back to plain conversational Gemini
   chat until you upload something.

## Chat memory

Per your setup choice, conversation history is **in-memory only** — it
lives in Streamlit's `session_state` and resets when the app restarts.
Within a running session you can keep multiple chat threads (sidebar →
"New chat"), and each is tracked independently. The uploaded document
index (FAISS), by contrast, **is** persisted to disk in `data/vectorstore/`
so you don't have to re-upload files every restart — only the chat
threads reset.

If you later want chat history to survive restarts too, swap `src/memory.py`
for a SQLite-backed version — the rest of the app doesn't need to change,
since `app.py` only calls the functions in `memory.py`.

## Production notes

- Logs rotate at 5MB in `logs/app.log`.
- `GOOGLE_API_KEY` is validated on startup; the app refuses to run without it.
- Uploaded files are size-capped (`MAX_UPLOAD_MB`, default 50MB).
- The Gemini model name is configurable in `.env`
  (`GEMINI_CHAT_MODEL`) — default is `gemini-2.5-flash`. Google's model
  lineup moves fast; check https://ai.google.dev/gemini-api/docs/models
  if you hit a deprecation notice.
- `allow_dangerous_deserialization=True` is used when loading the FAISS
  index — this is safe because the index is only ever written by this
  same app, never loaded from an untrusted source. Don't point
  `VECTORSTORE_DIR` at a directory you didn't create yourself.
