# DocuMind – Intelligent Document Understanding System

DocuMind is a standalone application for understanding documents: upload PDF, DOCX or TXT files, then summarize them, ask questions, extract key information and compare them. All AI processing runs on your own computer. It needs no API keys or cloud accounts.

> **Status: Phase 2 (document ingestion).** You can upload, validate, extract and store documents. Summaries, question answering, comparison and insights are not implemented yet, so their modules are still placeholders. The full README is written in Phase 8.

## Architecture

The code has four layers, and dependencies only point downward:

```
ui/         Streamlit pages. Display and input only.
  │
core/       Document processing, retrieval and AI logic. Never imports Streamlit.
  │
database/   SQLite metadata: documents, chunks, chat history.
  │
utils/      Pure helpers. Imports nothing from the layers above.

config.py   Every setting. Any layer may import it.
```

### Modules

| Module | Responsibility |
| --- | --- |
| `core/document_loader.py` | Extract text and structure from PDF (PyMuPDF), DOCX and TXT, keeping page numbers |
| `core/text_cleaner.py` | Normalize whitespace, control characters and paragraphs |
| `core/ingestion.py` | Run an upload through validation, storage, extraction and metadata recording |
| `core/chunker.py` | Split text into sentence-aligned, overlapping chunks with source metadata |
| `core/embedding_engine.py` | Load the embedding model once; embed chunks and queries |
| `core/vector_store.py` | Own the FAISS index and NumPy fallback: add, remove, search, persist, rebuild |
| `core/retriever.py` | Turn a query into ranked chunks, with document filtering and a relevance floor |
| `core/keyword_extractor.py` | TF-IDF keywords and keyphrases; NMF topics |
| `core/ollama_client.py` | Talk to the local Ollama service: status, model list, generation |
| `core/summarizer.py` | Quick, detailed and executive summaries, by LLM or extractive fallback |
| `core/question_answering.py` | Evidence-grounded answers with citations, or an explicit refusal |
| `core/document_comparator.py` | Semantic, lexical and numeric comparison of two documents |
| `database/schema.py` | Table definitions and schema version |
| `database/db_manager.py` | Connections, migrations and all metadata reads and writes |
| `ui/*.py` | One module per page, plus shared `components.py` |
| `utils/file_validation.py` | Extension, size and file signature checks; safe storage paths |
| `utils/text_utils.py` | Sentence splitting and token estimation |
| `utils/logging_config.py` | Rotating local log files |
| `utils/helpers.py` | Hashing, IDs and formatting |

### Data flow

**Ingestion**

```
upload → validate → SHA-256 duplicate check → store file → extract → clean
       → chunk → embed → add to FAISS → mark document "ready" in SQLite
```

**Question answering**

```
question → embed → FAISS top-K → drop results below the relevance floor
  ├─ no evidence left    → "not enough evidence in the document"
  ├─ Ollama available    → grounded prompt → answer + citations   (labelled LLM)
  └─ Ollama unavailable  → best-matching sentences + citations    (labelled extractive)
```

### Storage

| Data | Location | Notes |
| --- | --- | --- |
| Original files | `uploads/` | Stored under the document ID, never the user's filename |
| Extracted text, chunks, embeddings | `processed/<document_id>/` | JSON and NumPy `.npy`, no pickle |
| Vector index | `indexes/` | Derived data, rebuilt from `processed/` when needed |
| Metadata and chat history | `documind.db` | SQLite; holds no full document text |

**SQLite is the source of truth; FAISS is a rebuildable cache.** Vectors use integer IDs stored against each chunk row in an `IndexIDMap2` over `IndexFlatIP`. That index is exact, equals cosine similarity on normalized vectors, and supports deletion. A document is marked `ready` only after its vectors are saved, and the index is replaced atomically. At startup the index is checked against SQLite and rebuilt if they differ.

### Key design decisions

- **Data stays out of cloud-synced folders.** If the project is inside OneDrive, Dropbox, Google Drive or iCloud, data is stored in `%LOCALAPPDATA%\DocuMind\data` instead of `data/`. Sync clients would otherwise upload documents to their servers and can corrupt a live SQLite file. Set `DOCUMIND_DATA_DIR` to choose a different location.
- **Long chunks are embedded in windows.** `all-MiniLM-L6-v2` reads at most 256 tokens, so a 500-token chunk would be silently cut in half. Each chunk is embedded in windows of up to 256 tokens, then pooled and normalized. The LLM still receives the full chunk.
- **Ollama is restricted to loopback.** `config.py` rejects any Ollama URL that does not point at this machine.
- **Streamlit telemetry is off** and the server binds to `localhost` (see `.streamlit/config.toml`).

## Configuration

Settings live in `config.py`. These environment variables override them:

| Variable | Default |
| --- | --- |
| `DOCUMIND_DATA_DIR` | `data/`, or `%LOCALAPPDATA%\DocuMind\data` in a cloud-synced folder |
| `DOCUMIND_MAX_FILE_SIZE_MB` | `50` (also change `maxUploadSize` in `.streamlit/config.toml`) |
| `DOCUMIND_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `DOCUMIND_OLLAMA_URL` | `http://localhost:11434` (loopback only) |
| `DOCUMIND_OLLAMA_MODEL` | `qwen2.5:3b` |

## Setup (Windows)

Requires Python 3.12. If the project is in a cloud-synced folder, keep the virtual environment outside it, as shown here, so its files are not synced.

```powershell
py -3.12 -m venv "$env:LOCALAPPDATA\DocuMind\.venv"
& "$env:LOCALAPPDATA\DocuMind\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
streamlit run app.py
```

`py -3.12` finds python.org installs only. If it reports "No suitable Python runtime found", run `py --list` and use the listed tag instead, for example `py -V:Astral/CPython3.12.14 -m venv ...`.

Run the tests with `pytest`. Sample documents are generated during the run.

The embedding model (about 90 MB) downloads from Hugging Face on first use, then runs from the local cache. That download contains no document data.
