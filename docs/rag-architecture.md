# RAG Architecture

This document describes the Retrieval Augmented Generation (RAG) system implemented in `backend/app/services/`.

---

## How It Works — End to End

```
                        INDEXING PIPELINE (on manual upload)
        ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
        │  PDF/TXT │───▶│ Text Extract │───▶│  Chunking    │───▶│  Embedding  │
        │  Upload  │    │ (pdfplumber/ │    │ (LangChain   │    │ (HuggingF-  │
        └──────────┘    │  PyPDF2)     │    │  splitter)   │    │  ace MiniLM)│
                        └──────────────┘    └──────────────┘    └──────┬──────┘
                                                                        │
                                                                        ▼
                                                               ┌─────────────────┐
                                                               │  Qdrant Cloud   │
                                                               │  (vector store) │
                                                               └─────────────────┘

                        QUERY PIPELINE (on chat message)
        ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
        │  User    │───▶│  Embed query │───▶│  Similarity  │───▶│ Dedup top-k │
        │  Query   │    │  (MiniLM)    │    │  Search +    │    │  chunks     │
        └──────────┘    └──────────────┘    │  Qdrant      │    │ (filtered)  │
                                            │  filter      │    └──────┬──────┘
                                            └──────────────┘           │
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │  Prompt Assembly │
                                                              │  context+question│
                                                              └────────┬─────────┘
                                                                       │
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │  LLM             │
                                                              │  Gemini / Groq   │
                                                              └────────┬─────────┘
                                                                       │
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │  Answer +        │
                                                              │  Source Citations│
                                                              └──────────────────┘
```

---

## Indexing Pipeline

### 1. Text Extraction (`DocumentProcessor`)

**File:** `app/services/document_processor.py`

Supports PDF and plain text files. For PDFs, an advanced extraction pipeline is used:

- **Standard Text & Tables**: Uses `pdfplumber` first. It features complex multi-column table detection—if a lossy or complex layout is detected, it automatically falls back to raw page text to prevent data loss, while preserving markdown formatting for simple tables.
- **Fallback**: Uses `PyPDF2` if `pdfplumber` fails or returns empty text.
- **Image OCR**: Integrates `pymupdf` to extract images larger than 100x100 pixels. Uses `easyocr` (with models stored in `E:\easyocr_models`) to perform OCR. Extracted text > 20 characters is formatted as a distinct chunk with `"source": "image_ocr"` metadata.

After extraction the text is cleaned: strips blank lines, collapses consecutive newlines.

The first 50 lines are also scanned for **auto-metadata hints**:
- Detects `model` mentions → stored as `detected_model`
- Detects section type from keywords: `troubleshooting`, `installation`, `user guide`

### 2. Chunking (`RecursiveCharacterTextSplitter`)

**File:** `app/services/rag_service.py` — `_initialize()`

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,       # configurable via CHUNK_SIZE env var
    chunk_overlap=200,     # configurable via CHUNK_OVERLAP env var
    separators=["\n\n", "\n", " ", ""]
)
```

Splits by double-newline first (paragraph boundary), then single newline, then space, then characters — so chunks respect document structure.

### 3. Metadata per Chunk

Each chunk is stored with flat top-level payload fields:

```python
{
    "source_file": "Samsung_TV_Q7F_Manual.pdf",
    "device_type": "TV",
    "brand":       "Samsung",
    "model":       "Q7F",            # or "Unknown" if not provided
    "document_id": "<uuid>",
    "chunk_index": 0,
    "total_chunks": 42,
    "source":      "text",           # or "image_ocr" for extracted images
    "page":        12,               # Page number of extraction
    # + auto-detected: section_type, detected_model
}
```

> **Note — Dual Payload Schema:** The collection also contains points that were migrated from an older ChromaDB instance. Those points nest all fields under a `metadata` key (e.g. `metadata.device_type`). Both schemas are fully supported by the filter and retrieval logic.

### 4. Embedding

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — runs locally on CPU  
**Dimension:** 384  
**Normalization:** L2-normalized (`normalize_embeddings=True`)

### 5. Batched Upsert to Qdrant Cloud

**File:** `app/services/rag_service.py` — `add_documents()`

Large documents are uploaded in **batches of 25 chunks** to avoid Qdrant Cloud timeout limits. Each batch goes through two phases:

1. **Embed** — compute 384-dim vectors from text (CPU-bound, no network)
2. **Upsert** — push pre-computed `PointStruct` objects to Qdrant (fast network call)

Upsert is retried **up to 3 times** (5-second pause) before failing the batch. Embeddings are cached per batch so they are never recomputed on a retry.

Payload indexes are also created idempotently on startup for all filterable fields (`device_type`, `brand`, `model`, `document_id` — and their `metadata.*` equivalents) to enable efficient `must`/`should` filtering.

---

## Query Pipeline

### 1. Metadata Filtering

The user can optionally narrow results by `device_type`, `brand`, and/or `model` via the Device Selector UI. These map to Qdrant `FieldCondition` filters.

Because two payload schemas coexist, each filter field matches **either** schema via a `should` (OR) sub-clause. Multiple filters are wrapped in a `must` (AND) block:

```python
Filter(must=[
    Filter(should=[
        FieldCondition(key="device_type",          match=MatchValue(value="TV")),
        FieldCondition(key="metadata.device_type", match=MatchValue(value="TV")),
    ]),
    Filter(should=[
        FieldCondition(key="brand",          match=MatchValue(value="Samsung")),
        FieldCondition(key="metadata.brand", match=MatchValue(value="Samsung")),
    ]),
])
```

### 2. Similarity Search

```python
vector_store.similarity_search_with_score(query, k=top_k * 2, filter=qdrant_filter)
```

- Default `top_k = 5` (configurable via `RETRIEVAL_TOP_K`)
- Fetches `2× top_k` candidates so the deduplication step below has room to work
- Qdrant uses **cosine similarity**; scores are returned directly as relevance values (0–1)
- Chunks below the relevance threshold (`0.3` default, via `RELEVANCE_THRESHOLD`) are discarded

All blocking Qdrant calls are run in a thread executor (`loop.run_in_executor`) to keep the FastAPI async loop unblocked.

### 3. Deduplication

After retrieval, chunks whose first 150 characters share more than **70% word-level overlap** with an already-accepted chunk are dropped. This prevents five nearly-identical consecutive chunks from the same manual page from filling the entire context window.

### 4. Prompt Assembly

Passing chunks are formatted with inline metadata so the LLM can prioritise them:

```
[Excerpt 1 | Source: Samsung_TV_Q7F_Manual.pdf | Page: 12 | Section: Troubleshooting | Relevance: 85%]
<chunk text>

---

[Excerpt 2 | Source: Samsung_TV_Q7F_Manual.pdf | Page: 13 | Section: Troubleshooting | Relevance: 78%]
<chunk text>
```

### 5. Prompt Template

The system prompt instructs the LLM to behave as a **friendly device assistant**:

- Answer **only** from the manual excerpts provided
- Write in a natural, conversational tone — not like a technical document
- Use a clean numbered list for step-by-step answers
- Add a single `📖 Source:` line at the end — no mid-sentence citations
- Include safety tips if mentioned in the excerpts
- Honestly admit when the excerpts don't contain enough information
- **Never invent information** not present in the excerpts

### 6. LLM Generation

**File:** `app/services/rag_service.py` — `generate_answer()`

Two LLM providers are supported, selected per request via the `ai_model` parameter (set by the model selector dropdown in the UI):

| Provider | Model | Config key | Notes |
|---|---|---|---|
| `gemini` (default) | configurable via `LLM_MODEL` | `GOOGLE_API_KEY` | Primary LLM |
| `groq` | `llama-3.3-70b-versatile` (via `GROQ_MODEL`) | `GROQ_API_KEY` | Falls back to Gemini if not configured |

Default generation settings:
```
temperature = 0.3   (low = more factual, less creative)
max_tokens  = 1000
```

Uses `llm.ainvoke(prompt)` — fully async.

### 7. Response

Returns the answer plus up to **5 source citations**:

```json
{
  "answer": "...",
  "sources": [
    {
      "content": "first 200 chars of chunk...",
      "source_file": "Samsung_TV_Q7F_Manual.pdf",
      "page_number": 12,
      "section_name": "Troubleshooting",
      "relevance_score": 0.847
    }
  ]
}
```

---

## Document Lifecycle in MongoDB

```
Upload → PENDING → PROCESSING → INDEXED
                              → FAILED (with error_message)
```

`ManualDocument` stores: `filename`, `device_type`, `brand`, `model`, `file_path`, `file_size`, `status`, `chunks_count`, `processed_at`.

The `/api/v1/devices` endpoint reads this collection dynamically to populate the Device Selector dropdowns — no separate `DeviceCategory` collection is needed.

---

## RAG Evaluation

An offline evaluation harness is available to measure retrieval and generation quality without a live browser session:

**File:** `backend/evaluation/evaluate_rag.py`

```powershell
cd backend
python evaluation/evaluate_rag.py
```

- Reads ground-truth Q&A pairs from `evaluation/eval_questions_real.json`
- Calls `rag_service.generate_answer()` directly for each question
- Generates comprehensive, aggregated performance reports summarizing retrieval hit rates, answer relevance, and latency, rather than just granular, per-question logs.
- Writes detailed results to `evaluation/results/rag_eval_<timestamp>.json`

Use `evaluation/discover_indexed_data.py` to inspect what documents and device categories are currently indexed in Qdrant.

---

## Configuration Reference

All values are overridable via `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension (must match model) |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `5` | Final number of chunks returned per query |
| `RELEVANCE_THRESHOLD` | `0.3` | Min cosine relevance score to keep a chunk |
| `LLM_PROVIDER` | `gemini` | Primary provider: `gemini` |
| `LLM_MODEL` | `gemini-*` | Gemini model name |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | `1000` | Max tokens in LLM response |
| `QDRANT_URL` | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION_NAME` | `device_manuals` | Qdrant collection name |

---

## Key Files

```
backend/app/services/
├── rag_service.py          # RAGService — embeddings, Qdrant, LLM selection, retrieval, dedup
└── document_processor.py   # DocumentProcessor — PDF extraction, chunking, indexing

backend/app/api/
├── chat.py                 # POST /api/v1/chat — calls rag_service.generate_answer()
├── documents.py            # POST /api/v1/upload-manual — calls DocumentProcessor
└── devices.py              # GET  /api/v1/devices — dynamic lookup from ManualDocument

backend/app/core/
└── config.py               # All RAG settings (Qdrant, LLM, chunk, threshold, etc.)

backend/evaluation/
├── evaluate_rag.py          # Offline evaluation harness
├── eval_questions_real.json # Ground-truth Q&A pairs
└── discover_indexed_data.py # Inspect indexed Qdrant data

backend/scripts/
└── migrate_chroma_to_qdrant.py  # One-time ChromaDB → Qdrant migration
```
