# Companion AI — Device Troubleshooting Chatbot

An AI-powered chatbot that helps users troubleshoot device problems using RAG (Retrieval Augmented Generation). Upload device manuals and get intelligent, source-cited answers — with support for multiple LLM providers.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Axios, React Markdown, Lucide React |
| Backend | FastAPI (Python 3.13), Uvicorn |
| Database | MongoDB (Beanie ODM + Motor) |
| Vector Store | **Qdrant Cloud** |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| AI / LLM | LangChain + **Google Gemini / Groq** (switchable) |
| Auth | JWT (`python-jose`) + bcrypt |
| Document Parsing | pdfplumber, PyPDF2, pymupdf, easyocr, pillow |

## Project Structure

```
Companion AI/
├── backend/
│   ├── app/
│   │   ├── api/              # Route handlers
│   │   │   ├── auth.py       # Register / login / JWT
│   │   │   ├── chat.py       # Chat endpoint (multi-model)
│   │   │   ├── devices.py    # Device/brand/model lookup (dynamic from MongoDB)
│   │   │   ├── documents.py  # Manual upload & Qdrant indexing
│   │   │   ├── feedback.py   # Thumbs up/down per message
│   │   │   └── health.py     # Health check (MongoDB + Qdrant)
│   │   ├── core/
│   │   │   ├── config.py     # Pydantic settings (all env vars)
│   │   │   ├── database.py   # MongoDB / Beanie connection
│   │   │   └── auth.py       # JWT helpers, password hashing
│   │   ├── models/
│   │   │   ├── database.py   # Beanie documents (User, Chat, ManualDocument, …)
│   │   │   └── schemas.py    # Pydantic request / response schemas
│   │   ├── services/
│   │   │   ├── rag_service.py          # RAG pipeline (Qdrant retrieval + LLM generation)
│   │   │   └── document_processor.py  # PDF/TXT parsing & chunking
│   │   └── main.py           # FastAPI app, middleware, router registration
│   ├── evaluation/
│   │   ├── evaluate_rag.py           # End-to-end RAG evaluation script
│   │   ├── eval_questions_real.json  # Ground-truth Q&A pairs
│   │   ├── discover_indexed_data.py  # Inspect what's in Qdrant
│   │   ├── indexed_catalog.json      # Cached catalog of indexed documents
│   │   └── results/                  # Evaluation output JSON files
│   ├── scripts/
│   │   └── migrate_chroma_to_qdrant.py  # One-time ChromaDB → Qdrant migration
│   ├── data/
│   │   ├── uploads/     # Raw uploaded manuals
│   │   └── processed/   # Post-processing artefacts
│   ├── test_rag.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Auth.js / Auth.css            # Login & register forms
│   │   │   ├── ChatInterface.js / .css        # Main chat window + model selector
│   │   │   ├── DeviceSelector.js / .css       # Device / brand / model dropdowns
│   │   │   ├── MessageBubble.js / .css        # Individual message rendering (Markdown)
│   │   │   ├── Sidebar.js / .css             # Chat history sidebar with toggle
│   │   │   └── TypingIndicator.js / .css     # Animated typing indicator
│   │   ├── services/
│   │   │   └── api.js    # Axios API client
│   │   ├── App.js
│   │   └── index.css
│   └── package.json
├── docs/
│   ├── installation.md      # Full setup guide
│   └── rag-architecture.md  # RAG pipeline deep-dive
└── README.md
```

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB running on `localhost:27017`
- [Qdrant Cloud](https://cloud.qdrant.io) account (free tier works)
- At least one LLM API key (Google Gemini or Groq)

### 1. Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
# App
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=device_troubleshoot

# Qdrant Cloud
QDRANT_URL=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=device_manuals

# LLM Providers (add at least one)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-google-key
GROQ_API_KEY=your-groq-key

# Optional tuning
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
```

Start the API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```powershell
cd frontend
npm install
npm start
```

### Access

| Service | URL |
|---|---|
| Chat UI | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/v1/health |

## Features

- **Multi-model LLM support** — switch between Google Gemini and Groq (Llama 3.3) per chat session from a dropdown in the UI
- **RAG-based responses** with source citations pulled from uploaded device manuals
- **PDF & TXT manual upload** with automatic Qdrant indexing and payload filtering
- **Advanced Document Processing** — includes complex multi-column table detection and OCR image extraction to prevent data loss
- **Clean Chat UI** — Device/brand/model filtering configuration is integrated directly into the sidebar to keep the chat workspace uncluttered
- **LLM-Generated Chat Titles** — automatically summarizes the first query to generate a descriptive title, persisted in MongoDB
- **Collapsible sidebar** with full conversation history
- **Typing indicator** — animated dots while the AI is generating a response
- **User feedback** — thumbs up / down on every message
- **JWT authentication** — register, login, protected endpoints
- **Health check endpoint** — reports MongoDB and Qdrant connectivity

## RAG Evaluation

An offline evaluation harness is included to measure retrieval and generation quality:

```powershell
cd backend
python evaluation/evaluate_rag.py
```

- Reads ground-truth questions from `evaluation/eval_questions_real.json`
- Queries the live RAG pipeline and scores responses
- Generates comprehensive, aggregated performance reports to `evaluation/results/rag_eval_<timestamp>.json`
- Metrics include: retrieval hit rate, answer relevance, latency

## Migration (ChromaDB → Qdrant)

If you have data in an older ChromaDB instance, a one-off migration script is provided:

```powershell
cd backend
python scripts/migrate_chroma_to_qdrant.py
```

## Documentation

| File | Description |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Full setup guide |
| [`docs/rag-architecture.md`](docs/rag-architecture.md) | RAG pipeline deep-dive: embeddings, chunking, Qdrant retrieval, LLM generation |
