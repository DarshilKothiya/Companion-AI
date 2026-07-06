# Installation Guide

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **MongoDB 7.0+** — [Download](https://www.mongodb.com/try/download/community)
- **Google Gemini or Groq API Key**
- **Qdrant Cloud Account** (Free tier works)

---

## Step 1 — Start MongoDB

Install MongoDB and start it locally:

```powershell
mongod --dbpath="C:\data\db"
```

Default connection: `mongodb://localhost:27017`

---

## Step 2 — Backend Setup

```powershell
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure `.env`

Copy the example and fill in your values:

```powershell
copy .env.example .env
```

Minimum required settings:

```env
GOOGLE_API_KEY=your-google-api-key-here
GROQ_API_KEY=your-groq-api-key-here
SECRET_KEY=any-random-string-at-least-32-chars
MONGODB_URL=mongodb://localhost:27017
QDRANT_URL=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=device_manuals
```

### Start the Backend API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Step 3 — Frontend Setup

```powershell
cd frontend
npm install
npm start
```

---

## Access

| Service | URL |
|---|---|
| Chat UI | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/v1/health |

---

## Initial Data Setup

Device categories are auto-populated when you upload manuals via the UI.

To manually seed categories via the MongoDB shell:

```javascript
use device_troubleshoot

db.device_categories.insertMany([
  { name: "Refrigerator", brands: ["Samsung", "LG", "Whirlpool"], models: {}, created_at: new Date(), updated_at: new Date() },
  { name: "Washing Machine", brands: ["Samsung", "LG", "Bosch"], models: {}, created_at: new Date(), updated_at: new Date() },
  { name: "Air Conditioner", brands: ["Samsung", "LG", "Daikin"], models: {}, created_at: new Date(), updated_at: new Date() },
  { name: "TV", brands: ["Samsung", "LG", "Sony"], models: {}, created_at: new Date(), updated_at: new Date() },
  { name: "Dishwasher", brands: ["Bosch", "Whirlpool", "Samsung"], models: {}, created_at: new Date(), updated_at: new Date() }
])
```

---

## Troubleshooting

### Backend won't start
- Verify MongoDB is running: `mongosh --eval "db.stats()"`
- Double-check `.env` values (especially `GOOGLE_API_KEY` or `GROQ_API_KEY` and Qdrant credentials)

### Frontend can't connect
- Confirm backend is on port 8000
- Check `REACT_APP_API_URL` in `frontend/.env` (if set)

### Document processing fails
- Ensure Qdrant Cloud credentials are correct and collection exists
- Verify your API keys (Gemini/Groq) are active and have sufficient quotas
- Check backend logs for specific errors

---

## Deployment

### Backend (Hugging Face Spaces)
1. Create a new Docker Space on Hugging Face.
2. Upload the `backend/` directory contents (including the `Dockerfile`).
3. Set all `.env` variables as Space Secrets (e.g., `MONGODB_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `GOOGLE_API_KEY`).
4. The space will build and run the FastAPI server automatically.

### Frontend (Vercel)
1. Import the `frontend/` directory into Vercel.
2. Set the `REACT_APP_API_URL` environment variable to your Hugging Face Space URL.
3. Deploy! Vercel handles the React build process automatically.
