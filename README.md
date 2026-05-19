# Simple RAG System 

Overview
- Ingests PDFs from the data/ folder or via the API upload endpoint.
- Splits text into chunks, creates embeddings, and stores them in a local Qdrant database.
- Uses Google Gemini for answer generation with retrieved context.

What runs locally
- Qdrant is embedded and stored on disk at storage/qdrant/.
- Embeddings are computed locally using sentence-transformers.
- PDFs are stored locally in data/.

Requirements
- Python 3.10+
- A Google Gemini API key

Setup
1) Create and activate a virtual environment.
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Create a .env file with your key:
```
GOOGLE_API_KEY=your_key_here
```

Run the API
From the repository root:
```
uvicorn src.interfaces.api:app --reload
```

Key endpoints
- GET /health
- GET /documents
- POST /upload
- POST /ask

Notes
- Put PDFs in data/ to ingest them, or upload via /upload.
- The local vector store lives in storage/qdrant/.
