# SaaS Support RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built in Python for a SaaS technical support use case. This API ingests your internal Markdown/PDF documentation and accurately answers user queries while strictly preventing hallucinations.

## 🚀 Architecture

*   **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) - High performance, async, built-in validation.
*   **Orchestration**: [LangChain](https://python.langchain.com/) - Chains the LLM, prompts, and vector database.
*   **Vector Database**: [FAISS](https://faiss.ai/) - Local, blazing-fast vector similarity search capable of running on CPU.
*   **LLM & Embeddings**: OpenAI's `gpt-3.5-turbo` and `text-embedding-3-small`.
*   **Observability**: `structlog` for structured JSON logging (query times, token counts) perfectly suited for Datadog or ELK.

## 📁 Project Structure

```text
rag-saas-support/
├── api/             # FastAPI application and routes
│   ├── dependencies.py # Dependency injection
│   ├── main.py         # App entry point
│   └── routes.py       # API endpoints (GET /health, POST /query)
├── data/
│   ├── raw/         # Put your Markdown (.md) or .pdf files here
│   └── vectorstore/ # Output directory for FAISS local DB
├── scripts/
│   ├── ingest.py    # Script to populate your Vector DB
│   └── evaluate.py  # Script for offline QA evaluation
├── src/             # Core Business Logic
│   ├── config.py    # Environment variables
│   ├── ingestion/   # Document loaders and chunkers
│   ├── rag/         # Pipeline orchestration and prompts
│   ├── retrieval/   # FAISS VectorDB wrapper
│   └── utils/       # Logger and helpers
├── tests/           # Pytest unit and integration tests
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## 🛠️ Local Setup

1.  **Clone down the repository**
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment**:
    Copy `.env.example` to `.env` and add your OpenAI API Key.
    ```bash
    cp .env.example .env
    ```
5.  **Ingest Data**:
    Add some dummy PDF or Markdown files into `data/raw/` then run:
    ```bash
    python scripts/ingest.py
    ```

## 🏃‍♂️ Running the Services

### Option 1: Local Server
Start the API in one terminal:
```bash
uvicorn api.main:app --reload
```
Access the Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).

Start the Streamlit Frontend in another terminal:
```bash
streamlit run frontend/app.py
```
Access the Chat UI at [http://localhost:8501](http://localhost:8501).

### Option 2: Docker / Docker Compose
This runs both the API and the Streamlit frontend.
```bash
docker-compose up --build
```

## 🧪 Testing

Run tests locally:
```bash
pytest tests/ -v
```

## 🚢 Deployment Guide (Render)

This project contains an optimized, non-root `Dockerfile` ready for a PaaS like Render.

1.  Create a fresh GitHub repository and push this code.
    ```bash
    git init
    git add .
    git commit -m "feat: initial commit for RAG support API"
    git branch -M main
    git remote add origin https://github.com/your-username/repo-name.git
    git push -u origin main
    ```
2.  Go to [Render.com](https://render.com) -> New Web Service.
3.  Connect the GitHub repository.
4.  Render will automatically detect the `Dockerfile`.
5.  In the Environment Variables section in Render, add:
    *   `OPENAI_API_KEY`: `your_key_here`
    *   `LOG_LEVEL`: `INFO`
6.  *Crucial Step for Production DB*: Since FAISS saves to disk locally, consider Render's "Disk" feature mounted at `/app/data` to persist your vectorstore across deploys. Alternatively, swap `VectorDB` to use a cloud provider like Pinecone or Qdrant for stateless horizontal scaling.

## 📈 Production Features (Bonus)
*   **Cost Tracking**: Every query response metadata logs the token usage estimation.
*   **JSON logging**: Look at the console output; logs are formatted uniquely to be ingested by log aggregators.
*   **Hallucination Prevention**: Prompts explicitly block answering outside the provided context.
