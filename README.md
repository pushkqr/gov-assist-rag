<div align="center">
  <img src="assets/mimir-hero.svg" alt="Mimir Logo" width="100%"/>
</div>

# Mimir Engine

Mimir is an **Extensible AI-powered Retrieval-Augmented Generation (RAG) Engine** designed as the foundational backend for deploying secure, citation-backed conversational interfaces across government intranets.

Built for flexibility, Mimir separates the core AI retrieval logic from the frontend presentation layer. While the repository includes a reference implementation (e.g., an Officer Portal), the engine itself is completely department-agnostic. By simply swapping the frontend stylesheet and connecting a different Weaviate collection, Mimir can instantly power dedicated portals for the Department of Finance, Health, Police, or Revenue—requiring zero backend code changes.

Named after the Norse figure who guarded the Well of Wisdom, Mimir represents the institutional memory and secure intelligence infrastructure of the modern digital government.

---

## Key Features & Capabilities

- **Lightweight, High-Performance Architecture**: 
  - Entirely powered by **FastAPI** for lightning-fast backend endpoints.
  - A beautiful, zero-dependency vanilla JS/CSS frontend with native **Dark Mode** and mobile-responsive layouts.
  - Streamed Server-Sent Events (SSE) for real-time answer generation.

- **Agentic RAG Pipeline**:
  - Uses Google Gemini for query understanding, dense embeddings (`gemini-embedding-001`), and generative answering.
  - **Hybrid Search Engine**: Combines **Dense Vector Search** with **BM25 Sparse Keyword Search**, merged natively in **Weaviate** using Alpha Fusion for unparalleled retrieval accuracy.

- **Enterprise-Grade Security & Authentication**:
  - **Zero-Trust Intranet Geofencing**: The middleware mathematically validates incoming network requests against authorized government subnets (e.g., `10.0.0.0/8`). Requests from public networks are dropped at the perimeter.
  - **Token-Based Identity**: A built-in SQLite token registry completely replaces vulnerable passwords. Chat histories are securely mapped to hashed officer tokens.
  - **Admin token CRUD API**: Fully baked API for IT departments to provision, audit, and revoke officer tokens programmatically.

- **Cross-Device Persistent Sessions**:
  - Chat histories are saved securely to the local SQLite database instead of browser storage. Officers can seamlessly transition between desktop and mobile on the government intranet without losing conversation context.

---

## Architecture

### System Flow
The following sequence demonstrates how a user's query is processed from ingestion through to the streamed response:

```mermaid
sequenceDiagram
    participant User as User / Frontend
    participant API as FastAPI Backend
    participant Search as Hybrid Search Engine
    participant LLM as Generative LLM

    User->>API: Submits Query
    API->>LLM: Generate Query Variations
    LLM-->>API: Returns Variations
    API->>Search: Embed Queries & Run Hybrid Search (Dense + BM25)
    Search-->>API: Top K Relevant Documents (Alpha Fusion)
    API->>LLM: Build Prompt with Context & Query
    LLM-->>API: Stream Answer (Server-Sent Events)
    API-->>User: Stream Response & Citations to UI
```

### Component Architecture
This diagram outlines the core services and their interactions:

```mermaid
graph TD
    subgraph Frontend
        UI[Vanilla JS/CSS UI]
    end

    subgraph Backend [FastAPI Server]
        API[Core Endpoints]
        Router[LLM Router & Rate Limiter]
        Ingest[Ingestion Pipeline]
        Ret[Retrieval Pipeline]
    end

    subgraph Data & Services
        Weaviate[(Weaviate DB)]
        Translation[Translation Service]
    end

    subgraph External APIs
        LLM[GenAI / LLM Providers]
        DocAI[OCR / Document Parsing]
    end

    UI <-->|HTTP / SSE| API
    API --> Ret
    API --> Ingest
    Ret --> Router
    Ingest --> Router
    Router <--> LLM
    Ingest <--> DocAI
    Ingest -->|Chunk & Embed| Weaviate
    Ret <-->|Hybrid Search| Weaviate
    Ingest <--> State
    Ret <--> Translation
    Ingest <--> Translation
```

### Directory Tree

```text
rag/
├── main.py                             # CLI entry point (Ingestion, Retrieval, Benchmark)
├── app.py                              # FastAPI server and core endpoints (/ask, /workspaces)
├── requirements.txt                    # Python dependencies
│
├── templates/                          # Frontend UI (Vanilla HTML/JS/CSS)
│   ├── landing.html                    # Public-facing landing page
│   └── app.html                        # Authenticated chat interface
│
├── scratch/                            # Temporary & Persisted State
│   ├── ingestion_state.json            # File hashing & incremental ingestion state tracker
│   └── mimir_cache.json                # Persistent JSON query cache
│
├── core/                               # Shared Core Infrastructure
│   └── utils.py                        # API rate-limit retry, throttle, & LLM routing
│
├── ingestion/                          # Ingestion Pipeline
│   ├── pipeline.py                     # Ingestion orchestrator & hash skip logic
│   └── parsers.py                      # PyMuPDF4LLM -> Markdown parser sequence
│
├── retrieval/                          # Retrieval & Generation Pipeline
│   ├── pipeline.py                     # Hybrid search & streaming response orchestrator
│   ├── search.py                       # Alpha fusion, LLM reranking, & evidence extraction
│   └── query.py                        # Contextualization & multi-query expansion
│
└── benchmark/                          # Corpus Evaluation & Benchmark Harness
    ├── benchmark.json                  # Standardized 30-case grounded evaluation dataset
    └── runner.py                       # Benchmark execution & LLM judge runner
```

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- Google Gemini API Key
- Cerebras API Key (for LLM Generation)
- [Weaviate](https://weaviate.io/) (Can run locally or remotely)
- Translation Service (Can run locally or remotely)

### 2. Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/pushkqr/mimir.git
   cd mimir
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Deploy Weaviate (Local or Remote)**:
   Weaviate is deployed as a standalone microservice. You can run it locally or deploy it to a remote server using its dedicated compose file:
   ```bash
   cd data
   docker-compose up -d
   cd ..
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   Create a `.env` file in the root directory. You can copy the provided `.env.example`:

   ```env
   # Required: API keys
   GOOGLE_API_KEY=your_gemini_api_key
   CEREBRAS_API_KEY=your_cerebras_api_key
   
   # Remote / External Services (Optional depending on architecture)
   # Point these to your hosted microservices if deploying in a distributed environment
   TRANSLATION_SERVICE_URL=http://<YOUR_TRANSLATION_IP>:8000/translate
   WEAVIATE_URL=http://<YOUR_WEAVIATE_IP>
   WEAVIATE_GRPC_PORT=50051
   WEAVIATE_API_KEY=your-secure-weaviate-api-key

   # Model Configuration
   GEN_MODEL_NAME=gemini-2.5-flash
   EMBED_MODEL_NAME=gemini-embedding-001

   # Security
   # Set this to protect your app. Leave empty for public access.
   MIMIR_AUTH_TOKEN=your_secure_password
   
   # Admin token used for CRUD operations on officer tokens.
   MIMIR_ADMIN_TOKEN=SUPER-SECRET-ADMIN-TOKEN
   
   # Intranet Geofencing (Comma-separated CIDRs).
   # Use 0.0.0.0/0 to allow all public traffic for live demos.
   MIMIR_ALLOWED_SUBNETS=
   ```

---

## Usage

### Launching the Web Application

Start the FastAPI server via Uvicorn:

```bash
python app.py
# or
uvicorn app:app --reload
```
Navigate to `http://localhost:8000/` to view the landing page, or `http://localhost:8000/app` to access the chat interface.

### Ingestion & CLI Pipeline (`main.py`)

To ingest documents, test interactive CLI retrieval, or run corpus benchmarks, configure the execution flags in `main.py`:

```python
RUN_INGESTION = True    # Re-index PDFs in docs/
RUN_RETRIEVAL = False   # Run interactive CLI chat
RUN_BENCHMARK = False   # Run corpus benchmark evaluation
```

Then execute:

```bash
python main.py
```

---

## Disclaimer

Mimir is designed for administrative decision support. While it prioritizes strict retrieval-based grounding, always verify outputs against official published government circulars and gazette notifications.
