# GovAssist

GovAssist is an AI-powered Retrieval-Augmented Generation (RAG) system and interactive chat surface designed specifically for querying, comparing, and analyzing government policy documents, circulars, and notifications.

Built for citation-backed grounding and high-precision retrieval, responses are synthesized strictly from indexed local government documents, eliminating hallucinated policy text.

---

## Key Features & Capabilities

- **Resilient Multi-Stage PDF Parsing Pipeline**:
  - **Stage 1: PyMuPDF4LLM**: High-speed (0.4–0.9s) native Markdown parser generating structural `#`, `##`, `###` headers for text PDFs.
  - **Stage 2: Google Cloud Document AI OCR**: High-accuracy OCR processor for scanned/image-based PDFs.
  - **Stage 3: LLM Semantic Markdown Structuring**: Uses `Gemini 2.5 Flash` to automatically format raw OCR text into structured Markdown headers (`# Subject`, `## Section`, `### Subsection`), backed by a rule-based regex formatter fallback.
  - **Stage 4: Gemini Vision API**: Multimodal extraction safety net with a **30-second hard timeout**.
- **High-Speed GCP Batch Translation**:
  - Automatically translates Devanagari/Marathi text and transliterates proper names (award winners, districts, departments) using **GCP Cloud Translation v3 (`translate_text`)**.
  - Uses smart sub-batching (< 25,000 characters per call) to translate all Marathi passages in a section in **1 single API call (~1.2s)** with Gemini LLM fallback.
- **Hybrid API Routing (Vertex AI + AI Studio)**:
  - **Dense & Sparse Embeddings**: Dedicated AI Studio client via `GEMINI_API_KEY` (`gemini-embedding-001`, 1536-dim).
  - **Generation & LLM Judge**: Vertex AI.
  - **Translation & OCR**: GCP Cloud Translation v3 and GCP Document AI.
- **Hybrid Search Engine**: Combines **Dense Vector Search** (`gemini-embedding-001`) with **BM25 Sparse Keyword Search**, merged via **Reciprocal Rank Fusion (RRF)** in Qdrant.
- **True Agentic RAG Pipeline**: An autonomous function-calling Supervisor agent (`SPEC_MODEL_NAME`) routes queries, extracts keywords, and seamlessly escalates from fast lookups to deep analytical retrieval natively. Includes robust execution loops (`MAX_ITERATIONS = 3`) and graceful fallbacks.
- **Automated Metadata Filtering**: Extracts implicit constraints (such as publication year or section titles) directly from queries using LLM filter parsing.
- **Contextual Conversation Memory**: Rewrites follow-up questions into standalone queries while preventing topic-drift contamination from prior conversation history.
- **Grounded Benchmark & Evaluation Harness**: 30-case dataset (`benchmark.json`) audited directly against corpus contents, combining term-match scoring and a balanced LLM judge evaluator to output detailed performance reports and letter grades (A–D).
- **Streamlit Control Room UI**: Streamlit-based dark theme UI styled with **Inter** and **JetBrains Mono**, featuring:
  - **One-Click Copy**: Native, iframe-free copy buttons dynamically injected via `st.html()`.
  - **Session Persistence**: Automatic saving and restoration of chat history across page refreshes (`temp/chat_session.json`).
  - **Live Agentic Status**: Real-time loading spinners indicating the Supervisor's execution loop state.
- **Automated Unit Test Suite**: 23 unit tests in `tests/` covering parsing, sub-batch translation, API routing, ingestion state, retrieval logic, and evaluation metrics.

---

## Modular Architecture & Directory Tree

```text
rag/
├── main.py                             # CLI entry point (Ingestion, Retrieval, Benchmark)
├── app.py                              # Streamlit web application entry point
├── benchmark.json                      # Standardized 30-case grounded evaluation dataset
├── test_docai.py                       # Document AI pilot testing script
├── test_cloud_translate.py             # GCP Cloud Translation v3 batch pilot script
├── test_models.py                      # Vertex AI model verification script
├── requirements.txt                    # Python dependencies
│
├── temp/                               # Session State Persistence
│   └── chat_session.json               # Server-side serialized chat history
│
├── scratch/                            # Temporary & Persisted State
│   └── ingestion_state.json            # File hashing & incremental ingestion state tracker
│
├── core/                               # Shared Core Infrastructure
│   ├── __init__.py                     # Exports get_logger, get_sparse_model
│   ├── log_config.py                   # Central logging configuration (DEBUG=true/false)
│   ├── embedding.py                    # BM25 sparse embedding model singleton
│   └── utils.py                        # API rate-limit retry, throttle, & AI Studio routing
│
├── ingestion/                          # Ingestion Pipeline
│   ├── __init__.py                     # Exports run_ingestion
│   ├── pipeline.py                     # Ingestion orchestrator & hash skip logic
│   ├── parsers.py                      # PyMuPDF -> DocAI OCR -> Gemini Vision parser sequence
│   ├── metadata.py                     # Metadata extraction (year, category, doc_number)
│   ├── chunking.py                     # Hierarchical child chunking & GCP batch translation
│   └── state.py                        # File hashing & incremental ingestion state manager
│
├── retrieval/                          # Retrieval & Generation Pipeline
│   ├── __init__.py                     # Exports run_retrieval
│   ├── pipeline.py                     # Hybrid search & streaming response orchestrator
│   ├── search.py                       # RRF fusion, LLM reranking, & evidence extraction
│   ├── query.py                        # Contextualization & multi-query expansion
│   └── support.py                      # Context formatting & response extraction helpers
│
├── benchmark/                          # Corpus Evaluation & Benchmark Harness
│   ├── __init__.py                     # Exports run_benchmark, load_benchmark_cases
│   ├── runner.py                       # Benchmark execution & LLM judge runner
│   └── evaluation.py                   # Term-matching scoring metrics
│
├── ui/                                 # Streamlit Frontend & Design System
│   ├── style.css                       # Theme tokens, typography, & custom scrollbars
│   ├── style.py                        # CSS injector utility
│   ├── components.py                   # Top branding strip, logo loader & welcome screen
│   ├── sidebar.py                      # Control room metrics & quick-action triggers
│   └── copy_button.py                  # Clipboard copy button component
│
└── tests/                              # Automated Unit Test Suite (23 Tests)
    ├── test_parsers.py                 # Markdown header formatting & section fallback tests
    ├── test_chunking.py                # GCP sub-batch translation & chunking tests
    ├── test_core_utils.py              # Vertex AI & AI Studio routing tests
    ├── test_ingestion.py               # Document metadata extraction tests
    ├── test_ingestion_state.py         # File hashing & state tracking tests
    ├── test_retrieval.py               # Fast-path & context deduplication tests
    ├── test_evaluation.py              # Term-match scoring tests
    └── test_benchmark.py              # Benchmark execution tests
```

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- Google Gemini API Key (AI Studio)
- Google Cloud Project with Vertex AI and Document AI enabled

### 2. Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/gov-assist-rag.git
   cd rag
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:

   ```env
   GEMINI_API_KEY=your_ai_studio_api_key
   GOOGLE_CLOUD_PROJECT=your_gcp_project_id
   GOOGLE_CLOUD_LOCATION=asia-south1
   TRANSLATE_LOCATION=global
   USE_VERTEX_AI=True
   USE_AISTUDIO_FOR_EMBEDDINGS=True

   # Document AI Configuration
   DOCAI_LOCATION=asia-south1
   DOCAI_PROCESSOR_ID=your_docai_processor_id

   # Models & Verbosity
   EMBED_MODEL_NAME=gemini-embedding-001
   GEN_MODEL_NAME=gemini-2.5-flash
   SPEC_MODEL_NAME=gemini-2.5-flash
   DEBUG=false
   ```

---

## Usage

### Launching the Web Interface

Start the Streamlit GovAssist Control Room:

```bash
streamlit run app.py
```

### Ingestion & CLI Pipeline (`main.py`)

To ingest documents, test interactive CLI retrieval, or run corpus benchmarks, configure the execution flags in `main.py`:

```python
RUN_INGESTION = True    # Re-index PDFs in docs/
RUN_RETRIEVAL = False   # Run interactive CLI chat
RUN_BENCHMARK = True    # Run corpus benchmark evaluation
```

Then execute:

```bash
python main.py
```

### Running Unit Tests

To run the full automated unit test suite (23 tests):

```bash
python -m unittest discover -s tests
```

---

## Disclaimer

GovAssist is designed for administrative decision support. While it prioritizes strict retrieval-based grounding, always verify outputs against official published government circulars and gazette notifications.
