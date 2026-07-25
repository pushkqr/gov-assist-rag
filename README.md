# GovAssist

GovAssist is an AI-powered Retrieval-Augmented Generation (RAG) system and chat surface designed specifically for querying, comparing, and analyzing government policy documents, circulars, and notifications.

Built for citation-backed grounding and high-precision retrieval, responses are synthesized strictly from indexed local government documents, eliminating hallucinated policy text.

---

## Key Features & Capabilities

- **Resilient Multi-Tier PDF Processing**:
  - **Tier 1: Google Document AI Layout Processor**: Extracts structured layout blocks (`document_layout.blocks`) with explicit page numbers and section headers (`### Section`).
  - **Tier 2: Gemini Vision API**: High-precision multimodal extraction with a **45-second hard timeout**.
  - **Tier 3: Local PyMuPDF Parser**: Offline fallback ensuring ingestion never stalls.
- **Hybrid Search Engine**: Combines **Dense Vector Search** (`gemini-embedding-001`) with **BM25 Sparse Keyword Search**, merged via **Reciprocal Rank Fusion (RRF)** in Qdrant.
- **Adaptive Fast/Deep Retrieval**: Automatically routes simple queries through a fast lightweight path, and seamlessly escalates to deep retrieval with LLM re-ranking when evidence is sparse.
- **Automated Metadata Filtering**: Extracts implicit constraints (such as publication year or section titles) directly from queries using LLM filter parsing.
- **Bilingual Devanagari Support**: Automatically translates and transliterates Marathi Devanagari text into English prefixes for cross-lingual keyword matching.
- **Contextual Conversation Memory**: Rewrites follow-up questions into standalone queries while preventing topic-drift contamination from prior conversation history.
- **Corpus Benchmarking & Evaluation**: Built-in benchmark suite (`benchmark/` & `benchmark.json`) combining term-match scoring and a balanced LLM judge evaluator to output detailed performance reports and letter grades (A–D).
- **Configurable Standard Logging**: Powered by a central logging module (`core/log_config.py`) controllable via `DEBUG=true/false` in `.env`.
- **Modern UI Control Room**: Streamlit-based dark theme UI styled with **Inter** and **JetBrains Mono**, featuring:
  - **One-Click Copy**: Copy button on all assistant responses.
  - **Session Persistence**: Automatic saving and restoration of chat history across page refreshes (`temp/chat_session.json`).
  - **Fast / Deep Mode Switch**: Instant toggle between fast answer mode and deep analytical retrieval.
  - **Resilient Error Handling**: User-friendly error messaging for network connectivity or rate limit events.

---

## Modular Architecture & Directory Tree

```text
rag/
├── main.py                             # CLI entry point (Ingestion, Retrieval, Benchmark)
├── app.py                              # Streamlit web application entry point
├── benchmark.json                      # Standardized 30-case evaluation dataset
├── test_docai.py                       # Document AI pilot testing script
├── requirements.txt                    # Python dependencies
│
├── core/                               # Shared Core Infrastructure
│   ├── __init__.py                     # Exports get_logger, get_sparse_model
│   ├── log_config.py                   # Central logging configuration (DEBUG=true/false)
│   ├── embedding.py                    # BM25 sparse embedding model singleton
│   └── utils.py                        # API rate-limit retry & throttle wrappers
│
├── ingestion/                          # Ingestion Pipeline
│   ├── __init__.py                     # Exports run_ingestion
│   ├── pipeline.py                     # Ingestion orchestrator & hash skip logic
│   ├── parsers.py                      # DocAI Layout, Gemini Vision (45s), PyMuPDF
│   ├── metadata.py                     # Metadata extraction (year, category, doc_number)
│   ├── chunking.py                     # Hierarchical child chunking & Marathi translation
│   └── state.py                        # File hashing & incremental ingestion state tracker
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
└── tests/                              # Automated Unit Test Suite
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_evaluation.py
    ├── test_benchmark.py
    └── test_ingestion_state.py
```

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- Google Gemini API Key (AI Studio)
- Google Cloud Document AI Processor (optional for DocAI Layout parsing)

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
   Copy `.env.example` to `.env` and set your credentials:

   ```env
   GEMINI_API_KEY=your_ai_studio_api_key
   GOOGLE_GENAI_USE_ENTERPRISE=False

   # Document AI Configuration
   DOCAI_PROJECT_ID=your_gcp_project_id
   DOCAI_LOCATION=asia-southeast1
   DOCAI_PROCESSOR_ID=your_processor_id

   # Models & Verbosity
   EMBED_MODEL_NAME=gemini-embedding-001
   GEN_MODEL_NAME=gemma-4-31b-it
   SPEC_MODEL_NAME=gemini-3.5-flash
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
RUN_RETRIEVAL = True    # Run interactive CLI chat
RUN_BENCHMARK = True    # Run corpus benchmark evaluation
```

Then execute:

```bash
python main.py
```

### Running Unit Tests

To run the automated test suite:

```bash
python -m unittest discover -s tests
```

---

## Disclaimer

GovAssist is designed for administrative decision support. While it prioritizes strict retrieval-based grounding, always verify outputs against official published government circulars and gazette notifications.
