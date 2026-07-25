# GovAssist

GovAssist is an AI-powered Retrieval-Augmented Generation (RAG) system and chat surface designed specifically for querying, comparing, and analyzing government policy documents, circulars, and notifications.

Built for citation-backed grounding and high-precision retrieval, responses are synthesized strictly from indexed local government documents, eliminating hallucinated policy text.

---

## Key Features & Capabilities

- **Hybrid Search Engine**: Combines **Dense Vector Search** (Gemini embeddings) with **BM25 Sparse Keyword Search**, merged via **Reciprocal Rank Fusion (RRF)** in Qdrant.
- **Adaptive Fast/Deep Retrieval**: Automatically routes short/simple queries through a fast lightweight path, and seamlessly escalates to deep retrieval with LLM re-ranking when evidence is sparse.
- **Automated Metadata Filtering**: Extracts implicit constraints (such as publication year or section titles) directly from queries using LLM filter parsing.
- **Contextual Conversation Memory**: Rewrites follow-up questions into standalone queries while preventing topic-drift contamination from prior conversation history.
- **Corpus Benchmarking & Evaluation**: Built-in benchmark suite (`benchmark.py` & `benchmark.json`) combining term-match scoring and a balanced LLM judge evaluator to output detailed performance reports and letter grades (A–D).
- **Modern UI Control Room**: Streamlit-based dark theme UI styled with **Inter** and **JetBrains Mono**, featuring:
  - **One-Click Copy**: Copy button on all assistant responses.
  - **Session Persistence**: Automatic saving and restoration of chat history across page refreshes (`temp/chat_session.json`).
  - **Fast / Deep Mode Switch**: Instant toggle between fast answer mode and deep analytical retrieval.
  - **Resilient Error Handling**: User-friendly error messaging for network connectivity or rate limit events.

---

## Core Components

```
├── app.py                 # Main Streamlit web application & session state runner
├── main.py                # Command-line entry point for Ingestion, Retrieval, & Benchmarks
├── ingestion.py           # Hierarchical PDF document parser, metadata extractor & vector indexer
├── ingestion_state.py     # File hashing & incremental ingestion state tracker
├── retrieval.py           # Runtime hybrid retrieval orchestrator & response streamer
├── retrieval_pipeline.py  # Query contextualizer, filter extractor & generation prompt builder
├── retrieval_support.py   # Context deduplication & response text extractor helpers
├── evaluation.py          # Term coverage evaluation metric functions
├── benchmark.py           # Automated evaluation runner, LLM judge, and report printer
├── utils.py               # Throttling & rate-limit retry wrappers for GenAI APIs
├── benchmark.json         # Standardized 30-case evaluation dataset
└── ui/                    # Design system & frontend components
    ├── style.css          # Theme tokens, Inter typography, animations & scrollbar styles
    ├── style.py           # CSS injector utility
    ├── components.py      # Top branding strip, logo loader & welcome screen
    ├── sidebar.py         # Control room metrics & quick-action triggers
    └── copy_button.py     # Clipboard copy button component
```

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- Google Gemini API Key

### 2. Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/gov-assist-rag.git
   cd gov-assist-rag
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
   Copy `.env.example` to `.env` and enter your API credentials:
   ```bash
   cp .env.example .env
   ```
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   GEN_MODEL_NAME=gemma-4-31b-it
   EMBED_MODEL_NAME=gemini-embedding-001
   ```

---

## Usage

### Launching the Web Interface

Start the Streamlit GovAssist Control Room:

```bash
python -m streamlit run app.py
```

### Ingestion & CLI Pipeline (`main.py`)

To ingest documents, test interactive CLI retrieval, or run corpus benchmarks, configure the toggles in `main.py`:

```python
RUN_INGESTION = True    # Re-index PDFs in docs/
RUN_RETRIEVAL = True    # Run interactive CLI chat
RUN_BENCHMARK = True    # Run corpus benchmark evaluation
```

Then execute:

```bash
python main.py
```

---

## Disclaimer

GovAssist is designed for administrative decision support. While it prioritizes strict retrieval-based grounding, always verify outputs against official published government circulars and gazette notifications.
