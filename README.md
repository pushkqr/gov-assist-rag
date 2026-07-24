# GovAssist

GovAssist is an advanced Retrieval-Augmented Generation (RAG) engine and chat interface designed specifically for querying, comparing, and analyzing government policy documents, rules, and notifications. 

Built with a focus on citation-backed grounding and high-precision retrieval, the system ensures responses are generated strictly from indexed local documents, preventing hallucinated policy text.

## Architecture Overview

GovAssist operates on a dual-model architecture to optimize speed, cost, and accuracy:
- **Embedding Model**: Fast, lightweight model for generating dense vectors from document chunks.
- **Generative Model**: High-parameter instruction-tuned model for synthesizing answers and formatting structured data.
- **Cross-Encoder**: A local re-ranking model that mathematically scores retrieved chunks against the user query for maximum contextual relevance before generation.

## Core Components

### 1. Ingestion Engine (`ingestion.py`)
Parses complex government PDFs (supporting both English and regional languages) using a lightweight local parser (`pymupdf4llm`). The engine dynamically extracts metadata (such as document year and department) using regex rules, chunks the text semantically, generates embeddings, and indexes them into a local Qdrant vector database.

### 2. Retrieval & Contextualization (`retrieval.py`)
Handles the runtime query processing. 
- **Query Contextualization**: Analyzes conversation history to rewrite follow-up questions into standalone queries, while actively detecting abrupt topic switches to prevent irrelevant filters from carrying over.
- **Hybrid Search**: Combines semantic vector search with keyword-based metadata filtering.
- **Cross-Encoder Re-ranking**: Re-scores the top retrieved chunks for maximum precision.

### 3. Frontend Control Room (`app.py` & `ui/`)
A modularized, custom-styled Streamlit application providing a professional chat surface.
- **Real-time Streaming**: Consumes the backend generator to provide typewriter-style responses.
- **Modular UI**: The massive aesthetic styling and component logic are cleanly separated into `ui/style.css`, `ui/components.py`, and `ui/sidebar.py`.
- **Quick Actions**: Sidebar shortcuts for common policy workflows like summarizing notifications or generating eligibility checklists.

## Installation and Setup

### Prerequisites
- Python 3.10+
- An API key for the generative and embedding models.

### Environment Setup

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/gov-assist-rag.git
cd gov-assist-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory and add your keys and model preferences:
```env
API_KEY=your_api_key_here
GEN_MODEL_NAME=gemma-4-31b-it
EMBED_MODEL_NAME=gemini-embedding-001
```

### Running the Application

1. **Ingest Documents**:
Place your target government PDFs in the `data/` directory, then run the ingestion script to build the local vector database:
```bash
python ingestion.py
```

2. **Launch the Interface**:
Start the Streamlit application to open the GovAssist Control Room:
```bash
python -m streamlit run app.py
```

## Disclaimer
GovAssist is a conceptual implementation. It may generate mistakes or misinterpret complex legal language. Always verify outputs with official published government circulars.
