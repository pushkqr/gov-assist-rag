import glob
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import models

from ingestion_state import compute_file_hash, save_ingestion_state, should_skip_file
from utils import embed_content_safe, generate_content_safe

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

_SPARSE_MODEL: Optional[SparseTextEmbedding] = None


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy singleton for BM25 sparse embedding model."""
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        _SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _SPARSE_MODEL


def extract_document_metadata(markdown_text: str, source_path: str, fallback_year: int = 2025) -> Dict[str, Any]:
    """Extract structured document metadata fields from markdown text."""
    normalized = (markdown_text or "").strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    title = None
    for line in lines:
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break

    if not title:
        title = os.path.splitext(os.path.basename(source_path))[0].replace("_", " ").replace("-", " ").strip()

    doc_number = None
    patterns = [
        r"document\s*(?:no\.?|number)\s*[:#-]?\s*([A-Za-z0-9\-/\.]+)",
        r"\bno\.\s*([A-Za-z0-9\-/\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            doc_number = match.group(1).strip()
            break

    if not doc_number:
        doc_number = os.path.splitext(os.path.basename(source_path))[0]

    year = fallback_year
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        year = int(year_match.group(0))

    issuing_authority = "Government"
    authority_patterns = [r"issued\s+by\s*[:\-]\s*(.+)", r"authority\s*[:\-]\s*(.+)"]
    for pattern in authority_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            issuing_authority = re.sub(r"\s+", " ", match.group(1)).strip()
            break

    document_category = "Document"
    for label, category in [
        ("notification", "Notification"),
        ("circular", "Circular"),
        ("order", "Order"),
        ("rule", "Rule"),
        ("guideline", "Guideline"),
        ("directive", "Directive"),
    ]:
        if re.search(rf"\b{label}\b", (title or ""), flags=re.IGNORECASE):
            document_category = category
            break

    return {
        "document_title": title,
        "year": year,
        "doc_number": doc_number,
        "issuing_authority": issuing_authority,
        "document_category": document_category,
    }


def chunk_and_embed_circular(client: genai.Client, markdown_text: str, global_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split document hierarchically and generate dense and sparse embeddings."""
    sparse_model = get_sparse_model()

    headers_to_split_on = [
        ("PART-", "Document_Part"),
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    parent_docs = markdown_splitter.split_text(markdown_text)

    database_payload = []
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=1536)
    model_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")

    for parent_doc in parent_docs:
        parent_context = parent_doc.page_content
        parent_metadata = parent_doc.metadata
        parent_id = str(uuid.uuid4())

        hierarchy_parts = [parent_metadata[k] for k in ["Document_Part", "Header_1", "Header_2", "Header_3"] if k in parent_metadata]
        hierarchy_context = " > ".join(hierarchy_parts)
        doc_info = f"Document: {global_metadata.get('doc_number', 'Unknown')} ({global_metadata.get('year', 'Unknown')})"
        full_context_prefix = f"{doc_info}\nSection: {hierarchy_context}" if hierarchy_context else doc_info

        section_title = hierarchy_context if hierarchy_context else os.path.basename(global_metadata.get("doc_number", "document"))
        parent_context_with_section = f"Section: {section_title}\n\n{parent_context}"

        child_texts = child_splitter.split_text(parent_context_with_section)
        if not child_texts:
            continue

        enriched_child_texts = [f"Context: {full_context_prefix}\n\nContent: {ct}" for ct in child_texts]
        sparse_embeddings = list(sparse_model.embed(child_texts))

        for i, child_text in enumerate(child_texts):
            dense_response = embed_content_safe(client, model=model_name, contents=enriched_child_texts[i], config=config)

            if not hasattr(dense_response, "embeddings") or not dense_response.embeddings:
                continue

            dense_vector = dense_response.embeddings[0].values
            sparse_vec = sparse_embeddings[i]
            vector_dict = {
                "dense": dense_vector,
                "bm25": models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
            }

            payload_metadata = {
                **parent_metadata,
                **global_metadata,
                "parent_id": parent_id,
                "parent_context": parent_context_with_section,
                "child_text": child_text,
                "section_title": section_title,
            }

            database_payload.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": vector_dict,
                    "metadata": payload_metadata,
                    "enriched_text_used_for_embedding": enriched_child_texts[i],
                }
            )

    return database_payload


def run_ingestion(
    client: genai.Client,
    docs_dir: str = "docs",
    target_files: Optional[List[str]] = None,
    use_local_parser: bool = False,
) -> List[Dict[str, Any]]:
    """Process PDF documents in target directory and return vector records."""
    if target_files:
        pdf_files = [os.path.join(docs_dir, f) for f in target_files]
    else:
        pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))

    if not pdf_files:
        return []

    all_processed_records = []
    state_path = os.getenv("INGESTION_STATE_PATH", os.path.join(os.getcwd(), "ingestion_state.json"))

    for target_file in pdf_files:
        if not os.path.exists(target_file):
            continue

        file_hash = compute_file_hash(target_file)
        if should_skip_file(target_file, file_hash, state_path):
            continue

        target_md = None

        if use_local_parser:
            if pymupdf4llm is None:
                print("Error: PyMuPDF4LLM is not installed.")
                continue
            try:
                target_md = pymupdf4llm.to_markdown(target_file)
            except Exception as e:
                print(f"Error extracting markdown via PyMuPDF for {target_file}: {e}")
                continue
        else:
            uploaded_file = client.files.upload(file=target_file)
            try:
                while uploaded_file.state == "PROCESSING":
                    time.sleep(3)
                    uploaded_file = client.files.get(name=uploaded_file.name)

                if uploaded_file.state == "FAILED":
                    continue

                prompt = (
                    "Extract the entire text from this document into clean, structural Markdown. "
                    "Preserve all headers (use #, ##, ###), tables, and lists exactly as they appear "
                    "in the original layout. Do not summarize or skip anything. Output ONLY the markdown text."
                )

                response = generate_content_safe(
                    client,
                    model=os.getenv("SPEC_MODEL_NAME", "gemini-3.1-flash-lite"),
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                target_md = response.text
                if not target_md:
                    continue
            except Exception as e:
                print(f"Error extracting markdown via Gemini for {target_file}: {e}")
                continue
            finally:
                client.files.delete(name=uploaded_file.name)

        if not target_md:
            continue

        doc_year = 2025
        year_match = re.search(r"\b(19|20)\d{2}\b", target_md[:2000])
        if year_match:
            doc_year = int(year_match.group(0))

        extracted_metadata = extract_document_metadata(target_md, target_file, fallback_year=doc_year)
        global_metadata = {
            "doc_type": "PDF Document",
            "issuing_authority": extracted_metadata.get("issuing_authority", "Government"),
            "year": extracted_metadata.get("year", doc_year),
            "doc_number": extracted_metadata.get("doc_number", os.path.basename(target_file)),
            "document_title": extracted_metadata.get("document_title", os.path.splitext(os.path.basename(target_file))[0]),
            "document_category": extracted_metadata.get("document_category", "Document"),
        }

        processed_records = chunk_and_embed_circular(client, target_md, global_metadata)
        all_processed_records.extend(processed_records)
        save_ingestion_state(target_file, file_hash, state_path, global_metadata)

    return all_processed_records
