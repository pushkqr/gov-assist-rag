import os
import re
import uuid
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from core.log_config import get_logger
from core.utils import embed_content_safe, embed_batch_safe, generate_content_safe
import core.deployment as deployment
import requests

logger = get_logger(__name__)


# Fixed namespace for chunk ids. It must never change: the ids of every chunk already in
# the corpus are derived from it, and a new namespace would silently make re-ingestion
# insert duplicates instead of updating in place.
_CHUNK_NAMESPACE = uuid.UUID("6f2b9c1e-4a7d-5e30-9b8a-1d0c3e5f7a92")


def _chunk_uuid(doc_key: str, parent_index: int, child_index: Optional[int] = None) -> str:
    """Stable id for a parent section or one of its child chunks.

    Derived from the source document and the chunk's position within it rather than
    randomly, so ingesting the same file twice produces the same ids and the second run
    updates the existing objects instead of writing a parallel copy of the document.
    """
    suffix = f":child:{child_index}" if child_index is not None else ""
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_key}:parent:{parent_index}{suffix}"))


class PartialEmbeddingError(RuntimeError):
    """Some passages of a document could not be embedded.

    Callers must not record the document as ingested when this is raised. Embedding
    failures are usually transient (a loaded embedding service timing out), so the
    document is worth retrying on the next run.
    """

try:
    from google.cloud import translate_v3 as translate
except ImportError:
    translate = None


def translate_marathi_batch_indictrans2(chunks: List[str]) -> List[str]:
    """Translate via the self-hosted IndicTrans2 microservice (microservices/translation/),
    the sovereign-mode alternative to GCP Cloud Translation. Same List[str] -> List[str]
    contract as translate_marathi_batch_gcp: an empty string for an item signals the caller
    to fall back to the Gemini per-chunk path for that one chunk, same as a GCP failure would.

    The microservice takes one text at a time (see microservices/translation/main.py), unlike
    GCP's true batch API, so this is one HTTP call per chunk. Slower, and an explicit instance
    of the same tradeoff the rest of the sovereign path makes: self-hosted costs time.
    """
    if not chunks:
        return []

    url = (
        os.environ.get("INGEST_TRANSLATION_SERVICE_URL")
        or os.environ.get("TRANSLATION_SERVICE_URL", "http://localhost:8001/translate")
    )
    timeout = float(os.environ.get("TRANSLATION_TIMEOUT_S", "10"))
    results = []
    failures = 0
    for chunk in chunks:
        try:
            response = requests.post(
                url, json={"text": chunk[:2000], "src_lang": "mar_Deva", "tgt_lang": "eng_Latn"},
                timeout=timeout,
            )
            response.raise_for_status()
            results.append((response.json().get("translated_text") or "").strip())
        except Exception as exc:
            logger.warning(f"IndicTrans2 translation failed for one chunk: {exc}")
            results.append("")
            failures += 1

    logger.info(f"  -> Translated {len(chunks) - failures}/{len(chunks)} Marathi chunks via IndicTrans2.")
    return results


def translate_marathi_batch_gcp(chunks: List[str]) -> List[str]:
    """Translate a list of Marathi text chunks using GCP Cloud Translation v3 (sub-batching under 30,720 codepoints)."""
    if not chunks or translate is None:
        return [""] * len(chunks)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("TRANSLATE_LOCATION", "global")
    parent = f"projects/{project_id}/locations/{location}"

    try:
        client = translate.TranslationServiceClient()
        clean_contents = [c[:2000] for c in chunks]

        # Group clean_contents into sub-batches of max 25,000 chars total
        sub_batches = []
        current_batch = []
        current_len = 0
        for text in clean_contents:
            if current_len + len(text) > 25000 and current_batch:
                sub_batches.append(current_batch)
                current_batch = [text]
                current_len = len(text)
            else:
                current_batch.append(text)
                current_len += len(text)
        if current_batch:
            sub_batches.append(current_batch)

        all_translations = []
        for sb in sub_batches:
            request = translate.TranslateTextRequest(
                contents=sb,
                target_language_code="en",
                parent=parent,
            )
            response = client.translate_text(request=request)
            all_translations.extend([t.translated_text for t in response.translations])

        logger.info(f"  -> Batch translated {len(chunks)} Marathi chunks via GCP Cloud Translation API in {len(sub_batches)} call(s).")
        return all_translations
    except Exception as exc:
        logger.warning(f"GCP Cloud Translation batch call failed: {exc}, falling back to Gemini.")
        return [""] * len(chunks)


def chunk_and_embed_circular(client: genai.Client, markdown_text: str, global_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split document hierarchically and generate dense and sparse embeddings."""
    headers_to_split_on = [
        ("PART-", "Document_Part"),
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    parent_docs = markdown_splitter.split_text(markdown_text)

    # Embedding is deferred to a single batched pass over the whole document (see the loop
    # below the parent_docs loop) rather than called once per chunk inside it. A document with
    # many small sections previously meant many small round trips; collecting every chunk
    # first and embedding them together is what actually cuts the call count, since batching
    # only within one parent section barely helps when most sections hold just a few chunks.
    pending_chunks: List[Dict[str, Any]] = []
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    model_name = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")

    # Identity is derived from the document and the chunk's position within it, never
    # randomly, so re-ingesting the same file reproduces the same ids. That makes a re-run an
    # update of the existing chunks rather than a second copy of them, and gives every chunk a
    # stable handle that survives re-ingestion for citation and debugging.
    # Both splitters are deterministic for the same input, so the indices below are stable.
    doc_key = (
        global_metadata.get("source_filename")
        or global_metadata.get("doc_number")
        or "unknown-document"
    )

    for parent_index, parent_doc in enumerate(parent_docs):
        parent_context = parent_doc.page_content
        parent_metadata = parent_doc.metadata
        parent_id = _chunk_uuid(doc_key, parent_index)

        hierarchy_parts = [parent_metadata[k] for k in ["Document_Part", "Header_1", "Header_2", "Header_3"] if k in parent_metadata]
        hierarchy_context = " > ".join(hierarchy_parts)
        doc_info = f"Document: {global_metadata.get('doc_number', 'Unknown')} ({global_metadata.get('year', 'Unknown')})"
        full_context_prefix = f"{doc_info}\nSection: {hierarchy_context}" if hierarchy_context else doc_info

        section_title = hierarchy_context if hierarchy_context else os.path.basename(global_metadata.get("doc_number", "document"))
        parent_context_with_section = f"Section: {section_title}\n\n{parent_context}"

        child_texts = child_splitter.split_text(parent_context_with_section)
        if not child_texts:
            continue

        processed_child_texts = []
        for ct in child_texts:
            ct_stripped = ct.strip()
            if ct_stripped.startswith("|"):
                idx = parent_context_with_section.find(ct)
                if idx != -1:
                    text_before = parent_context_with_section[:idx]
                    lines_before = text_before.splitlines()
                    
                    table_lines_above = []
                    preamble_str = ""
                    
                    for line in reversed(lines_before):
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                            
                        if line_stripped.startswith("|"):
                            table_lines_above.insert(0, line_stripped)
                        else:
                            text_chars = re.sub(r'[^a-zA-Z0-9\u0900-\u097F]', '', line_stripped)
                            if len(text_chars) > 3 and not line_stripped.startswith("#"):
                                preamble_str = line_stripped
                            break
                            
                    actual_headers = []
                    for line in table_lines_above:
                        content_chars = re.sub(r'[-|: ]', '', line)
                        if len(content_chars) == 0:
                            break
                        actual_headers.append(line)
                        if len(actual_headers) >= 2:
                            break
                            
                    prefix = ""
                    if preamble_str and preamble_str not in ct:
                        prefix += f"Context: {preamble_str}\n"
                        
                    if actual_headers:
                        table_header_str = "\n".join(actual_headers)
                        if not ct_stripped.startswith(actual_headers[0].strip()):
                            prefix += f"{table_header_str}\n"
                            
                    processed_child_texts.append(f"{prefix}{ct}" if prefix else ct)
                else:
                    processed_child_texts.append(ct)
            else:
                processed_child_texts.append(ct)
                
        child_texts = processed_child_texts

        logger.info(f"  -> Chunked into {len(child_texts)} passages. Translating & generating vector embeddings...")

        # Identify Devanagari chunks for batch translation
        marathi_indices = [idx for idx, ct in enumerate(child_texts) if re.search(r"[\u0900-\u097F]", ct)]
        marathi_chunks = [child_texts[idx] for idx in marathi_indices]

        # Batch translate via GCP Cloud Translation or the self-hosted IndicTrans2 microservice,
        # per INGEST_TRANSLATE_PROVIDER (core/deployment.py). Either way, an empty string for a
        # chunk falls through to the per-chunk Gemini fallback a few lines below, unchanged.
        if marathi_chunks:
            if deployment.ingest_translate_provider() == "indictrans2":
                batch_translations = translate_marathi_batch_indictrans2(marathi_chunks)
            else:
                batch_translations = translate_marathi_batch_gcp(marathi_chunks)
        else:
            batch_translations = []
        translation_map = dict(zip(marathi_indices, batch_translations))

        enriched_child_texts = []
        translated_texts = []
        for c_idx, ct in enumerate(child_texts):
            prefix = f"Context: {full_context_prefix}\n\nContent: {ct}"

            if c_idx in marathi_indices:
                tr_text = translation_map.get(c_idx, "")
                if not tr_text:
                    # Fallback to Gemini if GCP batch translation returned empty
                    logger.debug(f"     [Gemini Translation Fallback {c_idx+1}/{len(child_texts)}]")
                    try:
                        tr_prompt = (
                            "Translate the following Indic (Marathi/Hindi) government document text into concise English. "
                            "Transliterate all proper names, award names, and district names cleanly. "
                            f"Output ONLY the English text and names.\n\nIndic Text:\n{ct}"
                        )
                        tr_resp = generate_content_safe(
                            client,
                            model=os.getenv("SPEC_MODEL_NAME", "gemini-3.5-flash"),
                            contents=tr_prompt,
                            config=types.GenerateContentConfig(temperature=0.0, thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)),
                        )
                        tr_text = getattr(tr_resp, "text", "") or ""
                    except Exception as exc:
                        logger.error(f"Failed Gemini fallback translation: {exc}")

                if tr_text.strip():
                    prefix += f"\n\nEnglish Translation: {tr_text.strip()}"
                    translated_texts.append(tr_text.strip())
                else:
                    translated_texts.append(ct)
            else:
                translated_texts.append(ct)

            enriched_child_texts.append(prefix)

        for i, child_text in enumerate(child_texts):
            pending_chunks.append({
                "id": _chunk_uuid(doc_key, parent_index, i),
                "enriched_text": enriched_child_texts[i],
                "child_text": child_text,
                "translated_text": translated_texts[i],
                "parent_id": parent_id,
                "parent_context_with_section": parent_context_with_section,
                "parent_metadata": parent_metadata,
                "section_title": section_title,
            })

    if not pending_chunks:
        return []

    logger.info(f"  -> Embedding {len(pending_chunks)} passages ({model_name})...")
    vectors = embed_batch_safe(client, [p["enriched_text"] for p in pending_chunks], model_name=model_name)

    database_payload = []
    skipped = 0
    for chunk, dense_vector in zip(pending_chunks, vectors):
        if dense_vector is None:
            skipped += 1
            continue

        payload_metadata = {
            **chunk["parent_metadata"],
            **global_metadata,
            "parent_id": chunk["parent_id"],
            "parent_context": chunk["parent_context_with_section"],
            "child_text": chunk["child_text"],
            "translated_text": chunk["translated_text"],
            "section_title": chunk["section_title"],
        }

        database_payload.append(
            {
                "id": chunk["id"],
                "vector": {"dense": dense_vector},
                "metadata": payload_metadata,
                "enriched_text_used_for_embedding": chunk["enriched_text"],
            }
        )

    if skipped:
        # All or nothing, deliberately. Returning the partial payload would insert most of
        # the document and let the caller record it as ingested, so the missing passages
        # become invisible: the state file stops any later run from revisiting the file, and
        # nothing downstream can tell an incomplete document from a complete one. Failing the
        # whole document instead leaves it unrecorded, so the next run picks it up again.
        # Nothing has been written to the store at this point, so there is no partial state
        # to unwind and a retry cannot duplicate what a previous attempt inserted.
        raise PartialEmbeddingError(
            f"{skipped} of {len(pending_chunks)} passages could not be embedded"
        )

    return database_payload
