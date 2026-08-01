import os
import re
import uuid
from typing import Any, Dict, List

from google import genai
from google.genai import types
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from core.log_config import get_logger
from core.utils import embed_content_safe, generate_content_safe

logger = get_logger(__name__)

try:
    from google.cloud import translate_v3 as translate
except ImportError:
    translate = None


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

    database_payload = []
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=1536)
    model_name = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")

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

        # Batch translate via GCP Cloud Translation v3
        gcp_translations = translate_marathi_batch_gcp(marathi_chunks) if marathi_chunks else []
        gcp_trans_map = dict(zip(marathi_indices, gcp_translations))

        enriched_child_texts = []
        translated_texts = []
        for c_idx, ct in enumerate(child_texts):
            prefix = f"Context: {full_context_prefix}\n\nContent: {ct}"

            if c_idx in marathi_indices:
                tr_text = gcp_trans_map.get(c_idx, "")
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
                            model=os.getenv("SPEC_MODEL_NAME", "gemini-2.5-flash"),
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
            logger.debug(f"     [Embedding {i+1}/{len(child_texts)}] Generating dense vector ({model_name})...")
            dense_response = embed_content_safe(client, model=model_name, contents=enriched_child_texts[i], config=config)

            if not hasattr(dense_response, "embeddings") or not dense_response.embeddings:
                continue

            dense_vector = dense_response.embeddings[0].values
            vector_dict = {
                "dense": dense_vector,
            }

            payload_metadata = {
                **parent_metadata,
                **global_metadata,
                "parent_id": parent_id,
                "parent_context": parent_context_with_section,
                "child_text": child_text,
                "translated_text": translated_texts[i],
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
