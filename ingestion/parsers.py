import os
import re
import time
from typing import Optional

import pymupdf4llm
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai
from google import genai
from google.genai import types

from core.log_config import get_logger
from core.utils import generate_content_safe

logger = get_logger(__name__)


def format_plain_text_with_llm(client: genai.Client, raw_text: str) -> str:
    """Use Gemini Flash to structure raw OCR plain text into semantic Markdown headers."""
    if not raw_text or len(raw_text.strip()) < 50:
        return raw_text

    # If text already has markdown headers, return as-is
    if re.search(r"^#{1,3}\s+", raw_text, re.MULTILINE):
        return raw_text

    try:
        prompt = f"""You are an expert document structure parser.
Organize the following raw document OCR text into clean Markdown format.

Requirements:
1. Identify all document titles, circular/notification numbers, subjects, main sections, and sub-sections.
2. Mark main titles and subjects with '#', major sections with '##', and sub-sections/points with '###'.
3. Preserve all exact wording, numbers, table values, dates, and text content. Do NOT summarize or omit anything.
4. Output ONLY the formatted Markdown.

Raw Text:
{raw_text}
"""
        response = generate_content_safe(
            client,
            model=os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        formatted = getattr(response, "text", "") or ""
        if formatted and len(formatted.strip()) >= 50:
            logger.info("  -> LLM semantic structuring successfully generated Markdown headers.")
            return formatted.strip()
    except Exception as exc:
        logger.warning(f"LLM semantic structuring fallback to rule-based parser due to: {exc}")

    return format_plain_text_to_markdown(raw_text)


def format_plain_text_to_markdown(text: str) -> str:
    """Rule-based fallback: format plain OCR text with markdown headers (#, ##, ###)."""
    if not text:
        return ""

    if re.search(r"^#{1,3}\s+", text, re.MULTILINE):
        return text

    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue

        if re.match(r"^(Circular|Notification|F\.?\s*No\.?|G\.S\.R|ORDER|MEMORANDUM)\s", stripped, re.IGNORECASE):
            formatted_lines.append(f"# {stripped}")
        elif re.match(r"^Subject\s*:", stripped, re.IGNORECASE):
            formatted_lines.append(f"# {stripped}")
        elif re.match(r"^\d+\.\s+[A-Z]", stripped) and len(stripped) < 150:
            formatted_lines.append(f"## {stripped}")
        elif re.match(r"^\b(CHAPTER|PART|ANNEXURE|SCHEDULE)\b", stripped, re.IGNORECASE) and len(stripped) < 100:
            formatted_lines.append(f"## {stripped}")
        elif re.match(r"^\d+\.\d+(\.\d+)?\s+", stripped) and len(stripped) < 150:
            formatted_lines.append(f"### {stripped}")
        else:
            formatted_lines.append(line)

    result = "\n".join(formatted_lines)

    if not re.search(r"^#{1,3}\s+", result, re.MULTILINE) and len(result) > 1500:
        paragraphs = result.split("\n\n")
        section_chunks = []
        current_chunk = []
        current_len = 0
        section_idx = 1

        for p in paragraphs:
            current_chunk.append(p)
            current_len += len(p)
            if current_len >= 1200:
                section_chunks.append(f"## Section {section_idx}\n\n" + "\n\n".join(current_chunk))
                section_idx += 1
                current_chunk = []
                current_len = 0
        if current_chunk:
            section_chunks.append(f"## Section {section_idx}\n\n" + "\n\n".join(current_chunk))
        result = "\n\n".join(section_chunks)

    return result


def parse_pdf_with_document_ai(client: genai.Client, target_file: str) -> Optional[str]:
    """Parse PDF document using Google Cloud Document AI Document Processor (OCR)."""
    project_id = os.getenv("DOCAI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("DOCAI_LOCATION", "asia-south1")
    processor_id = os.getenv("DOCAI_PROCESSOR_ID")

    if not project_id or not processor_id:
        return None

    try:
        api_endpoint = f"{location}-documentai.googleapis.com"
        opts = ClientOptions(api_endpoint=api_endpoint)
        docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)
        processor_path = docai_client.processor_path(project_id, location, processor_id)

        with open(target_file, "rb") as f:
            file_content = f.read()

        raw_document = documentai.RawDocument(content=file_content, mime_type="application/pdf")
        request = documentai.ProcessRequest(
            name=processor_path,
            raw_document=raw_document,
        )

        t0 = time.time()
        response = docai_client.process_document(request=request, timeout=60.0)
        elapsed = time.time() - t0
        doc = getattr(response, "document", None)

        if doc and doc.text and len(doc.text.strip()) > 50:
            extracted_text = doc.text.strip()
            formatted_md = format_plain_text_with_llm(client, extracted_text)
            logger.info(f"  -> Document AI OCR extracted {len(formatted_md)} chars in {elapsed:.1f}s.")
            return formatted_md
    except Exception as exc:
        logger.error(f"[Document AI OCR] Processing failed for {os.path.basename(target_file)}: {exc}")
    return None


def parse_pdf_with_gemini_vision(client: genai.Client, target_file: str, timeout: int = 30) -> Optional[str]:
    """Parse PDF using Gemini Vision API with configurable timeout."""
    filename = os.path.basename(target_file)
    logger.info(f"Parsing {filename} with Gemini Vision API for markdown extraction ({timeout}s timeout)...")
    try:
        uploaded_file = client.files.upload(file=target_file)
        start_time = time.time()
        timed_out = False
        try:
            while uploaded_file.state == "PROCESSING":
                if time.time() - start_time > timeout:
                    logger.warning(f"[Timeout] Gemini Vision API extraction timed out (>{timeout}s) for {filename}.")
                    timed_out = True
                    break
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            if not timed_out and uploaded_file.state != "FAILED":
                prompt = (
                    "Extract the entire text from this document into clean, structural Markdown. "
                    "Preserve all headers (use #, ##, ###), tables, and lists exactly as they appear "
                    "in the original layout. Do not summarize or skip anything. Output ONLY the markdown text."
                )
                response = generate_content_safe(
                    client,
                    model=os.getenv("SPEC_MODEL_NAME", "gemini-3.5-flash"),
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(temperature=0.0, thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)),
                )
                result = getattr(response, "text", "") or ""
                if result and len(result.strip()) >= 100:
                    return format_plain_text_with_llm(client, result.strip())
        finally:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Gemini Vision API parsing failed for {filename}: {e}")
    return None


def parse_pdf_with_pymupdf(target_file: str) -> Optional[str]:
    """Parse PDF using local PyMuPDF library."""
    filename = os.path.basename(target_file)
    try:
        result = pymupdf4llm.to_markdown(target_file)
        if result and len(result.strip()) >= 100:
            return result
    except Exception as e:
        logger.error(f"PyMuPDF failed for {filename}: {e}")
    return None


def parse_pdf(client: genai.Client, target_file: str) -> Optional[str]:
    """Orchestrate PDF parsing: PyMuPDF -> DocAI OCR -> Gemini Vision -> PyMuPDF fallback."""
    filename = os.path.basename(target_file)

    # Step 1: Primary parser is PyMuPDF
    logger.info(f"Parsing {filename} with PyMuPDF...")
    target_md = parse_pdf_with_pymupdf(target_file)

    # Step 2: Fallback to Document AI OCR if PyMuPDF returned empty or < 100 chars
    if not target_md or len(target_md.strip()) < 100:
        logger.warning(f"PyMuPDF yielded insufficient content for {filename}. Falling back to Google Document AI OCR...")
        target_md = parse_pdf_with_document_ai(client, target_file)

    # Step 3: Fallback to Gemini Vision if DocAI OCR also returned empty or < 100 chars
    if not target_md or len(target_md.strip()) < 100:
        logger.warning(f"Document AI OCR yielded insufficient content for {filename}. Falling back to Gemini Vision API...")
        target_md = parse_pdf_with_gemini_vision(client, target_file)

    # Step 4: Final fallback to PyMuPDF if anything remains
    if not target_md or len(target_md.strip()) < 100:
        target_md = parse_pdf_with_pymupdf(target_file)

    return target_md
