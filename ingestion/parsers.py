import os
import time
from typing import Optional

from google import genai
from google.genai import types

from core.log_config import get_logger
from core.utils import generate_content_safe

logger = get_logger(__name__)

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

try:
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai_v1beta3 as documentai
except ImportError:
    documentai = None
    ClientOptions = None


def parse_layout_blocks_to_markdown(document_layout) -> str:
    """Convert Document AI Layout Parser blocks into structured, page-tagged Markdown."""
    blocks = getattr(document_layout, "blocks", [])
    if not blocks:
        return ""

    markdown_chunks = []
    current_header = "General"

    for block in blocks:
        page_span = getattr(block, "page_span", None)
        page_num = getattr(page_span, "page_start", 1) if page_span else 1

        text_block = getattr(block, "text_block", None)
        if not text_block or not text_block.text:
            continue

        text = text_block.text.strip()
        block_type = getattr(text_block, "type_", None) or getattr(text_block, "type", "paragraph")
        block_type_str = str(block_type).lower()

        if "header" in block_type_str or text.startswith("#"):
            current_header = text.lstrip("#").strip()
            markdown_chunks.append(f"\n### {current_header}\n")
        elif "table" in block_type_str:
            markdown_chunks.append(f"\n<!-- Page {page_num} | Table Block -->\n{text}\n")
        else:
            markdown_chunks.append(f"\n<!-- Page {page_num} | Section: {current_header} -->\n{text}\n")

    return "".join(markdown_chunks)


def parse_pdf_with_document_ai(target_file: str) -> Optional[str]:
    """Parse PDF document using Google Cloud Document AI Layout Processor."""
    if documentai is None or ClientOptions is None:
        return None

    project_id = os.getenv("DOCAI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("DOCAI_LOCATION", "asia-southeast1")
    processor_id = os.getenv("DOCAI_PROCESSOR_ID")

    if not project_id or not processor_id:
        return None

    try:
        api_endpoint = f"{location}-documentai.googleapis.com"
        opts = ClientOptions(api_endpoint=api_endpoint)
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        processor_path = client.processor_path(project_id, location, processor_id)

        with open(target_file, "rb") as f:
            file_content = f.read()

        raw_document = documentai.RawDocument(content=file_content, mime_type="application/pdf")
        process_options = documentai.ProcessOptions(
            layout_config=documentai.ProcessOptions.LayoutConfig(
                chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                    chunk_size=500,
                    include_ancestor_headings=True,
                )
            )
        )

        request = documentai.ProcessRequest(
            name=processor_path,
            raw_document=raw_document,
            process_options=process_options,
        )

        t0 = time.time()
        response = client.process_document(request=request, timeout=60.0)
        elapsed = time.time() - t0
        doc = getattr(response, "document", None)

        if doc:
            document_layout = getattr(doc, "document_layout", None)
            formatted_md = parse_layout_blocks_to_markdown(document_layout) if document_layout else ""
            if not formatted_md and doc.text and len(doc.text.strip()) > 100:
                formatted_md = doc.text.strip()

            if formatted_md:
                logger.info(f"  -> Document AI extracted {len(formatted_md)} chars ({len(getattr(document_layout, 'blocks', []))} layout blocks) in {elapsed:.1f}s.")
                return formatted_md
    except Exception as exc:
        logger.error(f"[Document AI] Processing failed for {os.path.basename(target_file)}: {exc}")
    return None


def parse_pdf_with_gemini_vision(client: genai.Client, target_file: str, timeout: int = 45) -> Optional[str]:
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
                    logger.warning(f"[Timeout] Gemini Vision API extraction timed out (>{timeout}s) for {filename}. Falling back to PyMuPDF.")
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
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                result = getattr(response, "text", "") or ""
                if result and len(result.strip()) >= 100:
                    return result
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
    if pymupdf4llm is None:
        return None
    filename = os.path.basename(target_file)
    logger.warning(f"Fallback: Parsing {filename} with PyMuPDF local parser...")
    try:
        result = pymupdf4llm.to_markdown(target_file)
        if result and len(result.strip()) >= 100:
            return result
    except Exception as e:
        logger.error(f"PyMuPDF fallback failed for {filename}: {e}")
    return None


def parse_pdf(client: genai.Client, target_file: str, use_local_parser: bool = False) -> Optional[str]:
    """Orchestrate PDF parsing: DocAI → PyMuPDF (if local) → Gemini Vision (45s timeout) → PyMuPDF fallback."""
    filename = os.path.basename(target_file)
    target_md = None

    if not use_local_parser:
        logger.info(f"Parsing {filename} with Google Document AI Layout Processor...")
        target_md = parse_pdf_with_document_ai(target_file)

    if not target_md and use_local_parser:
        logger.info(f"Parsing {filename} with PyMuPDF local parser...")
        try:
            if pymupdf4llm is not None:
                target_md = pymupdf4llm.to_markdown(target_file)
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {filename}: {e}, falling back to Gemini Vision API.")
            target_md = None

    if not target_md or len(target_md.strip()) < 100:
        target_md = parse_pdf_with_gemini_vision(client, target_file)

    if not target_md or len(target_md.strip()) < 100:
        target_md = parse_pdf_with_pymupdf(target_file)

    return target_md
