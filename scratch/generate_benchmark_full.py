"""
Document-level benchmark generator for the full corpus.
Generates 2-3 diverse questions per document using Gemini, fed the full doc text.

Usage:
    python scratch/generate_benchmark_full.py [--limit N] [--out benchmark/benchmark_100.json]

Strategy:
  - Feed FULL document text (not chunks) to Gemini
  - Ask for officer-perspective questions with explicit diversity requirements
  - Incrementally saves so it can be resumed if interrupted
  - Deduplicates across documents
"""
import os, sys, json, glob, time, argparse
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

import pymupdf4llm
from google import genai
from google.genai import types
from core.utils import get_genai_client

GENERATION_PROMPT = """You are helping build a benchmark test set for a Government Document RAG system used by Maharashtra government officers.

Given the following government document text, generate exactly {n_questions} test questions. The questions must reflect how a REAL officer would naturally ask about this document - conversational, practical, not academic.

DIVERSITY REQUIREMENTS - include these types across your questions:
1. One SIMPLE factual question in English (what, who, how many - answerable from a single sentence)
2. One question in MARATHI (natural officer language, not a direct translation)
3. One COMPLEX question requiring synthesis of multiple facts from the document
{extra_requirements}

RULES:
- Questions must be answerable from THIS document only
- Do NOT ask about things not in the document  
- Use colloquial phrasing an officer would use, not legalese
- For Marathi questions, write in natural Devanagari script
- Each question must have a clear, verifiable answer from the document text

Return ONLY a valid JSON array with this structure (no markdown, no explanation):
[
  {{
    "query": "the question text",
    "expected_answer": "the ideal answer based on the document",
    "expected_terms": ["key", "terms", "that", "must", "appear"],
    "category": "simple_english|marathi_query|complex_english|hindi_query|gr_number_lookup",
    "source_doc": "{filename}"
  }}
]

DOCUMENT TEXT:
---
{doc_text}
---"""

NOT_FOUND_QUESTIONS = [
    {
        "query": "What is the pension amount for retired government professors in Maharashtra?",
        "expected_answer": "The documents available in the system do not contain information about pension amounts for retired government professors. Please check the Finance Department rules.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "शासकीय महाविद्यालयातील विद्यार्थ्यांसाठी शिष्यवृत्तीची रक्कम किती आहे?",
        "expected_answer": "उपलब्ध दस्तऐवजांमध्ये शासकीय महाविद्यालयातील शिष्यवृत्तीबद्दल माहिती नाही.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "Government engineering college mein admission ke liye minimum percentage kya chahiye?",
        "expected_answer": "The retrieved documents do not contain information about minimum percentage requirements for engineering college admissions.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "What is the hostel fee for students at Government Engineering College Kolhapur?",
        "expected_answer": "The documents in the system do not contain information about hostel fees. The GR on the new Kolhapur Government Engineering College only covers course approval and staffing plans.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "What are the medical leave rules for teaching staff in Maharashtra government colleges?",
        "expected_answer": "The retrieved documents do not contain information about medical leave rules for teaching staff.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
]


def read_document(filepath):
    """Read document text - PDF via pymupdf4llm, txt directly."""
    if filepath.endswith(".pdf") or filepath.endswith(".PDF"):
        try:
            return pymupdf4llm.to_markdown(filepath)
        except Exception as e:
            print(f"  [WARN] PDF read failed: {e}")
            return None
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def generate_questions_for_doc(gemini_client, doc_text, filename, n_questions=3, trim_chars=9000):
    """Call Gemini to generate questions for a single document."""
    # Only trim — caller already skipped docs that are too big
    if len(doc_text) > trim_chars:
        doc_text = doc_text[:trim_chars] + "\n\n[... document continues, truncated ...]"

    extra = ""
    if n_questions >= 4:
        extra = "4. One question as a GR number lookup (asking about this specific GR number)"

    prompt = GENERATION_PROMPT.format(
        n_questions=n_questions,
        extra_requirements=extra,
        filename=filename,
        doc_text=doc_text
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
        )
        text = response.text.strip()
        # Strip markdown if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        questions = json.loads(text)
        return questions
    except Exception as e:
        print(f"  [ERROR] Generation failed for {filename}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    parser.add_argument("--out", default="benchmark/benchmark_100.json", help="Output file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument(
        "--max-doc-chars", type=int, default=15000,
        help="Skip docs larger than this many characters (default 15000 ~3750 tokens). "
             "Use 0 to disable skipping."
    )
    args = parser.parse_args()

    gemini = get_genai_client()

    # Collect all documents
    pdf_files = sorted(glob.glob("docs/*.pdf") + glob.glob("docs/*.PDF"))
    orgpedia_en = sorted(glob.glob("docs/orgpedia_mahGRs/*.en.txt"))

    all_docs = [(f, "pdf") for f in pdf_files] + [(f, "orgpedia") for f in orgpedia_en]

    if args.limit:
        all_docs = all_docs[:args.limit]

    print(f"Total documents to process: {len(all_docs)} (PDFs: {len(pdf_files)}, Orgpedia: {len(orgpedia_en)})")

    # Load existing if resuming
    already_done = set()
    all_questions = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        already_done = {q.get("source_doc") for q in all_questions if q.get("source_doc")}
        print(f"Resuming: {len(already_done)} docs already processed, {len(all_questions)} questions loaded")

    # Add not-found questions once (only if not resuming or not already present)
    if not any(q.get("category") == "not_found" for q in all_questions):
        all_questions.extend(NOT_FOUND_QUESTIONS)
        print(f"Added {len(NOT_FOUND_QUESTIONS)} not-found questions")

    for i, (filepath, doc_type) in enumerate(all_docs):
        filename = os.path.basename(filepath)
        if filename in already_done:
            print(f"[{i+1}/{len(all_docs)}] SKIP {filename} (already done)")
            continue

        try:
            if doc_type == "pdf":
                import pymupdf4llm
                doc_text = pymupdf4llm.to_markdown(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    doc_text = f.read()
        except Exception as e:
            print(f"[{i+1}/{len(all_docs)}] SKIP {filename} (Error reading file: {e})")
            continue

        print(f"[{i+1}/{len(all_docs)}] Processing {filename} ({len(doc_text):,} chars)...")

        # Skip docs that are too large to feed meaningfully
        max_chars = args.max_doc_chars
        if max_chars > 0 and len(doc_text) > max_chars:
            print(f"  [SKIP] Too large ({len(doc_text):,} chars > {max_chars:,} limit). "
                  f"Run with --max-doc-chars 0 to force truncation instead.")
            continue

        # PDFs get 4 questions (has GR number lookup), orgpedia gets 3
        n_q = 4 if doc_type == "pdf" else 3
        questions = generate_questions_for_doc(gemini, doc_text, filename, n_q, trim_chars=9000)

        if questions:
            print(f"  Generated {len(questions)} questions")
            all_questions.extend(questions)

            # Save incrementally after each doc
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)

        # Rate limit: small delay between calls
        time.sleep(0.5)

    print(f"\nDone! Total questions: {len(all_questions)}")
    print(f"Saved to: {args.out}")

    # Summary by category
    cats = {}
    for q in all_questions:
        c = q.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
