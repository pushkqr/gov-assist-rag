# 01 — Hybrid Retrieval: The Best of Both Worlds

**In one line:** Vector search finds concepts, keyword search finds exact GR numbers, dynamic Alpha Fusion combines them — so Mimir never misses a critical policy regardless of how the officer phrases the question.

---

## Why Dense Vector Search Isn't Enough

Dense embeddings map text into a high-dimensional "meaning space." Ask *"what are the rules for taking leave?"* and a vector search brilliantly knows that *"annual leave"*, *"casual leave"*, and *"vacation days"* all live nearby in that space.

**The fatal flaw:** Vector models are terrible at exact jargon matches. If an officer asks about *"GR No. MUWAD-2016/(38/16)/MASHI-1"*, the vector model may pull up vaguely related posts-extension documents because they are *semantically* similar — completely missing the specific GR the officer needs.

---

## Why BM25 (Keyword Search) Isn't Enough

BM25 is the algorithm behind Elasticsearch and traditional search engines. It counts how many times a word appears in a document, adjusted by how rare that word is across the whole corpus.

Search for *"MUWAD-2016/(38/16)"* and BM25 is a sniper rifle — it hits the exact document instantly.

**The fatal flaw:** If an officer asks *"posts that were extended in Higher Education offices last quarter"*, but the circular only uses the Marathi term *"तात्पुरती पदे"*, BM25 returns nothing because the words don't literally match.

---

## The Mimir Solution: Dynamic Alpha Fusion

We don't choose. We run both.

When a query arrives, Mimir fires two searches simultaneously inside Weaviate:
1. **Dense Vector Search:** Finds top chunks by meaning and concepts using **BGE-M3** embeddings (1024-d multilingual model).
2. **Sparse BM25 Search:** Finds top chunks by exact keyword overlap across `translated_text`, `child_text`, `parent_context`, `section_title`, and `doc_number`.

Weaviate then combines the two ranked lists using **Reciprocal Rank Fusion** weighted by `alpha`:

```python
# The math behind the magic
Hybrid_Score = (alpha × Vector_Score) + ((1 - alpha) × BM25_Score)
```

---

## Dynamic Alpha Tuning

Mimir doesn't use a fixed alpha — it adapts based on the query:

```python
# From retrieval/search.py
gr_code_pattern = r"[A-Z]{2,}[-/]\d+|P\.?No\.?\s*\d+|No\.\s+\d+/"
if re.search(gr_code_pattern, standalone_query):
    alpha = 0.25   # BM25-heavy: exact GR number lookup
else:
    alpha = 0.50   # Balanced: general policy question
```

| Query type | Alpha | Effect |
|---|---|---|
| `"GR No. MUWAD-2016/(38/16)/MASHI-1"` | 0.25 | BM25 dominates — finds the exact circular |
| `"how many temporary posts were extended?"` | 0.50 | Balanced — finds by meaning |
| Marathi query (translated to English first) | 0.50 | Vector excels at semantic proximity after translation |

---

## Why BGE-M3 Instead of Gemini Embeddings?

The original design used Google's `gemini-embedding-001` (via Vertex AI). We migrated to **BGE-M3** (BAAI) for two reasons:

1. **No quota limits.** Vertex AI embedding quotas throttled ingestion to a crawl even on a few hundred documents, and the corpus has since grown into the thousands. BGE-M3 runs on a dedicated Infinity server: no rate limits, no per-call cost at runtime. The cost moves rather than disappearing, and on a CPU node it is throughput. Passage length dominates it, so batch size and request timeout have to be sized together against your own corpus.
2. **Multilingual by design.** BGE-M3 is explicitly trained on multilingual corpora including Indic languages. This means translated Marathi text embeds more faithfully in the same space as English queries.

The tradeoff: BGE-M3 produces 1024-d vectors vs. Gemini's 3072-d. In practice, for a focused government document corpus this is more than sufficient.

---

## The BM25 Query Properties

The BM25 search targets these fields specifically:

```python
query_properties=[
    "translated_text",    # GCP-translated English version of Marathi chunks
    "parent_context",     # Full parent section text
    "section_title",      # Structural header hierarchy
    "child_text",         # The original chunk text
    "doc_number",         # GR number (e.g. "MUWAD-2016/(38/16)/MASHI-1")
]
```

`doc_number` in BM25 is the key to GR number lookup accuracy — if an officer types a GR number verbatim, it scores almost perfectly against the exact document.

---

## Deep Search Mode: LLM Reranking

> **Not currently exposed.** `run_retrieval` defaults to `fast_mode=True` and nothing in the API or UI passes `False`, so this path is unreachable in the running product. It is documented here because the code exists, not as a shipped feature. Every live query uses the cross-encoder path above.

In non-fast mode (`fast_mode=False`), after the hybrid search returns 150 candidates, an LLM judge reranks them:

```python
# retrieval/search.py — rerank_results()
# Gives the LLM all candidate passages, asks for ranked indices by relevance
# Returns top 12 reranked passages for generation
```

This is expensive but gives the highest accuracy for complex synthesis queries that span multiple documents.
