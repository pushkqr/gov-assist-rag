# 01 — Hybrid Retrieval: The Best of Both Worlds

**In one line:** Vector search finds concepts, keyword search finds exact part numbers, and Reciprocal Rank Fusion combines them so Mimir never misses a critical policy.

---

## Why Dense Vector Search Isn't Enough

The modern AI boom is obsessed with Dense Embeddings (Vector Search). You take a sentence, squash it into an array of 384 floating-point numbers, and plot it in "meaning space." 

If you ask *"What is the policy for taking time off?"*, vector search is brilliant because it knows that *"annual leave"*, *"PTO"*, and *"vacation days"* all live in the exact same meaning space. 

**The fatal flaw:** Vector models are terrible at exact matches for jargon. If an employee asks, *"What is the deadline for filing Form 1040-EZ?"*, the vector model might pull up documents about "Form 1099" or "Tax Deadlines" because they are *semantically* similar, entirely missing the specific "1040-EZ" document.

## Why BM25 (Keyword Search) Isn't Enough

BM25 is the algorithm behind Elasticsearch. It's the old-school way of searching: counting how many times a word appears in a document, adjusted by how rare that word is across the whole library.

If you search for *"Form 1040-EZ"*, BM25 is a sniper rifle. It will hit the exact document instantly.

**The fatal flaw:** If an employee asks about *"taking time off"*, but the HR manual only uses the term *"paid annual leave"*, BM25 returns absolutely zero results because the words literally don't match.

---

## The Mimir Solution: Reciprocal Rank Fusion (RRF)

We don't choose. We run both.

When you ask a question, Mimir fires off two searches simultaneously to our Qdrant database:
1. **The Vector Search:** Finds the top 20 chunks based on meaning and concepts.
2. **The Sparse BM25 Search:** Finds the top 20 chunks based on exact keyword overlap.

We then use a mathematical algorithm called **Reciprocal Rank Fusion (RRF)** to combine the two lists. 

```python
# The math behind the magic
RRF_Score = 1 / (k + Vector_Rank) + 1 / (k + BM25_Rank)
```

If a document chunk ranks #1 in the Vector search, it gets a high score. If it ranks #1 in the Keyword search, it gets a high score. If it ranks highly in *both*, its score compounds and it rockets to the absolute top of the context window.

This guarantees that whether a user is asking a vague conceptual question, or searching for a highly specific government statute number, the correct context is injected into the LLM prompt every single time.
