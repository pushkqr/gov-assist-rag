# 04 — Architecture Deep Dive

**In one line:** Mimir is a production RAG engine for government policy documents, built on hybrid retrieval, self-hosted open-weight models, and a measured evaluation harness. It passes 88 of 100 benchmark cases inside a sub-10-second latency budget.

Every figure and code reference here is taken from the source. Where something is unmeasured, it says so.

---

## System at a Glance

| | |
|---|---|
| Corpus | 533 documents (33 PDFs, 500 Orgpedia GRs), 10,194 indexed chunks |
| Accuracy | 88/100 benchmark pass rate, judge average 4.13/5, term average 0.653 |
| Latency | 3 to 8 seconds end to end |
| Infrastructure | 3 droplets, roughly $10 to $20/month |
| Languages | English, Marathi, Hindi |

---

## 1. Hybrid Retrieval with Dynamic Alpha

Two retrieval paradigms fail in opposite directions.

**Dense vector search** matches meaning. It finds a document about "continuation of temporary posts" when the officer asked about "rules for extending temporary staff," which share no keywords. It is weak on exact identifiers, because a GR number is a character string, not a concept.

**BM25 keyword search** matches literally. Perfect for `MUWAD-2016/(38/16)/MASHI-1`. Useless when the officer's phrasing differs from the document's.

Weaviate runs both and fuses them, weighted by alpha. The key decision is that **alpha is not fixed** ([search.py:220-227](../retrieval/search.py#L220-L227)):

| Query shape | Alpha | Reasoning |
|---|---|---|
| Contains a GR-code pattern | 0.25 | Lean keyword: the identifier is an exact string |
| Everything else | 0.50 | Balanced fusion |

Detection is a regex for GR-code patterns against the standalone query. This single adjustment fixed an entire class of failures where dense similarity drowned out an exact identifier match.

### Deterministic Query Expansion

Rather than spend an LLM call expanding every query, fast mode uses `build_fast_search_query` ([search.py:21](../retrieval/search.py#L21)), which appends bilingual aliases when it detects known policy vocabulary. Triggers include GR references (adding `Government Resolution No`, `शासन निर्णय`, `क्रमांक`), appointments, professorial designations, probation, transfers, and dates.

It is deterministic and costs no tokens. The tradeoff is coverage: it helps only for vocabulary that was anticipated. Learned expansion would generalize better at the cost of latency.

### Known bug: alias dilution

The GR trigger fires on the *phrase* "government resolution" or "GR", not on an actual GR code. So a prose question about a specific named entity gets generic vocabulary appended:

```
"What department issued the Government Resolution for the Maharashtra State
 Loksahitya Samiti's term extension?"
  + "Government Resolution No GR number Government Decision शासन निर्णय क्रमांक"
```

Those added terms match essentially every document in a corpus of government resolutions, so they swamp the one distinctive token, `Loksahitya`. Measured directly against BM25:

| Query sent to BM25 | Correct document in top 5 |
|---|---|
| Raw user question | **Yes**, ranks 1 through 4 |
| After alias expansion | **No**, returns unrelated scholarship GRs |

The expansion demotes the correct document from rank 1 to off the list, and the pipeline then answers "no information found" for a document it holds and can trivially find.

The fix is to gate the alias block on a GR *code* pattern rather than the phrase, which is what the alpha-tuning regex already detects. It is deliberately not applied here: some queries that mention "Government Decision" in prose currently succeed via this path, so the change needs a full dual-grader benchmark run to confirm it is a net gain rather than a trade of one failure class for another.

---

## 2. Self-Hosted Embeddings and Reranking

The embedding model and the cross-encoder reranker share one droplet behind [Infinity](https://github.com/michaelfeil/infinity), which exposes an OpenAI-compatible API.

- **BGE-M3** (BAAI), multilingual, 1024-dimensional. Every ingestion and query embedding routes here.
- **BGE-reranker-v2-m3**, a cross-encoder that scores query and candidate jointly.

Self-hosting was not premature optimization. Vertex AI embedding quotas were actively throttling ingestion, which is what forced the move. The result is zero per-query embedding cost and no quota ceiling.

### Why Reranking Needs a Second Pass

The first search compares query and document *separately*, since both were embedded independently. A cross-encoder reads them *together*, which is more accurate and much slower. So retrieval is a funnel: `FAST_MODE_CANDIDATE_LIMIT=20` candidates from hybrid search, narrowed to `FAST_MODE_RERANK_LIMIT=12` after reranking.

**Measured cost.** Reranking accounts for roughly 3.4s of a 5.3s end-to-end response, about 60 to 65 percent of total latency. Because the funnel is narrow (20 in, 12 out), it discards only 8 candidates; most of what it buys is *ordering* the 12 that survive.

**Measured benefit, and a lesson about measuring it.** An A/B over 10 benchmark cases, scored by term overlap alone, showed *no* difference in accuracy and a 4s saving, which looked like a clear case for switching it off (`RERANK_ENABLED=false`).

That conclusion was wrong. Spot-checking real queries showed a large qualitative gap the metric could not see. On a Marathi query about temporary-post approvals:

- **With reranking:** the specific GR plus its exact validity dates.
- **Without:** a hedged list of three different GRs, none identified as the answer.

Both answers contain many of the same terms, so term overlap scored them identically. The difference is precision, which only the LLM judge half of the harness detects.

The lesson generalizes past this system: **a cheap proxy metric can report "no regression" for a change that noticeably degrades output.** The dual-grader design exists for exactly this, and using half of it produced a confidently wrong answer. Reranking stays enabled; the toggle remains for deployments that would rather have 2s responses than maximum precision.

### The Latency Regression

An early version of `build_rerank_text` fed roughly 2,200 characters per candidate into the reranker, including full parent context. Search latency rose to 13 to 15 seconds, breaking the sub-10-second requirement.

The cause is structural: a cross-encoder scores `(query, document)` as a pair, so per-candidate length multiplies directly into total latency. The fix capped each candidate at roughly 900 characters, keeping document anchors (title, document number, section) and truncating the body to 600. Latency returned to 3 to 8 seconds with no measurable accuracy loss.

**The lesson worth carrying:** with cross-encoders, the input budget *is* a latency budget.

### Result Diversification

Hybrid search frequently returns many chunks from one document. `diversify_results` ([search.py:89](../retrieval/search.py#L89)) caps chunks per document at `FAST_MODE_MAX_CHUNKS_PER_DOC` (default 4), keyed on `source_filename` with fallback to `doc_number` then `document_title`, holding overflow back to backfill if the limit isn't reached.

This prevents evidence pile-on: five chunks from one document repeating a detail read to the model as strong corroboration, even when that document is the wrong one.

---

## 3. Multilingual Support

Officers query in Marathi, Hindi, and English. The corpus is indexed in English.

**At query time:** Devanagari input is detected and sent to a self-hosted IndicTrans2 service (AI4Bharat / IIT Madras) on its own droplet, returning English before retrieval.

**At ingestion time:** Marathi source text is batch-translated through GCP Cloud Translation v3 and stored alongside the original, so chunks carry both.

Translating at query time rather than searching Marathi directly is a deliberate choice about hybrid search. BM25 is purely lexical, so a Devanagari query against English text shares no tokens and that half of the search returns nothing. BGE-M3 is multilingual and would partly cope, but translating keeps **both** halves working in every language, against one index rather than parallel ones.

**Operational note:** this was the system's most fragile component. It deadlocked silently under RAM pressure on an undersized droplet, hanging on every Indic query with no error output. `docker stats` showing near-zero CPU during the hang is what proved deadlock rather than slowness.

---

## 4. Generation: Open-Weight Models with Failover

Two open-weight models alternate round-robin per request ([pipeline.py:59](../retrieval/pipeline.py#L59)):

```python
_CEREBRAS_MODELS = ["gpt-oss-120b", "gemma-4-31b"]
```

Both are served through Cerebras for inference speed, at temperature 0. On any Cerebras failure the request falls back to Gemini 2.5 Flash via Vertex AI ([pipeline.py:193-204](../retrieval/pipeline.py#L193-L204)).

**The architecturally significant point:** Cerebras is an inference host, not a model vendor. `gpt-oss-120b` and `gemma-4-31b` are openly published, as are BGE-M3, BGE-reranker-v2-m3, and IndicTrans2. **No proprietary model is load-bearing.** Moving generation on-premise means serving those same weights locally and repointing configuration, not re-architecting. Gemini remains only as a fallback path and for ingestion-time batch translation.

---

## 5. Ingestion

### Three-Tier Parser Fallback

Government PDFs arrive in inconsistent condition, so extraction degrades through three tiers:

1. **PyMuPDF4LLM** — fast, local, handles well-formed PDFs
2. **Google Document AI OCR** — scanned or image-only circulars
3. **Gemini Vision** — last resort for what defeats both

Orgpedia GRs arrive as pre-translated `.en.txt` plaintext and skip parsing entirely. That difference matters when assessing corpus difficulty: a corpus of only `.en.txt` files never exercises tiers 2 or 3.

### Table-Aware Chunking

Naive fixed-size chunking splits tables mid-body, orphaning rows from their headers. A row reading `| Kolhapur | 30 |` is meaningless once separated from the header saying what 30 counts.

`chunk_and_embed_circular` ([chunking.py:65](../ingestion/chunking.py#L65)) detects table boundaries and prepends the nearest preceding header rows to isolated row chunks before embedding.

### Parent-Child Hierarchy

Child chunks are the embedded, searchable unit. Parent sections supply surrounding context at generation time. This keeps the search index tight while giving the model enough context to interpret what it retrieved.

### Idempotence

Ingestion tracks processed files by hash in `scratch/ingestion_state.json`. Re-running is safe and never duplicates. This is what makes scaling the corpus a scheduling problem rather than an engineering one.

---

## 6. Security

### Zero-Trust Network Gating

Middleware validates the client IP against an allowlist **before** authentication is checked ([app.py:74-92](../app.py#L74-L92)). The list is environment-configured via `MIMIR_ALLOWED_SUBNETS`, defaulting to loopback and RFC1918 private ranges. Requests from outside are refused with 403.

Deploying inside a department means setting that variable to the department's range. Public exposure requires an explicit configuration change, so the secure posture is the default.

### Token Identity

No passwords. Officers hold generated tokens; only SHA-256 hashes are stored ([db.py](../db.py)). Comparison uses `hmac.compare_digest` to avoid timing leaks. An admin API and console handle provisioning, renaming, and revocation.

### Known Gaps

Stated plainly, because pretending otherwise is worse:

- **No per-document authorization.** Every authenticated officer sees the whole corpus. Acceptable while the corpus is published material; a prerequisite before anything confidential is indexed.
- **Admin routes bypass the subnet gate**, self-checking the admin token instead. Defensible, but inconsistent with the zero-trust posture elsewhere.
- **Audit logging is informal.** Conversations persist per token, but there is no compliance-grade audit trail.

---

## 7. Evaluation

Knowing whether a change helped is the hard part of RAG. Mimir has a harness rather than intuition.

**Dataset.** Candidate questions were generated document by document, then hand-filtered to 100 verified cases after removing ambiguous phrasing and near-duplicate document collisions. Coverage spans simple English, Marathi, Hindi, multi-document synthesis, direct GR lookups, and deliberately out-of-corpus questions.

**Dual grading.** A deterministic scorer checks that required terms (GR numbers, dates, counts) actually appear. An LLM judge scores factual accuracy 0 to 5 against a human-verified expected answer. Term matching alone is too rigid; a judge alone is too lenient on missing specifics.

**Pass rule.** `judge >= 3.0`, or `judge >= 2.0 AND term >= 0.5`.

**Result: 88/100.** Judge average 4.13, term average 0.653, up from an 83/100 baseline. The improvement came from two fixes, not from touching the test set:

1. Resolving the translation deadlock, which had been silently failing every Indic query.
2. Wiring three retrieval helpers that existed but were never called: `build_fast_search_query`, `build_rerank_text`, and `diversify_results`.

**The remaining 12%** share one pattern: the correct document is retrieved, but the model states a date or GR number belonging to a near-identical order about a different person or case. This is generation-side entity attribution, not retrieval failure. Naming the failure mode precisely is what makes it addressable.

An attempted prompt-level fix scored 87/100, worse than baseline, and was reverted. Recorded because the negative result is part of the finding.

---

## 8. Deployment

| Droplet | Runs |
|---|---|
| Application | FastAPI backend, Docker, Caddy TLS reverse proxy |
| Inference | BGE-M3 and BGE-reranker-v2-m3 via Infinity |
| Translation | IndicTrans2 |

Weaviate runs remotely. Total infrastructure is roughly $10 to $20/month. Cerebras is on a free tier and GCP on startup credits, so that is a pilot figure, not a production one.

### Secrets and Docker

A production incident worth recording. `gcp-key.json` was being copied into the image by `COPY . .`. When the file did not exist at build time, Docker created an empty **directory** at that path and baked it into the layer. Once the real key appeared on the host, bind-mounting a file onto a path the image believed was a directory failed with a type mismatch, and every embedding call died with `Is a directory`.

Two things fixed it: adding `gcp-key.json` to `.dockerignore`, and rebuilding with `--no-cache` to drop the poisoned layer. Secrets now arrive only as runtime bind mounts, which is also the correct security posture.

---

## Design Tradeoffs

| Decision | Bought | Cost |
|---|---|---|
| Hybrid over dense-only | Exact identifier lookups work | Extra latency per query |
| Dynamic alpha | Fixed a failure class | Regex heuristic, not learned |
| Self-hosted embeddings | No quota ceiling, no per-query cost | A droplet to operate |
| Cross-encoder reranking | Better ordering into the model | Input length is a latency budget |
| Deterministic expansion | No token cost, no added latency | Covers only anticipated vocabulary |
| Two generation models | No single point of failure | Two integrations to maintain |
| Open-weight throughout | On-prem is configuration, not rewrite | Rules out frontier-only capabilities |
| Translate-then-retrieve | Both halves of hybrid search work | Translation is a hard dependency |

---

## Known Limitations

- **Retrieval does not see conversation history.** Chat history reaches the generation prompt but not the search step, so a follow-up like "what is the number of this GR" is searched without its antecedent and retrieves broadly. Fixing this means rewriting the query from history before search.
- **Chat history is unbounded.** It concatenates without a window and will eventually exceed the context limit.
- **Entity attribution on near-duplicate documents**, the dominant remaining failure mode.
- **Ingestion is manually triggered.** The pipeline is idempotent and ready for scheduling; the scheduler is not built.
- **No per-document access control**, as noted above.
