# 04 — Benchmark & Evaluation: Proving the Truth

**In one line:** We don't guess if the system is accurate; we run a ruthless automated test harness across 30 complex policy questions and grade the pipeline's homework.

---

## The Grounded Dataset

Building a RAG system is easy. Knowing if it actually works when you tweak a prompt or change an embedding model is incredibly hard. "Vibes-based" testing (just asking it a few questions and seeing if it feels right) doesn't scale.

Mimir includes a built-in benchmark harness (`benchmark/benchmark.json`). This is a carefully curated dataset of 30 test cases based directly on the actual policy documents stored in the system.

Each test case includes:
1. **The Query:** A realistic question an employee might ask.
2. **The Expected Output:** A human-verified summary of the correct answer.
3. **Required Terms:** Specific jargon, dates, or IDs that *must* appear in the generated answer for it to be considered accurate.

---

## The Dual-Evaluator System

When you run `python main.py` with `RUN_BENCHMARK = True`, the system simulates a user asking all 30 questions. It collects the pipeline's answers and runs them through a dual-evaluator system:

### 1. The Term-Match Scorer (Deterministic)
The system checks the generated answer against the `Required Terms` array. If the answer misses a critical term (like a specific statute number or a deadline date), it gets penalized. This ensures the LLM didn't just write a fluffy, generic response, but actually retrieved the hard facts.

### 2. The LLM Judge (Semantic)
Term-matching is rigid. Sometimes the LLM provides the correct answer using slightly different phrasing. To account for this, Mimir spins up an independent "Judge" LLM. 

The Judge is given the human-verified expected output and the pipeline's generated output, and is asked to grade the generation on a scale of 0 to 5 based on semantic accuracy and lack of hallucination.

---

## The Final Grade

The harness averages the deterministic scores and the semantic scores to produce a final, undeniable letter grade (A–D). 

This means whenever you swap out the BM25 model, adjust the Reciprocal Rank Fusion weights, or change the Gemini generation parameters, you can run the benchmark script and quantitatively prove whether you made the system better or worse. 

Honesty is a feature, not an apology. We prove our accuracy with data.

---

## Real World Examples from the Benchmark

When evaluating the system, the LLM Judge looks for completeness and lack of hallucination. Here are actual snippets from our latest benchmark run:

### Example 1: Perfect Synthesis (Case 11)
**Question:** Why was the School Connect 2.0 campaign introduced?
**Agent's Pipeline Result:** *Successfully retrieved the exact circulars explaining the National Education Policy 2020 initiatives and assembled a grounded answer.*
**Judge Score:** PASS (5/5)
**Judge Reason:** > "The candidate answer fully covers all key facts from the ideal answer, including the role of NEP 2020, the need for awareness, and continuous guidance, while also providing extensive supporting details."

### Example 2: Cross-Document Routing (Case 26)
**Question:** Which document discusses the regulation of admissions through CAP?
**Agent's Pipeline Result:** *Autonomously recognized the need to search for multiple entities, ran iterative searches, and pulled data from two distinct acts.*
**Judge Score:** PASS (5/5)
**Judge Reason:** > "The candidate correctly identifies both documents and accurately describes their respective functions, providing comprehensive and precise details that fully address the query."

### Example 3: The "Negative Test" (Case 30)
**Question:** What is the procedure for obtaining a passport in Maharashtra?
**Agent's Pipeline Result:** *Recognized that the indexed HR and education policy corpus does not contain passport information. Refused to hallucinate.*
**Judge Score:** PASS (5/5)
**Judge Reason:** > "The candidate accurately states that the information is not available in the provided documents, avoiding hallucination and aligning with the ideal answer."

---

*For more detailed insights, you can review the raw data:*
- **[The 30-Case Benchmark Dataset](../benchmark/benchmark.json)**
- **[The Complete Evaluator Results](../benchmark/benchmark_results.json)**
