# 02 — The Cache Hero: Instant Answers

**In one line:** RAG is slow and expensive; our persistent cache intercepts repeated questions and serves them in zero milliseconds, saving your API quota and providing a snappy UX.

---

## The Problem with Traditional RAG

Every time a user asks a question in a standard RAG pipeline, the system:
1. Calls an embedding API (e.g., Google Gemini).
2. Performs a database search across vectors.
3. Assembles the context.
4. Calls a generative LLM API to stream an answer token-by-token.

This takes anywhere from **2 to 5 seconds**. 

If 50 employees all ask *"When is the office closed for Thanksgiving?"* on the same morning, a traditional system runs that 5-second, API-cost-incurring pipeline 50 times to generate the exact same answer. That is slow, expensive, and completely unnecessary.

---

## The Cheat Sheet Analogy

Imagine a librarian fielding questions from a long line of students. 

The first student asks for the Wi-Fi password. The librarian gets up, walks to the back room, searches the filing cabinet, finds the password, walks back, and tells the student. 

Before the librarian sits down, they write the password on a sticky note and stick it to the front desk. 

When the next 49 students ask for the Wi-Fi password, the librarian doesn't move. They just point to the sticky note. 

---

## How Mimir's Cache Works

Mimir implements a lightning-fast interception layer using a simple, persistent JSON dictionary (`scratch/mimir_cache.json`).

```python
# Example of the persistent cache structure
{
  "What is the holiday schedule for 2024?": {
    "answer": "The office is closed on the following dates...",
    "timestamp": "2024-01-01T10:00:00Z"
  }
}
```

1. **The Intercept:** The moment a user submits a query to the `/ask` endpoint, Mimir normalizes the string (lowercasing, stripping whitespace) and checks the dictionary.
2. **The Hit:** If the exact string exists, Mimir instantly returns the pre-generated answer. The LLM is never called. The vector database is never queried. The response time drops from ~3,000ms to **~2ms**.
3. **The Miss:** If the string is new, Mimir runs the full RAG pipeline, generates the answer, streams it to the user, and *then* saves it to the cache dictionary for the next person.

This single feature drastically improves the perceived performance of the app for common policy questions while dramatically reducing your cloud provider bill.
