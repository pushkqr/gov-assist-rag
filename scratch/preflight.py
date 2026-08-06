"""Pre-flight check for the Mimir demo.

Pings every service the live demo depends on and reports latency.
Read-only: touches nothing, changes nothing, safe to run any number of times.

    python -m scratch.preflight
"""
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

MARATHI_PROBE = "तंत्र शिक्षण संचालनालयातील तात्पुरत्या पदांना मान्यता देणारा शासन निर्णय"
results = []


def check(name, fn):
    start = time.perf_counter()
    try:
        detail = fn()
        elapsed = time.perf_counter() - start
        results.append((True, name, elapsed, detail))
    except Exception as exc:
        elapsed = time.perf_counter() - start
        results.append((False, name, elapsed, str(exc)[:120]))


def app_health():
    r = requests.get("https://mimir.pushkqr.app/health", timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("ready"):
        raise RuntimeError(f"not ready: {data}")
    return "ready"


def weaviate():
    from core.utils import get_weaviate_client
    from core.schema import CORPUS_COLLECTION

    # Resolved the same way app.py and scratch/regress.py resolve it. Hardcoding "GovDocs"
    # here meant this probe queried a collection the deployment had stopped using, so a
    # perfectly healthy stack reported a failed pre-flight and told the operator to fall
    # back to the recording. regress.py carried the identical bug and was fixed the same way.
    name = os.environ.get("CORPUS_COLLECTION", CORPUS_COLLECTION).strip() or CORPUS_COLLECTION
    client = get_weaviate_client()
    try:
        total = client.collections.get(name).aggregate.over_all(total_count=True).total_count
        return f"{total} chunks indexed in {name}"
    finally:
        client.close()


def _headers(key_env):
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def embeddings():
    url = os.getenv("LOCAL_EMBED_URL")
    if not url:
        raise RuntimeError("LOCAL_EMBED_URL not set")
    r = requests.post(
        url,
        json={"input": "temporary posts", "model": os.getenv("LOCAL_EMBED_MODEL_NAME", "BAAI/bge-m3")},
        headers=_headers("LOCAL_EMBED_API_KEY"),
        timeout=20,
    )
    r.raise_for_status()
    return f"{len(r.json()['data'][0]['embedding'])}-d vector"


def reranker():
    url = os.getenv("LOCAL_RERANK_URL")
    if not url:
        raise RuntimeError("LOCAL_RERANK_URL not set")
    r = requests.post(
        url,
        json={
            "query": "continuation of temporary posts",
            "documents": ["Temporary posts continued.", "Library book scheme.", "Admission rules."],
            "model": os.getenv("LOCAL_RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
            "top_n": 3,
            "return_documents": False,
        },
        headers=_headers("LOCAL_RERANK_API_KEY"),
        timeout=20,
    )
    r.raise_for_status()
    top = r.json()["results"][0]["index"]
    return f"ranked, top index {top}" + ("" if top == 0 else "  <-- expected 0, check model")


def translation():
    url = os.getenv("TRANSLATION_SERVICE_URL")
    if not url:
        raise RuntimeError("TRANSLATION_SERVICE_URL not set")
    r = requests.post(url, json={"text": MARATHI_PROBE, "source_lang": "mar_Deva", "target_lang": "eng_Latn"}, timeout=30)
    r.raise_for_status()
    out = (r.json().get("translated_text") or "").strip()
    if not out:
        raise RuntimeError("empty translation")
    if out == MARATHI_PROBE:
        raise RuntimeError("returned input unchanged (service likely deadlocked)")
    return out[:60]


def cerebras():
    from core.utils import get_cerebras_client
    client = get_cerebras_client()
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        model="gpt-oss-120b",
        max_completion_tokens=200,
        temperature=0.0,
    )
    msg = resp.choices[0].message
    text = (msg.content or getattr(msg, "reasoning", "") or "").strip()
    if not text:
        raise RuntimeError("empty completion")
    return f"responded ({len(text)} chars)"


CHECKS = [
    ("App /health", app_health),
    ("Weaviate", weaviate),
    ("Embeddings (BGE-M3)", embeddings),
    ("Reranker (cross-encoder)", reranker),
    ("Translation (IndicTrans2)", translation),
    ("Cerebras (generation)", cerebras),
]

print("\n  Mimir pre-flight\n  " + "-" * 56)
for name, fn in CHECKS:
    check(name, fn)
    ok, n, secs, detail = results[-1]
    print(f"  {'PASS' if ok else 'FAIL'}  {n:<26} {secs:5.2f}s  {detail}")

failed = [r for r in results if not r[0]]
# Weaviate's timing includes one-time client/gRPC connection setup, so it is not
# comparable to the other checks and is excluded from the slow-service warning.
slow = [r for r in results if r[0] and r[2] > 5 and r[1] != "Weaviate"]

print("  " + "-" * 56)
if failed:
    print(f"  {len(failed)} FAILED: " + ", ".join(r[1] for r in failed))
    print("  Do not rely on the live demo. Use the recording.")
elif slow:
    print("  All up, but slow: " + ", ".join(f"{r[1]} ({r[2]:.1f}s)" for r in slow))
    print("  Run a warm-up query and re-check.")
else:
    print("  All services healthy. Run one warm-up query and you are good.")
print()

sys.exit(1 if failed else 0)
