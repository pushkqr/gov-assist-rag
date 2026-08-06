"""Validate the configuration a running process actually has.

Configuration is spread across the application's own environment and one .env per service,
and the values that must agree across those files are not obvious from any single one of
them. When they disagree the symptom is a service that starts, answers health checks, and
then fails on real traffic, which is expensive to diagnose from the far end.

This checks the environment the calling process actually holds rather than what a file on
disk says, because those are not the same thing. A variable can be present in .env and
still never reach the container if the compose file does not list it, which is a real
failure this project has hit: the service had the value, the container did not.

Remote services are checked by probing them and comparing what they report against what
this side is configured to expect. A reranker whose loaded model differs from
LOCAL_RERANK_MODEL_NAME answers HTTP 400 on every query, so agreement is worth asserting
rather than assuming.
"""

import os
from typing import List, Tuple
from urllib.parse import urlparse

import core.deployment as deployment
from core.health import host_of

ERROR = "error"
WARN = "warn"
OK = "ok"

Finding = Tuple[str, str, str]  # (level, subject, detail)


def _host(url: str) -> str:
    return host_of(url) if url else ""


def _required(sovereign_gen: bool, ingest_indictrans: bool) -> List[Tuple[str, str]]:
    """(variable, what breaks without it). Conditional on the providers actually in use."""
    required = [
        ("WEAVIATE_URL", "no vector store to search"),
        ("WEAVIATE_API_KEY", "Weaviate refuses the connection"),
        ("LOCAL_EMBED_URL", "queries cannot be embedded"),
        ("LOCAL_EMBED_API_KEY", "Infinity rejects the request"),
        ("LOCAL_RERANK_URL", "retrieval cannot rerank candidates"),
        ("TRANSLATION_SERVICE_URL", "Marathi and Hindi queries fail"),
        ("MIMIR_ADMIN_TOKEN", "the admin console is unreachable"),
    ]
    if sovereign_gen:
        required += [
            ("LOCAL_GEN_URL", "no generation endpoint in sovereign mode"),
            ("LOCAL_GEN_MODEL", "the generation service is asked for an unnamed model"),
        ]
    if ingest_indictrans:
        required += [
            ("INGEST_TRANSLATION_SERVICE_URL",
             "ingestion silently falls back to the query translation service"),
        ]
    return required


def check_config(probe_services: bool = True) -> List[Finding]:
    findings: List[Finding] = []

    sovereign_gen = deployment.gen_provider() == "local"
    ingest_indictrans = deployment.ingest_translate_provider() == "indictrans2"

    # 1. Present at all.
    for var, consequence in _required(sovereign_gen, ingest_indictrans):
        if os.getenv(var, "").strip():
            findings.append((OK, var, "set"))
        else:
            findings.append((ERROR, var, f"missing: {consequence}"))

    # 2. Values that have to agree with each other.
    embed_url = os.getenv("LOCAL_EMBED_URL", "")
    rerank_url = os.getenv("LOCAL_RERANK_URL", "")
    if embed_url and rerank_url:
        if _host(embed_url) == _host(rerank_url):
            embed_key = os.getenv("LOCAL_EMBED_API_KEY", "")
            rerank_key = os.getenv("LOCAL_RERANK_API_KEY", "")
            if embed_key and rerank_key and embed_key != rerank_key:
                findings.append((ERROR, "Infinity API key",
                                 "LOCAL_EMBED_API_KEY and LOCAL_RERANK_API_KEY differ but point "
                                 "at the same server, so one of the two will be rejected"))
            else:
                findings.append((OK, "Infinity API key", "consistent across embed and rerank"))
        else:
            findings.append((WARN, "Infinity host",
                             f"embeddings ({_host(embed_url)}) and reranking ({_host(rerank_url)}) "
                             "are on different hosts, which is supported but unusual"))

    # The embedding model is not a preference. Every stored vector is 1024-dimensional and a
    # different model makes the whole index unreadable rather than merely worse.
    embed_model = os.getenv("LOCAL_EMBED_MODEL_NAME", "") or os.getenv("EMBED_MODEL_NAME", "")
    if embed_model and embed_model != "BAAI/bge-m3":
        findings.append((ERROR, "LOCAL_EMBED_MODEL_NAME",
                         f"is {embed_model}, not BAAI/bge-m3: the existing index cannot be read "
                         "with a different embedding model"))
    elif embed_model:
        findings.append((OK, "LOCAL_EMBED_MODEL_NAME", "BAAI/bge-m3, matches the stored index"))

    # Two translation instances exist so ingestion can use the larger, slower model. Pointing
    # both at one URL still works, which is why it is worth saying out loud.
    query_tr = os.getenv("TRANSLATION_SERVICE_URL", "")
    ingest_tr = os.getenv("INGEST_TRANSLATION_SERVICE_URL", "")
    if ingest_indictrans and query_tr and ingest_tr:
        if _host(query_tr) == _host(ingest_tr):
            findings.append((WARN, "translation split",
                             "query and ingestion translation share one host, so ingestion is "
                             "using the smaller query-tier model"))
        else:
            findings.append((OK, "translation split", "query and ingestion use separate services"))

    # Batch size and request timeout are one coupled decision on CPU: throughput scales with
    # passage length, so a large batch of long passages can exceed the timeout every time.
    try:
        batch = int(os.getenv("EMBED_BATCH_SIZE", "64"))
        timeout = float(os.getenv("LOCAL_EMBED_BATCH_TIMEOUT_S", "60"))
        # ~3.4s per long passage measured on a 2-vCPU node.
        worst_case = batch * 3.4
        if worst_case > timeout:
            findings.append((WARN, "EMBED_BATCH_SIZE",
                             f"{batch} passages could need ~{int(worst_case)}s but "
                             f"LOCAL_EMBED_BATCH_TIMEOUT_S is {int(timeout)}s; long-passage "
                             "documents will time out"))
        else:
            findings.append((OK, "EMBED_BATCH_SIZE",
                             f"{batch} within the {int(timeout)}s batch timeout"))
    except ValueError:
        findings.append((WARN, "EMBED_BATCH_SIZE", "not a number"))

    if not probe_services:
        return findings

    # 3. What the remote services actually report, versus what this side expects.
    from core.health import probe, probe_embeddings, probe_generation

    embed_result = probe(probe_embeddings)
    if embed_result["status"] != "up":
        findings.append((ERROR, "embeddings service", embed_result["detail"]))
    elif "1024" not in embed_result["detail"]:
        findings.append((ERROR, "embedding width",
                         f"service returned {embed_result['detail']}, but the corpus is "
                         "1024-dimensional"))
    else:
        findings.append((OK, "embeddings service", "returns 1024-d vectors"))

    if sovereign_gen:
        gen_result = probe(probe_generation)
        if gen_result["status"] != "up":
            findings.append((ERROR, "generation service", gen_result["detail"]))
        else:
            findings.append((OK, "generation service", f"reachable, serving {gen_result['detail']}"))

    return findings
