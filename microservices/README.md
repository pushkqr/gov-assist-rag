# Mimir, on-premise deployment

Every component Mimir depends on, packaged to run inside a department's own network with no
outbound internet access at runtime.

The design goal was that going on-premise is a configuration change, not a rewrite. Retrieval
already talks to its embedding, reranking and translation services over HTTP, so those move
by changing a base URL. Generation was the one component bound to a third-party API; it now
sits behind the same kind of interface.

## What runs where

| Component | Image | Self-hosted | Purpose |
|---|---|---|---|
| Vector store | `semitechnologies/weaviate` | yes | Chunk storage, hybrid dense + BM25 search |
| Embeddings | `michaelf34/infinity` | yes | BGE-M3, 1024-dimensional vectors |
| Reranker | `michaelf34/infinity` | yes | BGE cross-encoder over retrieved candidates |
| Translation | built from `./translation` | yes | IndicTrans2, Marathi to English |
| Generation | `ollama/ollama` | yes | Answer synthesis |
| Document parsing | built from `../docling_ingestion` | yes | Optional, for scanned or table-heavy PDFs |

No component in this stack calls out to a third party once the images and model weights are
present. Weights are pulled once at install time and cached in named volumes.

## Bringing it up

```bash
cp microservices/.env.example microservices/.env
# fill in WEAVIATE_API_KEY and INFINITY_API_KEY

docker compose -f microservices/docker-compose.yml --env-file microservices/.env up -d
docker compose -f microservices/docker-compose.yml exec generation ollama pull qwen3:4b
```

Then point the application's `.env` at the stack. The commented block at the bottom of
`.env.example` lists exactly which variables change; `GEN_PROVIDER=local` is the switch that
moves generation off Cerebras.

Add `--profile full` to include the optional Docling parser.

## Hardware

The stack runs on CPU as written, which is enough to demonstrate the full pipeline. It is not
enough to be pleasant.

- **CPU only, 16GB RAM.** Works. Reranking dominates query latency and generation with a 4B
  model is slow enough to be felt. Suitable for evaluation.
- **One NVIDIA GPU with 16GB or more.** The intended configuration. Uncomment the `deploy`
  blocks for `infinity` and `generation` in the compose file. Reranking drops by roughly an
  order of magnitude, and a 12B to 30B generation model becomes practical.

Disk: about 12GB for images, plus roughly 5GB of model weights, plus the corpus index.

## Verification status

Being explicit about what has and has not been run, because a deployment artifact that
quietly does not work is worse than none.

**Verified end to end.** The application's local generation path. Retrieval, reranking,
citation and the conflict-detection logic were all exercised against a self-hosted
OpenAI-compatible server, with the correct model name reported through to the UI, streaming
intact, and a verified fallback when the server is unreachable.

**Written and reviewed, not yet run.** The compose file and the translation service. The
translation service implements the exact request and response contract the application
already uses in `retrieval/search.py`, and the Weaviate and Infinity services are configured
to match the working remote deployment. They have not been brought up on a machine with
enough disk and memory to pull the weights.

Two things to check on first run, both of which are version-sensitive rather than design
issues:

- Infinity's CLI flags. `v2 --model-id` is correct for current versions; confirm against the
  tag you pull.
- `IndicTransToolkit` install. It occasionally needs building from source depending on the
  Python and PyTorch combination.

## Why generation was the last piece

Embeddings, reranking and translation were self-hosted from early on, because those models
are small enough to run economically on commodity hardware and because sending every query's
text to a third-party embedding API was not acceptable for government documents.

Generation stayed on a hosted API for one reason: speed. Cerebras returns a full answer in
about two seconds, which is what keeps end-to-end query time under six. A 4B model on a CPU
node will not match that, and a 30B model on one GPU will not either.

So the tradeoff is explicit rather than hidden. If a deployment requires that no query text
leaves the building, `GEN_PROVIDER=local` satisfies that today and costs response time. If it
does not, the hosted path is faster. Both use identical retrieval, identical grounding, and
identical citations, because generation is the last step and it only ever sees text that
retrieval already selected.
