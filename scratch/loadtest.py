"""Retrieval latency against corpus size.

Answers "does this hold up at volume" with a measurement rather than an assertion. Loads
synthetic chunks into a throwaway collection, and at each checkpoint measures hybrid search
latency there while also querying the live corpus, so any degradation of production is
visible immediately rather than discovered later.

Two notes on honesty of the numbers:

Vectors are random. Random high-dimensional vectors have no cluster structure, which is the
unfavourable case for HNSW: a real corpus is easier to traverse than this. Timings here are
an upper bound, not a flattering one.

Only the vector store is exercised. Embedding, reranking and generation are fixed costs per
query that do not vary with corpus size, so they are excluded deliberately; the question is
how search scales, and those stages would swamp the signal.

    python -m scratch.loadtest --max 100000
    python -m scratch.loadtest --cleanup-only
"""

import argparse
import random
import statistics
import time

from dotenv import load_dotenv

load_dotenv()

import weaviate.classes as wvc  # noqa: E402

from core.utils import get_weaviate_client  # noqa: E402
from core.schema import collection_properties  # noqa: E402

TEST_COLLECTION = "LoadTest"
LIVE_COLLECTION = "GovDocs"
DIMS = 1024

# Vocabulary lifted from the register of the real corpus, so BM25 sees a realistic term
# distribution rather than uniform noise.
VOCAB = """government resolution maharashtra department higher technical education circular
notification sanction posts temporary continuation directorate approval professor assistant
associate appointment probation transfer recruitment eligibility criteria scholarship
admission intake capacity institute college university grant aided unaided principal
secretary mantralaya dated subject reference decision clause section provision applicable
committee proposal budget expenditure financial year candidate reserved category relaxation
""".split()


def synth_text(rng: random.Random, words: int = 60) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(words))


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def measure(collection, rng, runs=25, limit=20):
    """Hybrid dense + BM25 latency, matching what the application actually issues."""
    timings = []
    for _ in range(runs):
        vector = [rng.uniform(-1, 1) for _ in range(DIMS)]
        query = " ".join(rng.choice(VOCAB) for _ in range(6))
        started = time.perf_counter()
        collection.query.hybrid(
            query=query,
            query_properties=["translated_text", "parent_context", "child_text", "doc_number"],
            vector=vector,
            alpha=0.5,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=2_000)
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()

    rng = random.Random(20260805)
    client = get_weaviate_client()

    try:
        if client.collections.exists(TEST_COLLECTION):
            print(f"Dropping existing {TEST_COLLECTION}")
            client.collections.delete(TEST_COLLECTION)
        if args.cleanup_only:
            print("Cleanup done.")
            return

        client.collections.create(name=TEST_COLLECTION, properties=collection_properties())
        test = client.collections.get(TEST_COLLECTION)
        live = client.collections.get(LIVE_COLLECTION)

        # Discarded warm-up. The first queries of a run carry connection and gRPC channel
        # setup and read several times slower than steady state, which would otherwise be
        # mistaken for a baseline.
        measure(live, rng, runs=8)

        live_baseline = measure(live, rng)
        print(f"\nLive corpus baseline: p50 {statistics.median(live_baseline):.0f} ms  "
              f"p95 {percentile(live_baseline, 95):.0f} ms  "
              f"({live.aggregate.over_all(total_count=True).total_count} chunks)\n")

        checkpoints = [c for c in (10_000, 25_000, 50_000, 100_000, 200_000) if c <= args.max]
        print(f"{'size':>9}  {'insert':>8}  {'p50':>7}  {'p95':>7}  {'p99':>7}   {'live p50':>8}")
        print("-" * 60)

        loaded = 0
        for target in checkpoints:
            insert_started = time.perf_counter()
            while loaded < target:
                count = min(args.batch, target - loaded)
                with test.batch.fixed_size(batch_size=200) as batch:
                    for i in range(count):
                        batch.add_object(
                            properties={
                                "child_text": synth_text(rng),
                                "parent_context": synth_text(rng, 120),
                                "translated_text": synth_text(rng),
                                "document_title": f"Synthetic Resolution {loaded + i}",
                                "doc_number": f"LOAD-{2000 + (loaded + i) % 25}/{(loaded + i) % 900}",
                                "year": 2000 + ((loaded + i) % 25),
                                "source_filename": f"synthetic-{(loaded + i) % 5000}.pdf",
                                "issuing_authority": "Government",
                                "document_category": "Resolution",
                            },
                            vector=[rng.uniform(-1, 1) for _ in range(DIMS)],
                        )
                loaded += count
            insert_s = time.perf_counter() - insert_started

            # Report what actually landed, not what was requested: individual batches can
            # time out under load, and quietly labelling the row 100,000 would overstate it.
            indexed = test.aggregate.over_all(total_count=True).total_count
            timings = measure(test, rng)
            live_now = measure(live, rng, runs=10)
            print(f"{indexed:>9,}  {insert_s:>7.0f}s  "
                  f"{statistics.median(timings):>6.0f}m  {percentile(timings, 95):>6.0f}m  "
                  f"{percentile(timings, 99):>6.0f}m   {statistics.median(live_now):>7.0f}m")

        print("\nLive corpus after load: p50 "
              f"{statistics.median(measure(live, rng)):.0f} ms")
    finally:
        try:
            if client.collections.exists(TEST_COLLECTION):
                client.collections.delete(TEST_COLLECTION)
                print(f"Dropped {TEST_COLLECTION}")
        finally:
            client.close()


if __name__ == "__main__":
    main()
