import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams

from benchmark import load_benchmark_cases, print_benchmark_report, run_benchmark
from ingestion import run_ingestion
from retrieval import run_retrieval


def main():
    load_dotenv()
    client = genai.Client()
    qdrant = QdrantClient(path="local_qdrant_db")
    collection_name = "gov_docs"

    RUN_INGESTION = False
    RUN_RETRIEVAL = True
    RUN_BENCHMARK = False
    USE_LOCAL_PARSER = True

    if RUN_INGESTION:
        print("\n" + "=" * 50)
        print("MODULE: INGESTION PIPELINE")
        print("=" * 50)

        print(f"Creating/Updating Qdrant collection: '{collection_name}' with Hybrid Search support")
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

        records = run_ingestion(client, docs_dir="docs", use_local_parser=USE_LOCAL_PARSER)
        if records:
            print(f"\n[Database Hook] -> Upserting {len(records)} chunks into Qdrant...")
            points = [
                PointStruct(id=record["id"], vector=record["vector"], payload=record["metadata"])
                for record in records
            ]
            qdrant.upsert(collection_name=collection_name, points=points)
            print("Upsert complete! Vectors are now stored locally.")

    if RUN_BENCHMARK:
        print("\n" + "=" * 50)
        print("MODULE: CORPUS BENCHMARK & EVALUATION HARNESS")
        print("=" * 50)
        cases = load_benchmark_cases()
        if cases:
            benchmark_report = run_benchmark(client, qdrant, cases)
            print_benchmark_report(benchmark_report)
        else:
            print("No benchmark cases found. Create a benchmark.json file.")

    if RUN_RETRIEVAL:
        print("\n" + "=" * 50)
        print("MODULE: RETRIEVAL & GENERATION PIPELINE (INTERACTIVE MODE)")
        print("=" * 50)
        print("Type 'exit' or 'quit' to stop testing. Type 'clear' to reset conversation history.")

        chat_history = []

        while True:
            try:
                query = input("\n[GovAssist] Ask a question: ")
                if query.strip().lower() in ["exit", "quit"]:
                    print("Exiting GovAssist...")
                    break
                if query.strip().lower() == "clear":
                    chat_history.clear()
                    print("[Memory] Conversation history cleared!")
                    continue
                if not query.strip():
                    continue

                retrieval_result = run_retrieval(client, qdrant, query, collection_name, chat_history)

                print(f"\n{'-' * 50}\nAssistant Response:\n")
                full_answer = ""
                if retrieval_result["status"] != "success":
                    full_answer = (
                        retrieval_result.get("response_text")
                        or "I couldn't find any relevant documents to answer your question."
                    )
                    print(full_answer)
                else:
                    response_stream = retrieval_result["answer_stream"]
                    for chunk in response_stream:
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_answer += chunk
                print(f"\n{'-' * 50}")

                chat_history.append({"role": "user", "text": query})
                chat_history.append({"role": "model", "text": full_answer})

            except KeyboardInterrupt:
                print("\nExiting GovAssist...")
                break


if __name__ == "__main__":
    main()
