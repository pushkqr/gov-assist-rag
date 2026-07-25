import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams

from benchmark import load_benchmark_cases, print_benchmark_report, run_benchmark
from ingestion import run_ingestion
from log_config import get_logger
from retrieval import run_retrieval

logger = get_logger(__name__)


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    qdrant = QdrantClient(path="local_qdrant_db")
    collection_name = "gov_docs"

    RUN_INGESTION = True
    RUN_RETRIEVAL = False
    RUN_BENCHMARK = True
    USE_LOCAL_PARSER = False

    if RUN_INGESTION:
        print("\n" + "=" * 50)
        print("MODULE: INGESTION PIPELINE")
        print("=" * 50)

        logger.info(f"Creating/Updating Qdrant collection: '{collection_name}' with Hybrid Search support")
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )

        records = run_ingestion(client, docs_dir="docs", use_local_parser=USE_LOCAL_PARSER)
        if records:
            logger.info(f"Upserting {len(records)} chunks into Qdrant...")
            points = [
                PointStruct(id=record["id"], vector=record["vector"], payload=record["metadata"])
                for record in records
            ]
            qdrant.upsert(collection_name=collection_name, points=points)

            # Create payload indices (active when deployed on Qdrant Server)
            logger.info("Setting Qdrant payload index schema...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    qdrant.create_payload_index(collection_name, field_name="year", field_schema=models.PayloadSchemaType.INTEGER)
                    qdrant.create_payload_index(collection_name, field_name="doc_number", field_schema=models.PayloadSchemaType.KEYWORD)
                    qdrant.create_payload_index(collection_name, field_name="document_category", field_schema=models.PayloadSchemaType.KEYWORD)
                    qdrant.create_payload_index(collection_name, field_name="section_title", field_schema=models.PayloadSchemaType.KEYWORD)
                except Exception as exc:
                    pass

            print("Upsert and indexing complete! Vectors are now stored locally.")

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
