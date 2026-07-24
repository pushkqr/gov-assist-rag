import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client import models

from ingestion import run_ingestion
from retrieval import run_retrieval

def main():
    # Load environment variables (API keys, MODEL_NAME, etc.)
    load_dotenv()
    
    # Initialize the central GenAI client to be shared across modules
    client = genai.Client()
    
    # Initialize Local Qdrant Client (Creates a folder named 'local_qdrant_db')
    qdrant = QdrantClient(path="local_qdrant_db")
    collection_name = "gov_docs"
    
    # ==========================================
    # CENTRAL TOGGLES - Turn modules ON or OFF
    # ==========================================
    RUN_INGESTION = False
    RUN_RETRIEVAL = True
    
    # Choose between parsing PDFs locally via PyMuPDF or remotely via Gemini File API
    USE_LOCAL_PARSER = True
    
    if RUN_INGESTION:
        print("\n" + "="*50)
        print("MODULE: INGESTION PIPELINE")
        print("="*50)
        
        # Drop and recreate collection for Hybrid Search schema
        print(f"Creating/Updating Qdrant collection: '{collection_name}' with Hybrid Search support")
        from qdrant_client import models
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)
            
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        
        # We can pass the target directory
        records = run_ingestion(client, docs_dir="docs", use_local_parser=USE_LOCAL_PARSER)
        
        if records:
            print(f"\n[Database Hook] -> Upserting {len(records)} chunks into Qdrant...")
            points = [
                PointStruct(
                    id=record["id"], 
                    vector=record["vector"], 
                    payload=record["metadata"]
                ) for record in records
            ]
            qdrant.upsert(collection_name=collection_name, points=points)
            print("Upsert complete! Vectors are now stored locally.")
        
    if RUN_RETRIEVAL:
        print("\n" + "="*50)
        print("MODULE: RETRIEVAL & GENERATION PIPELINE (INTERACTIVE MODE)")
        print("="*50)
        print("Type 'exit' or 'quit' to stop testing. Type 'clear' to reset conversation history.")
        
        chat_history = []
        
        while True:
            try:
                query = input("\n[GovAssist] Ask a question: ")
                if query.strip().lower() in ['exit', 'quit']:
                    print("Exiting GovAssist...")
                    break
                if query.strip().lower() == 'clear':
                    chat_history.clear()
                    print("[Memory] Conversation history cleared!")
                    continue
                if not query.strip():
                    continue
                    
                answer = run_retrieval(client, qdrant, query, collection_name, chat_history)
                print(f"\n{'-'*50}\nAssistant Response:\n{answer}\n{'-'*50}")
                
                # Append to history memory
                chat_history.append({"role": "user", "text": query})
                chat_history.append({"role": "model", "text": answer})
                
            except KeyboardInterrupt:
                print("\nExiting GovAssist...")
                break

if __name__ == "__main__":
    main()
