import os
import glob
import uuid
from google import genai
from google.genai import types
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from utils import generate_content_safe, embed_content_safe

def chunk_and_embed_circular(client: genai.Client, markdown_text: str, global_metadata: dict):
    """
    Process a document by splitting it hierarchically, extracting context, 
    and generating embeddings for retrieval.
    """
    from fastembed import SparseTextEmbedding
    from qdrant_client import models
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    headers_to_split_on = [
        ("PART-", "Document_Part"),
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    parent_docs = markdown_splitter.split_text(markdown_text)
    
    database_payload = []
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    
    print(f"Divided document into {len(parent_docs)} structural Parent sections.")
    
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=1536
    )
    model_name = os.getenv("MODEL_NAME", "text-embedding-004")
    
    for parent_doc in parent_docs:
        parent_context = parent_doc.page_content
        parent_metadata = parent_doc.metadata
        parent_id = str(uuid.uuid4())
        
        hierarchy_parts = []
        if "Document_Part" in parent_metadata: hierarchy_parts.append(parent_metadata["Document_Part"])
        if "Header_1" in parent_metadata: hierarchy_parts.append(parent_metadata["Header_1"])
        if "Header_2" in parent_metadata: hierarchy_parts.append(parent_metadata["Header_2"])
        if "Header_3" in parent_metadata: hierarchy_parts.append(parent_metadata["Header_3"])
        
        hierarchy_context = " > ".join(hierarchy_parts)
        doc_info = f"Document: {global_metadata.get('doc_number', 'Unknown')} ({global_metadata.get('year', 'Unknown')})"
        full_context_prefix = f"{doc_info}\nSection: {hierarchy_context}" if hierarchy_context else doc_info
        
        child_texts = child_splitter.split_text(parent_context)
        if not child_texts:
            continue
            
        enriched_child_texts = [f"Context: {full_context_prefix}\n\nContent: {child_text}" for child_text in child_texts]
        
        dense_response = embed_content_safe(
            client,
            model=model_name,
            contents=enriched_child_texts,
            config=config
        )
        sparse_embeddings = list(sparse_model.embed(child_texts))
            
        for i, child_text in enumerate(child_texts):
            dense_vector = dense_response.embeddings[i].values
            sparse_vec = sparse_embeddings[i]
            
            vector_dict = {
                "dense": dense_vector,
                "bm25": models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            }
            
            payload_metadata = {
                **parent_metadata,
                **global_metadata,
                "parent_id": parent_id,
                "parent_context": parent_context,
                "child_text": child_text,
            }
            
            database_payload.append({
                "id": str(uuid.uuid4()),
                "vector": vector_dict,
                "metadata": payload_metadata,
                "enriched_text_used_for_embedding": enriched_child_texts[i]
            })
            
    return database_payload

def run_ingestion(client: genai.Client, docs_dir: str = "docs", target_files: list = None):
    """
    Main entry point for the ingestion module.
    """
    if target_files:
        pdf_files = [os.path.join(docs_dir, f) for f in target_files]
    else:
        pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in {docs_dir}/")
        return []
        
    all_processed_records = []
    
    for target_file in pdf_files:
        if not os.path.exists(target_file):
            print(f"Warning: {target_file} does not exist, skipping.")
            continue
            
        print(f"\n--- Automating ingestion for: {target_file} ---")
        print(f"Uploading {target_file} to Gemini API for high-quality Markdown extraction...")
        
        # Upload the PDF directly to Google's servers
        uploaded_file = client.files.upload(file=target_file)
        print(f"Upload complete (File URI: {uploaded_file.uri}). Waiting for processing...", end="", flush=True)
        
        import time
        try:
            # Wait for the file to finish processing on Google's end
            while uploaded_file.state == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(3)
                uploaded_file = client.files.get(name=uploaded_file.name)
                
            print("\nProcessing complete! Generating markdown...")
                
            if uploaded_file.state == "FAILED":
                print(f"\nError: Gemini failed to process the PDF: {target_file}")
                continue
                
            prompt = (
                "Extract the entire text from this document into clean, structural Markdown. "
                "Preserve all headers (use #, ##, ###), tables, and lists exactly as they appear "
                "in the original layout. Do not summarize or skip anything. Output ONLY the markdown text."
            )
            
            # Using the latest cheap model requested with zero temperature for perfectly consistent structure
            response = generate_content_safe(
                client,
                model='gemini-3.5-flash-lite',
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            target_md = response.text
            if target_md is None:
                 print(f"Error: Gemini returned an empty response for {target_file}.")
                 continue
                 
            print(f"Extraction complete! Markdown length: {len(target_md)} characters.\n")
        except Exception as e:
            print(f"Error extracting markdown via Gemini for {target_file}: {e}")
            continue
        finally:
            # Clean up the file from Google's servers to save your quota storage
            client.files.delete(name=uploaded_file.name)
            print(f"Cleaned up {target_file} from Gemini storage.")

            
        global_metadata = {
            "doc_type": "PDF Document",
            "issuing_authority": "Government",
            "year": 2025,
            "doc_number": os.path.basename(target_file)
        }
        
        processed_records = chunk_and_embed_circular(client, target_md, global_metadata)
        all_processed_records.extend(processed_records)
        
    print(f"Successfully generated {len(all_processed_records)} Child vectors ready for upsert.")
    return all_processed_records
