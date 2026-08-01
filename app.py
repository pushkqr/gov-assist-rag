import json
import logging
import asyncio
from pathlib import Path
import os
import hmac
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from google import genai
import uvicorn
import weaviate
import weaviate.classes as wvc
from core.utils import get_genai_client, get_cerebras_client, get_weaviate_client
from retrieval import run_retrieval
from db import init_db, validate_token, save_history, get_history

logger = logging.getLogger(__name__)

app = FastAPI(title="Mimir")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_DIR = BASE_DIR / "docs"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")


_AUTH_TOKEN = os.environ.get("MIMIR_AUTH_TOKEN", "").strip()
_AUTH_OPEN = {"/", "/app", "/health", "/evidence", "/favicon.ico", "/favicon.svg", "/login", "/portal", "/api/login", "/api/history"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1"}

def _is_authenticated(request: Request) -> bool:
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer "): return False
    token = h[len("Bearer "):].strip()
    
    if _AUTH_TOKEN and hmac.compare_digest(token, _AUTH_TOKEN):
        return True
        
    return validate_token(token)

def _is_loopback_client(request: Request) -> bool:
    return bool(request.client) and request.client.host in _LOOPBACK_HOSTS

@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    if request.url.path not in _AUTH_OPEN:
        if _AUTH_TOKEN:
            if not _is_authenticated(request):
                return JSONResponse({"detail": "Unauthorized — provide the access token."}, status_code=401)
        elif not _is_loopback_client(request):
            return JSONResponse({"detail": "This instance is not configured for remote access."}, status_code=403)
    return await call_next(request)

gemini_client = None
cerebras_client = None
weaviate_client = None

@app.on_event("startup")
async def startup_event():
    global gemini_client, cerebras_client, weaviate_client
    try:
        init_db()
        gemini_client = get_genai_client()
        cerebras_client = get_cerebras_client()
        weaviate_client = get_weaviate_client()
        logger.info("Clients initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")

@app.get("/app", response_class=HTMLResponse)
async def serve_app(request: Request):
    return templates.TemplateResponse(request=request, name="app.html")

@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/portal", response_class=HTMLResponse)
async def serve_portal(request: Request):
    return templates.TemplateResponse(request=request, name="portal.html")

class LoginRequest(BaseModel):
    token: str

@app.post("/api/login")
async def api_login(req: LoginRequest):
    if not validate_token(req.token):
        return JSONResponse({"error": "Invalid Officer Token."}, status_code=401)
    return {"token": req.token}

class HistorySaveRequest(BaseModel):
    user_id: str
    history: List[Dict[str, Any]]

@app.post("/api/history")
async def api_save_history(req: HistorySaveRequest):
    save_history(req.user_id, req.history)
    return {"status": "ok"}

@app.get("/api/history")
async def api_get_history(user_id: str):
    history = get_history(user_id)
    return {"history": history}

@app.get("/health")
async def health():
    return {"auth": bool(_AUTH_TOKEN), "demo": False, "ready": True}

@app.get("/workspaces")
async def workspaces():
    return {"workspaces": [{"id": "default", "name": "Mimir Workspace"}]}

@app.get("/systems")
async def systems():
    return {"systems": []}

@app.get("/curation")
async def curation():
    return {"findings": []}

@app.get("/curation/aging")
async def curation_aging():
    return {"aging": []}

@app.get("/timeline")
async def timeline():
    return {"events": []}

@app.get("/graph")
async def graph():
    return {"nodes": [], "edges": []}

@app.get("/llm-config")
async def llm_config():
    return {"provider": "gemini"}

@app.get("/ingest-status")
async def ingest_status():
    return {"running": False}

@app.get("/download/{filename}")
async def download_file(filename: str):
    for root_dir, _, files in os.walk(DOCS_DIR):
        if filename in files:
            return FileResponse(os.path.join(root_dir, filename))
    return JSONResponse({"error": "File not found"}, status_code=404)

class AskRequest(BaseModel):
    query: str
    history: List[Dict[str, Any]] = []
    workspace: Optional[str] = "default"

@app.post("/ask-stream")
async def ask_stream(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    history = data.get("history", [])

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    formatted_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "text" in msg:
            formatted_history.append({"role": msg["role"], "text": msg["text"]})

    async def event_generator():
        try:
            def sync_status(msg: str):
                pass 
                
            loop = asyncio.get_running_loop()
            retrieval_result = await loop.run_in_executor(
                None,
                lambda: run_retrieval(
                    gemini_client=gemini_client,
                    cerebras_client=cerebras_client,
                    weaviate_client=weaviate_client,
                    query=query,
                    collection_name="GovDocs",
                    chat_history=formatted_history,
                    status_callback=sync_status
                )
            )
            
            status = retrieval_result.get("status")
            if status == "error":
                yield json.dumps({"error": retrieval_result.get("response_text")}) + "\n"
                return
                
            answer_stream = retrieval_result.get("answer_stream")
            evidence = retrieval_result.get("evidence", [])
            recommendations = retrieval_result.get("recommendations", [])
            
            if answer_stream:
                for chunk in answer_stream:
                    yield json.dumps({"t": chunk}) + "\n"
                    await asyncio.sleep(0.01)
            
            yield json.dumps({"done": True, "citations": evidence, "recommendations": recommendations}) + "\n"
            
        except Exception as e:
            logger.error(f"Error in ask-stream: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
            
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback: str

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    try:
        feedback_file = BASE_DIR / "scratch" / "feedback.json"
        os.makedirs(feedback_file.parent, exist_ok=True)
        
        feedback_entry = {
            "query": req.query,
            "response": req.response,
            "feedback": req.feedback,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        
        feedbacks = []
        if feedback_file.exists():
            with open(feedback_file, "r", encoding="utf-8") as f:
                try:
                    feedbacks = json.load(f)
                except:
                    pass
                    
        feedbacks.append(feedback_entry)
        
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, indent=2)
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

class SummarizeRequest(BaseModel):
    doc_id: str

@app.post("/summarize")
async def summarize_doc(req: SummarizeRequest):
    try:
        weaviate_collection = weaviate_client.collections.get("GovDocs")
        res = weaviate_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("doc_number").equal(req.doc_id),
            limit=50
        )
        pts = res.objects
        if not pts: return JSONResponse({"error": "Document not found."})
        text = "\n".join([(p.properties.get("child_text") or p.properties.get("parent_context") or "") for p in pts])
            
        resp = gemini_client.models.generate_content(
            model=os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash"),
            contents=f"Summarize the following document concisely:\n\n{text[:30000]}"
        )
        return {"summary": resp.text}
    except Exception as e:
        return JSONResponse({"error": str(e)})

class CompareRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str

@app.post("/compare")
async def compare_docs(req: CompareRequest):
    try:
        def fetch(did):
            weaviate_collection = weaviate_client.collections.get("GovDocs")
            res = weaviate_collection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("doc_number").equal(did),
                limit=50
            )
            pts = res.objects
            return "\n".join([(p.properties.get("child_text") or p.properties.get("parent_context") or "") for p in pts])
                
        t1, t2 = fetch(req.doc_id_1), fetch(req.doc_id_2)
        resp = gemini_client.models.generate_content(
            model=os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash"),
            contents=f"Compare these two documents and highlight the differences:\n\nDocument 1:\n{t1[:15000]}\n\nDocument 2:\n{t2[:15000]}"
        )
        return {"comparison": resp.text}
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
