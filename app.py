import json
import logging
import asyncio
from pathlib import Path
import os
import hmac
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from google import genai
from core.utils import get_genai_client
from retrieval import run_retrieval

logger = logging.getLogger(__name__)

app = FastAPI(title="Mimir")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Optional shared-secret auth — OFF by default. Set MIMIR_AUTH_TOKEN
# to require `Authorization: Bearer <token>` on every API route before any PUBLIC deploy.
_AUTH_TOKEN = os.environ.get("MIMIR_AUTH_TOKEN", "").strip()
_AUTH_OPEN = {"/", "/app", "/health", "/evidence", "/favicon.ico", "/favicon.svg"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1"}

def _is_authenticated(request: Request) -> bool:
    if not _AUTH_TOKEN: return False
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer "): return False
    return hmac.compare_digest(h[len("Bearer "):], _AUTH_TOKEN)

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

# Global clients
gemini_client = None
qdrant_client = None

@app.on_event("startup")
async def startup_event():
    global gemini_client, qdrant_client
    try:
        gemini_client = get_genai_client()
        qdrant_client = QdrantClient(path="local_qdrant_db")
        logger.info("Clients initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")

@app.get("/app", response_class=HTMLResponse)
async def serve_app(request: Request):
    return templates.TemplateResponse(request=request, name="app.html")

# -- Mock endpoints to satisfy frontend dependencies --
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

# -- Core Mimir Endpoints --

class AskRequest(BaseModel):
    query: str
    history: List[Dict[str, Any]] = []
    workspace: Optional[str] = "default"

@app.post("/ask-stream")
async def ask_stream(req: AskRequest):
    async def event_generator():
        mapped_history = []
        for h in req.history:
            mapped_history.append({
                "role": "user" if h.get("role") == "user" else "model",
                "text": h.get("content", h.get("text", ""))
            })

        try:
            # We must run this in a threadpool since run_retrieval uses sync threads and requests
            loop = asyncio.get_running_loop()
            
            def status_cb(msg):
                # We could stream status to frontend if needed
                pass
                
            retrieval_result = await loop.run_in_executor(
                None,
                lambda: run_retrieval(
                    gemini_client=gemini_client,
                    qdrant_client=qdrant_client,
                    query=req.query,
                    collection_name="gov_docs",
                    chat_history=mapped_history,
                    status_callback=status_cb
                )
            )
            
            status = retrieval_result.get("status")
            if status == "error":
                yield json.dumps({"error": retrieval_result.get("response_text")}) + "\n"
                return
                
            answer_stream = retrieval_result.get("answer_stream")
            evidence = retrieval_result.get("evidence", [])
            
            if answer_stream:
                for chunk in answer_stream:
                    yield json.dumps({"t": chunk}) + "\n"
                    # Small sleep to yield to event loop
                    await asyncio.sleep(0.01)
            
            # Send done and citations
            yield json.dumps({"done": True, "citations": evidence}) + "\n"
            
        except Exception as e:
            logger.error(f"Error in ask-stream: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
            
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
