import json
import time
import queue
import logging
import asyncio
import threading
from pathlib import Path
import os
import hmac
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, RedirectResponse
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
import ipaddress

logger = logging.getLogger(__name__)

app = FastAPI(title="Mimir")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_DIR = BASE_DIR / "docs"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")


_AUTH_TOKEN = os.environ.get("MIMIR_AUTH_TOKEN", "").strip()
_ADMIN_TOKEN = os.environ.get("MIMIR_ADMIN_TOKEN", "SUPER-SECRET-ADMIN-TOKEN").strip()
_AUTH_OPEN = {"/", "/app", "/health", "/evidence", "/favicon.ico", "/favicon.svg", "/login", "/portal", "/api/login", "/admin"}

_AUTHORIZED_SUBNETS = []
_env_subnets = os.environ.get("MIMIR_ALLOWED_SUBNETS")
if _env_subnets:
    _AUTHORIZED_SUBNETS = [ipaddress.ip_network(s.strip()) for s in _env_subnets.split(",") if s.strip()]
else:
    _AUTHORIZED_SUBNETS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("172.16.0.0/12"),
    ]

def _is_in_authorized_subnet(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return any(ip in subnet for subnet in _AUTHORIZED_SUBNETS)
    except ValueError:
        return False

def _is_authenticated(request: Request) -> bool:
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer "): return False
    token = h[len("Bearer "):].strip()
    
    if _AUTH_TOKEN and hmac.compare_digest(token, _AUTH_TOKEN):
        return True
        
    return validate_token(token)

@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    path = request.url.path
    is_admin_api = path.startswith("/api/admin/")

    # The network gate applies to admin routes too. Admin endpoints verify the admin token
    # themselves, so they skip the officer-token check below, but exempting them from the
    # subnet allowlist would have left a hole in the zero-trust perimeter.
    if path not in _AUTH_OPEN or is_admin_api:
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            client_host = x_forwarded_for.split(",")[0].strip()
        else:
            client_host = request.client.host if request.client else ""

        if not _is_in_authorized_subnet(client_host):
            return JSONResponse({
                "detail": "Network Access Denied. Device is outside authorized government intranet."
            }, status_code=403)

        if _AUTH_TOKEN and not is_admin_api:
            if not _is_authenticated(request):
                return JSONResponse({"detail": "Unauthorized — provide the access token."}, status_code=401)

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

@app.get("/app")
async def serve_app():
    """Retired in favour of /portal, which is the maintained officer interface.

    Kept as a redirect rather than removed so existing links and bookmarks still land
    somewhere sensible. templates/app.html is now unused.
    """
    return RedirectResponse(url="/portal", status_code=307)

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
    user_id: str = None
    history: List[Dict[str, Any]]

@app.post("/api/history")
async def api_save_history(req: HistorySaveRequest, request: Request):
    h = request.headers.get("authorization", "")
    token = h[len("Bearer "):].strip()
    save_history(token, req.history)
    return {"status": "ok"}

@app.get("/api/history")
async def api_get_history(request: Request, user_id: str = None):
    h = request.headers.get("authorization", "")
    token = h[len("Bearer "):].strip()
    history = get_history(token)
    return {"history": history}

class TokenCreateRequest(BaseModel):
    label: str

@app.get("/api/admin/tokens")
async def api_admin_list_tokens(request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    from db import list_tokens
    return {"tokens": list_tokens()}

@app.post("/api/admin/tokens")
async def api_admin_create_token(req: TokenCreateRequest, request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
        
    from db import generate_officer_token
    new_token = generate_officer_token(req.label)
    return {"token": new_token, "label": req.label}

class TokenUpdateRequest(BaseModel):
    label: str

@app.put("/api/admin/tokens/{token_hash}")
async def api_admin_update_token(token_hash: str, req: TokenUpdateRequest, request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    from db import update_token_label
    if update_token_label(token_hash, req.label):
        return {"status": "ok"}
    return JSONResponse({"error": "Token not found."}, status_code=404)

@app.delete("/api/admin/tokens/{token_hash}")
async def api_admin_delete_token(token_hash: str, request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    from db import delete_token
    if delete_token(token_hash):
        return {"status": "ok"}
    return JSONResponse({"error": "Token not found."}, status_code=404)

def _is_admin(request: Request) -> bool:
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer "):
        return False
    return hmac.compare_digest(h[len("Bearer "):].strip(), _ADMIN_TOKEN)


_FORBIDDEN = JSONResponse({"error": "Admin access required."}, status_code=403)

# Single-flight ingestion state. Ingestion is minutes-long and blocking, so it runs on a
# background thread and the panel polls this for progress.
_ingest = {"running": False, "file": None, "log": [], "error": None, "finished_at": None}


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html")


@app.post("/api/admin/login")
async def api_admin_login(req: LoginRequest):
    if not _ADMIN_TOKEN or not hmac.compare_digest(req.token.strip(), _ADMIN_TOKEN):
        return JSONResponse({"error": "Invalid admin token."}, status_code=401)
    return {"token": req.token.strip()}


@app.get("/api/admin/stats")
async def api_admin_stats(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    from db import list_tokens
    stats = {"chunks": None, "documents": 0, "pdfs": 0, "orgpedia": 0, "officers": len(list_tokens())}
    try:
        if DOCS_DIR.exists():
            # One source document is either a PDF or an Orgpedia GR. Orgpedia GRs ship as a
            # pair (<id>.pdf.txt original, <id>.pdf.en.txt translation), so count only the
            # .en.txt side to avoid double-counting one document as two.
            files = [p for p in DOCS_DIR.rglob("*") if p.is_file()]
            stats["files"] = len(files)
            stats["pdfs"] = sum(1 for p in files if p.suffix.lower() == ".pdf")
            # Orgpedia GRs ship as a pair per document (<id>.pdf.en.txt English,
            # <id>.pdf.mr.txt Marathi), so the file count is double the document count.
            stats["orgpedia"] = sum(1 for p in files if p.name.lower().endswith(".en.txt"))
            stats["documents"] = stats["pdfs"] + stats["orgpedia"]
    except Exception as e:
        logger.warning(f"Could not count source documents: {e}")
    try:
        agg = weaviate_client.collections.get("GovDocs").aggregate.over_all(total_count=True)
        stats["chunks"] = agg.total_count
    except Exception as e:
        logger.warning(f"Could not read Weaviate stats: {e}")
    return stats


@app.post("/api/admin/upload")
async def api_admin_upload(request: Request, file: UploadFile = File(...)):
    if not _is_admin(request):
        return _FORBIDDEN
    name = os.path.basename(file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are accepted."}, status_code=400)
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Empty file."}, status_code=400)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / name).write_bytes(data)
    return {"filename": name, "bytes": len(data)}


def _ingest_job(filename: str):
    _ingest.update(running=True, file=filename, log=[f"Starting ingestion of {filename}"], error=None, finished_at=None)
    try:
        from ingestion import run_ingestion
        records = run_ingestion(
            gemini_client,
            weaviate_client=weaviate_client,
            collection_name="GovDocs",
            docs_dir=str(DOCS_DIR),
            target_files=[filename],
        )
        _ingest["log"].append(f"Indexed {len(records)} chunks from {filename}")
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        _ingest["error"] = str(e)
        _ingest["log"].append(f"Failed: {e}")
    finally:
        _ingest["running"] = False
        _ingest["finished_at"] = __import__("datetime").datetime.now().isoformat()


class IngestRequest(BaseModel):
    filename: str


@app.post("/api/admin/ingest")
async def api_admin_ingest(req: IngestRequest, request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    if _ingest["running"]:
        return JSONResponse({"error": f"Ingestion already running for {_ingest['file']}."}, status_code=409)
    name = os.path.basename(req.filename or "").strip()
    if not (DOCS_DIR / name).is_file():
        return JSONResponse({"error": f"{name} not found in docs."}, status_code=404)
    threading.Thread(target=_ingest_job, args=(name,), daemon=True).start()
    return {"status": "started", "filename": name}


@app.get("/api/admin/ingest/status")
async def api_admin_ingest_status(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    return _ingest


@app.get("/api/admin/documents")
async def api_admin_documents(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    if not DOCS_DIR.exists():
        return {"documents": []}
    docs = [
        {"filename": p.name, "bytes": p.stat().st_size}
        for p in sorted(DOCS_DIR.rglob("*"), key=lambda x: x.name)
        if p.suffix.lower() == ".pdf"
    ]
    return {"documents": docs}


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
            # run_retrieval is blocking and runs on a worker thread, so its progress callbacks
            # are handed back through a queue and drained here while we wait. Without this the
            # user stares at a static spinner for the whole retrieval.
            status_q: "queue.Queue[str]" = queue.Queue()

            def sync_status(msg: str):
                try:
                    status_q.put_nowait(msg)
                except Exception:
                    pass

            _t_start = time.perf_counter()
            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(
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

            while not task.done():
                try:
                    yield json.dumps({"status": status_q.get_nowait()}) + "\n"
                except queue.Empty:
                    await asyncio.sleep(0.05)

            retrieval_result = await task
            while True:
                try:
                    yield json.dumps({"status": status_q.get_nowait()}) + "\n"
                except queue.Empty:
                    break


            status = retrieval_result.get("status")
            if status == "error":
                yield json.dumps({"error": retrieval_result.get("response_text")}) + "\n"
                return
                
            answer_stream = retrieval_result.get("answer_stream")
            evidence = retrieval_result.get("evidence", [])
            recommendations = retrieval_result.get("recommendations", [])
            metrics = dict(retrieval_result.get("metrics") or {})

            first_token_at = None
            if answer_stream:
                for chunk in answer_stream:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield json.dumps({"t": chunk}) + "\n"
                    await asyncio.sleep(0.01)

            if first_token_at is not None:
                metrics["first_token_s"] = round(first_token_at - _t_start, 3)
            metrics["total_s"] = round(time.perf_counter() - _t_start, 3)

            yield json.dumps({
                "done": True,
                "citations": evidence,
                "recommendations": recommendations,
                "metrics": metrics,
            }) + "\n"
            
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
