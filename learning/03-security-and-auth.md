# 03 — Security, Auth & Sandboxes

**In one line:** Policy documents are strictly confidential; Mimir uses a lightweight FastAPI middleware and frontend sandboxing to ensure no one accesses your data without the key.

---

## The Auth Gate (`_auth_gate`)

You cannot leave a corporate policy RAG endpoint exposed to the public internet. Mimir implements a rigorous, lightweight security layer right at the perimeter of the FastAPI application.

We define a tiny list of public routes (the landing page, the health check, and static assets). Everything else is guarded.

```python
_AUTH_OPEN = {"/", "/app", "/health", "/evidence", "/favicon.ico", "/favicon.svg"}

async def _auth_gate(request: Request, call_next):
    # If the path isn't explicitly open...
    if request.url.path not in _AUTH_OPEN:
        # Check if the server requires a token
        if _AUTH_TOKEN:
            if not _is_authenticated(request):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        # If no token is required, ensure it's a local request
        elif not _is_loopback_client(request):
            return JSONResponse({"detail": "Remote access forbidden without token."}, status_code=403)
            
    return await call_next(request)
```

**How it works:**
1. If you set `MIMIR_AUTH_TOKEN` in your `.env` file, the server goes into lockdown mode.
2. The frontend JavaScript checks `localStorage` for a saved token.
3. Every time the frontend makes an API request to `/ask` or `/workspaces`, it intercepts the `fetch` call and injects the `Authorization: Bearer <token>` header.
4. The middleware securely compares the incoming header against the server's environment variable using `hmac.compare_digest` to prevent timing attacks.

---

## Client-Side Workspace Sandboxing

While the backend is secured by the token, the frontend organizes data into clean, isolated **Workspaces**.

Different departments have different policies. You don't want your IT runbook conversation bleeding into your HR benefits conversation. 

Mimir's Vanilla JS frontend handles this by namespaceing `localStorage`. 

When a user switches from the "Default" workspace to the "HR" workspace, the frontend dynamically loads a completely different array of chat threads. The conversation history is effectively sandboxed to that specific domain, keeping the LLM's context window clean and focused strictly on the task at hand.
