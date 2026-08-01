# 02 — Security, Auth & Extensibility

**In one line:** Policy documents are highly confidential; Mimir uses a combination of Zero-Trust Intranet Geofencing, isolated Token Identities, and strict API access controls to ensure your data never leaves the network.

---

## 1. Zero-Trust Intranet Geofencing

You cannot leave a government RAG endpoint exposed to the public internet, even with passwords. Mimir implements a rigorous, preventative security layer right at the perimeter of the FastAPI application.

```python
_AUTHORIZED_SUBNETS = [
    ipaddress.ip_network("10.0.0.0/8"),       # Government / Enterprise Intranet
    ipaddress.ip_network("127.0.0.0/8"),      # Localhost loopback
]
```

Every incoming request to the API is mathematically verified against the authorized government subnets. If a hacker steals a valid access token and tries to use it from their home Wi-Fi or a coffee shop, the middleware intercepts the connection at the TCP/socket level and drops it with a `403 Network Access Denied` before the LLM is ever invoked.

---

## 2. Token-Based Multi-Tenancy

Mimir discards vulnerable email/password architectures in favor of a secure, hashed Token Registry stored in an isolated SQLite database (`db.py`).

- **Secure Storage**: Only the SHA-256 hashes of the tokens are stored in the database.
- **Cross-Device Sync**: The frontend passes the token in the `Authorization: Bearer` header. The backend extracts the token, hashes it, and queries the database for that specific officer's chat history.
- **Data Isolation**: This guarantees multi-tenant data isolation. The IDOR vulnerabilities common in client-side architectures are impossible because the server dictates identity purely by the cryptographic token, not client-provided IDs.

---

## 3. The Admin Token CRUD API

Mimir features a fully baked API for IT departments to programmatically provision and manage officer access. Protected by the `MIMIR_ADMIN_TOKEN` environment variable, the backend exposes:

- `POST /api/admin/tokens`: Generates a random secure token, hashes it, and stores it.
- `GET /api/admin/tokens`: Returns a list of all active tokens (hashes only).
- `DELETE /api/admin/tokens/{token_hash}`: Instantly revokes an officer's access globally.

---

## 4. An Extensible Engine

Because Mimir is built as an agnostic engine, extending it to a new department (e.g., Department of Finance) requires zero backend security changes. You simply:
1. Spin up a new Weaviate collection with Finance documents.
2. Deploy the Mimir Engine.
3. Provision Finance Officer tokens via the Admin API.

The underlying security, auth, and geofencing work out of the box for any department.
