import sqlite3
import json
import hashlib
import secrets

DB_PATH = "mimir_portal.db"

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            token_hash TEXT PRIMARY KEY,
            label TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            user_id TEXT PRIMARY KEY,
            history_json TEXT NOT NULL
        )
    ''')
    
    seed_tokens = [
        ("OFFICER-TOKEN-1", "Officer 1"),
        ("OFFICER-TOKEN-2", "Officer 2")
    ]
    
    for raw_token, label in seed_tokens:
        t_hash = hash_token(raw_token)
        c.execute("SELECT token_hash FROM tokens WHERE token_hash=?", (t_hash,))
        if not c.fetchone():
            c.execute("INSERT INTO tokens (token_hash, label) VALUES (?, ?)", (t_hash, label))
            
    conn.commit()
    conn.close()

def validate_token(token: str) -> bool:
    """Returns True if the token exists in the DB."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    t_hash = hash_token(token)
    c.execute("SELECT token_hash FROM tokens WHERE token_hash=?", (t_hash,))
    row = c.fetchone()
    conn.close()
    return bool(row)

def save_history(user_id: str, history_list: list):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    history_json = json.dumps(history_list)
    c.execute('''
        INSERT INTO history (user_id, history_json)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET history_json=excluded.history_json
    ''', (user_id, history_json))
    conn.commit()
    conn.close()

def get_history(user_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT history_json FROM history WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return []


def generate_officer_token(label: str) -> str:
    """Generates a new token, hashes it, stores it, and returns the raw token."""
    raw_token = f"OFFICER-{secrets.token_hex(8).upper()}"
    t_hash = hash_token(raw_token)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tokens (token_hash, label) VALUES (?, ?)", (t_hash, label))
    conn.commit()
    conn.close()
    return raw_token

def list_tokens() -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT token_hash, label FROM tokens")
    rows = c.fetchall()
    conn.close()
    return [{"token_hash": r[0], "label": r[1]} for r in rows]

def update_token_label(token_hash: str, new_label: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE tokens SET label=? WHERE token_hash=?", (new_label, token_hash))
    rows_affected = c.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0

def delete_token(token_hash: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM tokens WHERE token_hash=?", (token_hash,))
    rows_affected = c.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0
