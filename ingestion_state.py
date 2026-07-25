import hashlib
import json
import os
from typing import Any, Dict, Optional


def get_state_path(state_path: Optional[str] = None) -> str:
    if state_path:
        return state_path
    return os.path.join(os.getcwd(), "ingestion_state.json")


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_ingestion_state(state_path: Optional[str] = None) -> Dict[str, Any]:
    path = get_state_path(state_path)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"[Ingestion State] Failed to read state file {path}: {exc}")
    return {}


def save_ingestion_state(*args, **kwargs) -> None:
    file_path = None
    file_hash = None
    state_path = kwargs.get("state_path")
    metadata = kwargs.get("metadata")

    if len(args) == 2:
        file_path, file_hash = args
    elif len(args) == 3:
        first, second, third = args
        if os.path.exists(first) and os.path.isfile(first):
            file_path, file_hash, state_path = first, second, third
        else:
            state_path, file_path, file_hash = first, second, third
    elif len(args) == 4:
        first, second, third, fourth = args
        if os.path.exists(first) and os.path.isfile(first):
            file_path, file_hash, state_path, metadata = first, second, third, fourth
        else:
            state_path, file_path, file_hash, metadata = first, second, third, fourth
    else:
        raise TypeError("save_ingestion_state expects 2-4 positional arguments")

    if not file_path or not file_hash:
        raise ValueError("file_path and file_hash are required")

    path = get_state_path(state_path)
    state = load_ingestion_state(path)
    state[os.path.abspath(file_path)] = {
        "hash": file_hash,
        "metadata": metadata or {},
    }

    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[Ingestion State] Failed to save state file {path}: {exc}")


def should_skip_file(file_path: str, file_hash: str, state_path: Optional[str] = None) -> bool:
    state = load_ingestion_state(state_path)
    existing = state.get(os.path.abspath(file_path))
    if not existing:
        return False
    return existing.get("hash") == file_hash
