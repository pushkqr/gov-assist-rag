import hashlib
import json
import os
from typing import Any, Dict, Optional

from core.log_config import get_logger

logger = get_logger(__name__)


def get_state_path(state_path: Optional[str] = None) -> str:
    """Return default state file path if none specified."""
    if state_path:
        return state_path
    return os.path.join(os.getcwd(), "scratch", "ingestion_state.json")


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_ingestion_state(state_path: Optional[str] = None) -> Dict[str, Any]:
    """Load ingestion state dictionary from JSON file."""
    path = get_state_path(state_path)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.error(f"Could not read state file {path}: {exc}")
    return {}


def save_ingestion_state(
    file_path_or_state: str,
    file_hash_or_path: str,
    state_path_or_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Save updated file hash and metadata to ingestion state file."""
    kw_metadata = kwargs.get("metadata")
    if kw_metadata is not None:
        metadata = kw_metadata

    kw_state = kwargs.get("state_path")

    # Detect if state_path was passed as the first argument
    if os.path.exists(file_hash_or_path) and os.path.isfile(file_hash_or_path):
        state_path = file_path_or_state
        file_path = file_hash_or_path
        file_hash = state_path_or_hash
    elif kw_state:
        state_path = kw_state
        file_path = file_path_or_state
        file_hash = file_hash_or_path
    else:
        file_path = file_path_or_state
        file_hash = file_hash_or_path
        state_path = state_path_or_hash

    if not file_path or not file_hash:
        raise ValueError("file_path and file_hash are required")

    path = get_state_path(state_path)
    state = load_ingestion_state(path)
    state[os.path.abspath(file_path)] = {
        "hash": file_hash,
        "metadata": metadata or {},
    }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"Could not save state file {path}: {exc}")


def should_skip_file(file_path: str, file_hash: str, state_path: Optional[str] = None) -> bool:
    """Check if file has already been ingested with an identical hash."""
    state = load_ingestion_state(state_path)
    existing = state.get(os.path.abspath(file_path))
    if not existing:
        return False
    return existing.get("hash") == file_hash
