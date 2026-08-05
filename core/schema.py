"""Weaviate collection schema, shared by the corpus and the quarantine collection.

main.py still defines the same properties inline for the batch ingestion path. That copy is
left alone deliberately; this module exists so the quarantine collection is guaranteed to
match the corpus, which is what makes promoting a document a straight object copy rather
than a re-ingestion.
"""

from typing import Any

import weaviate.classes as wvc

from core.log_config import get_logger

logger = get_logger(__name__)

CORPUS_COLLECTION = "GovDocs"
QUARANTINE_COLLECTION = "Quarantine"

_TEXT_PROPERTIES = [
    "translated_text", "child_text", "parent_context", "document_title",
    "doc_number", "issuing_authority", "document_category", "source_filename",
    "supersedes", "references",
]


def collection_properties() -> list:
    props = [
        wvc.config.Property(name=name, data_type=wvc.config.DataType.TEXT)
        for name in _TEXT_PROPERTIES
    ]
    props.insert(5, wvc.config.Property(name="year", data_type=wvc.config.DataType.INT))
    return props


def ensure_collection(weaviate_client: Any, name: str) -> None:
    """Create the collection if it is missing. Safe to call on every request."""
    if weaviate_client.collections.exists(name):
        return
    logger.info(f"Creating Weaviate collection '{name}'")
    weaviate_client.collections.create(name=name, properties=collection_properties())
