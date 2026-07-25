from typing import Optional

from fastembed import SparseTextEmbedding

_SPARSE_MODEL: Optional[SparseTextEmbedding] = None


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy singleton for BM25 sparse embedding model."""
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        _SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _SPARSE_MODEL
