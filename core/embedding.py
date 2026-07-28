from typing import Optional, Any

_SPARSE_MODEL: Optional[Any] = None


def get_sparse_model() -> Any:
    raise DeprecationWarning("Fastembed sparse model has been removed. Weaviate handles BM25 natively.")
