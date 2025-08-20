from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple, Mapping
import chromadb
from chromadb.config import Settings

from pathlib import Path
from src.config import settings


def _abs_chroma_path() -> str:
    raw = Path(settings.CHROMA_PATH)
    if raw.is_absolute():
        return str(raw)
    root = Path(__file__).resolve().parents[1]
    return str((root / raw).resolve())


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=_abs_chroma_path(),
        settings=Settings(allow_reset=True)
    )


def get_or_create_collection(
    name: Optional[str] = None,
    space: str = "cosine"
) -> chromadb.api.models.Collection.Collection:
    client = get_client()
    collection_name = name or settings.COLLECTION_NAME

    try:
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(
            name = collection_name,
            metadata = {"hnsw:space": space}
        )


def recreate_collection(
    name: Optional[str] = None,
    space: str = "cosine"
) -> chromadb.api.models.Collection.Collection:
    client = get_client()
    collection_name = name or settings.COLLECTION_NAME

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return client.create_collection(
        name = collection_name,
        metadata = {"hnsw:space": space}
    )


def count_documents(
    collection: chromadb.api.models.Collection.Collection
) -> int:
    return collection.count()


def add_documents(
    collection: chromadb.api.models.Collection.Collection,
    documents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadata: Sequence[Dict[str, Any]],
    ids: Sequence[str]
) -> None:
    n = len(documents)
    if not (len(embeddings) == len(metadata) == len(ids) == n):
        raise ValueError(
            "Documents, embeddings, metadata and ids must have the same length."
        )

    collection.add(
        documents=list(documents),
        embeddings=list(embeddings),
        metadatas=list(metadata),
        ids=list(ids)
    )


def upsert_documents(
    collection: chromadb.api.models.Collection.Collection,
    documents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadata: Sequence[Dict[str, Any]],
    ids: Sequence[str]
) -> None:
    n = len(documents)
    if not (len(embeddings) == len(metadata) == len(ids) == n):
        raise ValueError(
            "Documents, embeddings, metadata and ids must have the same length."
        )

    collection.upsert(
        documents=list(documents),
        embeddings=list(embeddings),
        metadatas=list(metadata),
        ids=list(ids)
    )


def query_by_embedding(
    collection: chromadb.api.models.Collection.Collection,
    query_embeddings: Sequence[Sequence[float]],
    n_results: int
) -> Mapping[str, Any]:
    return collection.query(
        query_embeddings=list(query_embeddings),
        n_results=int(n_results)
    )


def _first_or_empty(
    result: Mapping[str, Any],
    key: str
) -> List[Any]:
    groups = result.get(key) or []
    return groups[0] if groups else []


def query_single(
    collection: chromadb.api.models.Collection.Collection,
    query_embedding: Sequence[float],
    n_results: int
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
    result = query_by_embedding(collection, [list(query_embedding)], n_results)

    ids = _first_or_empty(result, "ids")
    documents = _first_or_empty(result, "documents")
    metadata = _first_or_empty(result, "metadata")
    distances = _first_or_empty(result, "distances")

    ids = list(ids or [])
    documents = list(documents or [])
    metadata = list(metadata or [])
    distances = list(distances or [])

    m = min(len(ids), len(documents), len(metadata), len(distances))
    return ids[:m], documents[:m], metadata[:m], distances[:m]
