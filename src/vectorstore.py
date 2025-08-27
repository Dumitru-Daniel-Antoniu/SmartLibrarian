from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple, Mapping
import chromadb
from chromadb.config import Settings

from pathlib import Path
from src.config import settings


"""
ChromaDB vectorstore utilities for SmartLibrarian.
Provides functions to manage collections, add/upsert/query documents, and handle embeddings for semantic search.
"""


def _abs_chroma_path() -> str:
    """
    Resolve the absolute path for the ChromaDB storage directory.

    Returns:
        str: Absolute path to the ChromaDB directory.
    """
    raw = Path(settings.CHROMA_PATH)
    if raw.is_absolute():
        return str(raw)
    root = Path(__file__).resolve().parents[1]
    return str((root / raw).resolve())


def get_client() -> chromadb.PersistentClient:
    """
    Create and return a ChromaDB persistent client using the configured path.

    Returns:
        chromadb.PersistentClient: ChromaDB client instance.
    """
    return chromadb.PersistentClient(
        path=_abs_chroma_path(),
        settings=Settings(allow_reset=True)
    )


def get_or_create_collection(
    name: Optional[str] = None,
    space: str = "cosine"
) -> chromadb.api.models.Collection.Collection:
    """
    Retrieve an existing ChromaDB collection or create a new one if it does not exist.

    Args:
        name (Optional[str]): Name of the collection.
        space (str): Distance metric for the collection. Defaults to "cosine".

    Returns:
        chromadb.api.models.Collection.Collection: The collection object.
    """
    client = get_client()
    collection_name = name or settings.COLLECTION_NAME

    try:
        return client.get_collection(collection_name)
    except Exception:
        print("Collection does not exist")
        return client.create_collection(
            name = collection_name,
            metadata = {"hnsw:space": space, "embed_model": settings.EMBED_MODEL}
        )


def recreate_collection(
    name: Optional[str] = None,
    space: str = "cosine"
) -> chromadb.api.models.Collection.Collection:
    """
    Delete and recreate a ChromaDB collection.

    Args:
        name (Optional[str]): Name of the collection.
        space (str): Distance metric for the collection. Defaults to "cosine".

    Returns:
        chromadb.api.models.Collection.Collection: The new collection object.
    """
    client = get_client()
    collection_name = name or settings.COLLECTION_NAME

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return client.create_collection(
        name = collection_name,
        metadata = {"hnsw:space": space, "embed_model": settings.EMBED_MODEL}
    )


def count_documents(
    collection: chromadb.api.models.Collection.Collection
) -> int:
    """
    Count the number of documents in a ChromaDB collection.

    Args:
        collection (chromadb.api.models.Collection.Collection): The collection to count documents in.

    Returns:
        int: Number of documents in the collection.
    """
    return collection.count()


def add_documents(
    collection: chromadb.api.models.Collection.Collection,
    documents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadata: Sequence[Dict[str, Any]],
    ids: Sequence[str]
) -> None:
    """
    Add new documents, embeddings, metadata and ids to a ChromaDB collection.

    Args:
        collection (chromadb.api.models.Collection.Collection): The target collection.
        documents (Sequence[str]): List of document texts.
        embeddings (Sequence[Sequence[float]]): List of embedding vectors.
        metadata (Sequence[Dict[str, Any]]): List of metadata dictionaries.
        ids (Sequence[str]): List of document IDs.

    Raises:
        ValueError: If input sequences are not of equal length.
    """
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
    """
    Upsert (add or update) documents, embeddings, metadata and ids in a ChromaDB collection.

    Args:
        collection (chromadb.api.models.Collection.Collection): The target collection.
        documents (Sequence[str]): List of document texts.
        embeddings (Sequence[Sequence[float]]): List of embedding vectors.
        metadata (Sequence[Dict[str, Any]]): List of metadata dictionaries.
        ids (Sequence[str]): List of document IDs.

    Raises:
        ValueError: If input sequences are not of equal length.
    """
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
    """
    Query a ChromaDB collection using one or more embedding vectors.

    Args:
        collection (chromadb.api.models.Collection.Collection): The collection to query.
        query_embeddings (Sequence[Sequence[float]]): List of query embedding vectors.
        n_results (int): Number of top results to return.

    Returns:
        Mapping[str, Any]: Query result containing documents, metadatas and distances.
    """
    return collection.query(
        query_embeddings=[list(e) for e in query_embeddings],
        n_results=int(n_results),
        include=["documents", "metadatas", "distances"]
    )


def _first_or_empty(
    result: Mapping[str, Any],
    key: str
) -> List[Any]:
    """
    Extract the first group of results for a given key, or return an empty list.

    Args:
        result (Mapping[str, Any]): Query result mapping.
        key (str): Key to extract from the result.

    Returns:
        List[Any]: First group of results or empty list.
    """
    groups = result.get(key) or []
    return list(groups[0]) if groups else []


def query_single(
    collection: chromadb.api.models.Collection.Collection,
    query_embedding: Sequence[float],
    n_results: int
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
    """
    Query a ChromaDB collection with a single embedding and return structured results.

    Args:
        collection (chromadb.api.models.Collection.Collection): The collection to query.
        query_embedding (Sequence[float]): Query embedding vector.
        n_results (int): Number of top results to return.

    Returns:
        Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
            Lists of IDs, documents, metadata, and distances for the top results.
    """
    result = query_by_embedding(collection, [list(query_embedding)], n_results)

    ids = _first_or_empty(result, "ids")
    documents = _first_or_empty(result, "documents")
    metadata = _first_or_empty(result, "metadatas")
    distances = _first_or_empty(result, "distances")

    m = min(len(ids), len(documents), len(metadata), len(distances))
    return ids[:m], documents[:m], metadata[:m], distances[:m]
