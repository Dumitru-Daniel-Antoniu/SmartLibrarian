from typing import Any, Dict, List, Optional

from src.config import settings
from src.embeddings import embed_text
from src.vectorstore import get_or_create_collection, query_single


def _pack_hits(
    ids: list[str],
    documents: List[str],
    metadata: List[Dict[str, Any]],
    distances: List[float]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        title = metadata[i].get("title", "")
        out.append(
            {
                "id": ids[i],
                "title": title,
                "summary": documents[i],
                "distance": distances[i]
            }
        )
    return out


def semantic_search(
    query: str,
    k: Optional[int] = None
) -> Dict[str, Any]:
    top_k = int(k) if k is not None else int(settings.TOP_K)

    qv = embed_text(query)

    collection = get_or_create_collection()

    meta = getattr(collection, "metadatas", {}) or {}
    build_with = meta.get("embed_model")
    if build_with and build_with != settings.EMBED_MODEL:
        raise RuntimeError("Not the same embedding model used")
    else:
        print("Equal")

    ids, documents, metadata, distances = query_single(collection, qv, n_results=top_k)

    hits = _pack_hits(ids, documents, metadata, distances)

    return {
        "query": query,
        "k": top_k,
        "hits": hits,
        "raw": {
            "ids": ids,
            "documents": documents,
            "metadatas": metadata,
            "distances": distances
        }
    }


if __name__ == "__main__":
    test_query = "Friendship and magic"
    result = semantic_search(test_query, k=4)
    print(f"Query: {result['query']}")
    for i, h in enumerate(result["hits"], start=1):
        print(f"{i}. {h['title']}  (distance={h['distance']:.4f})")

