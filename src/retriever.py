import re

from typing import Any, Dict, List, Optional

from src.config import settings
from src.embeddings import embed_text
from src.vectorstore import get_or_create_collection, query_single


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+", text.lower()))


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

    query_tokens = _token_count(query)

    max_distance = float(settings.MAX_DISTANCE)

    collection = get_or_create_collection()

    ids, documents, metadata, distances = query_single(collection, qv, n_results=top_k)

    if query_tokens <= 2:
        max_distance = min(0.75, max_distance + 0.05)
    elif query_tokens >= 8:
        max_distance = max(0.50, max_distance - 0.05)

    if not distances or min(distances) > max_distance:
        return {
            "query": query,
            "k": top_k,
            "hits": []
        }

    filtered = [(i, d) for i, d in enumerate(distances) if d <= max_distance]
    if len(filtered) < int(settings.MIN_RESULTS):
        return {
            "query": query,
            "k": top_k,
            "hits": []
        }

    indices = [i for i, _ in sorted(filtered, key=lambda x: x[1])]
    ids = [ids[i] for i in indices]
    documents = [documents[i] for i in indices]
    metadata = [metadata[i] for i in indices]
    distances = [distances[i] for i in indices]

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

