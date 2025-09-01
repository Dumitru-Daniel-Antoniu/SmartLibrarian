import re

from typing import Any, Dict, List, Optional

from src.config import settings
from src.embeddings import embed_text
from src.vectorstore import get_or_create_collection, query_single


"""
Semantic search retriever for book summaries.
Embeds queries, searches the vectorstore, filters results by
distance and returns structured matches for downstream use.
"""


def _token_count(text: str) -> int:
    """
    Count the number of word tokens in the input text.

    Args:
        text (str): Input text to tokenize.

    Returns:
        int: Number of word tokens.
    """
    return len(re.findall(r"\w+", text.lower()))


def _pack_hits(
    ids: list[str],
    documents: List[str],
    metadata: List[Dict[str, Any]],
    distances: List[float]
) -> List[Dict[str, Any]]:
    """
    Package search results into a list of dictionaries with id,
    title, summary and distance.

    Args:
        ids (list[str]): List of document IDs.
        documents (List[str]): List of document summaries.
        metadata (List[Dict[str, Any]]): List of metadata
                                         dictionaries for each document.
        distances (List[float]): List of distance scores for each result.

    Returns:
        List[Dict[str, Any]]: List of structured search result dictionaries.
    """
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
    """
    Perform semantic search for a query against the book
    summaries vectorstore.

    Args:
        query (str): The search query string.
        k (Optional[int]): Number of top results to return.

    Returns:
        Dict[str, Any]: Dictionary containing the query, k,
                        filtered hits and raw result data.
    """
    top_k = int(k) if k is not None else int(settings.TOP_K)

    qv = embed_text(query)

    query_tokens = _token_count(query)

    max_distance = float(settings.MAX_DISTANCE)

    collection = get_or_create_collection()

    ids, documents, metadata, distances = query_single(
        collection,
        qv,
        n_results=top_k
    )

    if query_tokens <= 15:
        max_distance = 0.75
    elif query_tokens > 15 and query_tokens <= 30:
        max_distance = 0.65
    else:
        max_distance = 0.55

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


# If case for running the script directly for testing purposes
if __name__ == "__main__":
    test_query = "Friendship and magic"
    result = semantic_search(test_query, k=4)
    print(f"Query: {result['query']}")
    for i, h in enumerate(result["hits"], start=1):
        print(f"{i}. {h['title']}  (distance={h['distance']:.4f})")
