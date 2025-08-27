from __future__ import annotations

import argparse
from typing import List

from src.config import settings
from src.vectorstore import get_or_create_collection, count_documents
from src.retriever import semantic_search


"""
Script to inspect the Chroma vector index for book summaries.
Allows previewing indexed titles and running semantic search
queries via command-line arguments.
"""


def print_header() -> None:
    """
    Print header information about the Chroma index and configuration settings.
    """
    print("=== SmartLibrarian: Inspect Index ===")
    print(f"Chroma path: {settings.CHROMA_PATH}")
    print(f"Collection name: {settings.COLLECTION_NAME}")
    print(f"Embedding model: {settings.EMBED_MODEL}")
    print(f"Top_k: {settings.TOP_K}")
    print("-" * 50)


def show_titles() -> None:
    """
    Display the book titles from the Chroma collection as a quick preview.
    """
    collection = get_or_create_collection()
    n = count_documents(collection)
    print(f"Indexed items: {n}")
    if n == 0:
        print("No items found.")
        return

    try:
        preview = semantic_search(" ", k=n)
        print("\nSample titles:")
        for i, h in enumerate(preview["hits"], start=1):
            print(f"  {i:>2}. {h['title']}  (distance={h['distance']:.4f})")
        print()
    except Exception as e:
        print(f"Could not fetch preview titles: {e}")


def run_queries(queries: List[str], k: int) -> None:
    """
    Run semantic search queries against the Chroma index and display results.

    Args:
        queries (List[str]): List of query strings to search for.
        k (int): Number of top results to return per query.
    """
    for q in queries:
        print(f"Query: '{q}'  (top {k})")
        try:
            result = semantic_search(q, k=k)
            if not result["hits"]:
                print("No results found")
            for i, h in enumerate(result["hits"], start=1):
                title = h["title"]
                distance = h["distance"]
                snippet = (
                              h["summary"].splitlines()[0]
                              if h["summary"] else ""
                          )[:120]
                print(f"{i:>2}. {title} [distance={distance:.4f}]")
                if snippet:
                    print(f"{snippet}...")
        except Exception as e:
            print(f"Error running query: {e}")
        print()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for inspecting the Chroma index.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Inspect Chroma index and run sample queries."
    )
    p.add_argument(
        "--show",
        type=int,
        default=5,
        help="Show up to N titles as a quick preview."
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Top-k results per query (overrides settings.TOP_K)."
    )
    p.add_argument(
        "--query",
        action="append",
        default=[],
        help="Add a query to run (repeatable)."
    )
    return p.parse_args()


def main() -> None:
    """
    Main entry point for inspecting the Chroma index.

    Parses arguments, prints header, previews titles
    and runs queries if provided.
    """
    args = parse_args()
    print_header()
    show_titles()

    k = args.k if args.k is not None else int(settings.TOP_K)
    if args.query:
        run_queries(args.query, k=k)


if __name__ == "__main__":
    main()
