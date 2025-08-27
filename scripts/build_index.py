import sys
from typing import List, Dict

from src.config import settings, print_config_summary, require_openai_key
from src.data_loader import load_book_summaries
from src.embeddings import embed_texts
from src.vectorstore import (
    recreate_collection,
    add_documents,
    count_documents,
    get_or_create_collection
)


"""
Script to build a vector index of book summaries using OpenAI embeddings and Chroma vectorstore.
Loads book summaries, embeds them, and stores them in a Chroma collection for later retrieval.
"""


def _make_ids(n: int) -> List[str]:
    """
    Generate a list of unique string IDs for books in the format 'book-xxx'.

    Args:
        n (int): Number of IDs to generate.

    Returns:
        List[str]: List of generated IDs.
    """
    return [f"book-{i:03d}" for i in range(n)]


def _make_payload(records: List[Dict[str, str]]):
    """
    Prepare documents, metadata, and IDs from book summary records.

    Args:
        records (List[Dict[str, str]]): List of book summary records.

    Returns:
        Tuple[List[str], List[Dict[str, str]], List[str]]: Documents, metadata, and IDs.
    """
    documents = [r["summary"] for r in records]
    metadata = [{"title": r["title"]} for r in records]
    ids = _make_ids(len(records))
    return documents, metadata, ids


def main() -> None:
    """
    Main entry point for building the book summary index.

    Loads data, embeds summaries, creates/recreates the Chroma collection,
    adds documents, and prints a summary of the indexing process.
    """
    print("=== SmartLibrarian: Build Index ===")
    require_openai_key()

    print_config_summary()

    records = load_book_summaries()
    if not records:
        raise RuntimeError("No records parsed from dataset.")

    print(f"Loaded {len(records)} records from dataset.")

    documents, metadata, ids = _make_payload(records)

    print(f"Embedding {len(documents)} summaries with model '{settings.EMBED_MODEL}'...")
    vectors = embed_texts(documents, batch_size=64)
    print("Embeddings done.")

    print(f"Recreating collection '{settings.COLLECTION_NAME}' at {settings.CHROMA_PATH}")
    collection = recreate_collection(name=settings.COLLECTION_NAME, space="cosine")

    print("Adding documents to Chroma collection...")
    add_documents(collection, documents=documents, embeddings=vectors, metadata=metadata, ids=ids)

    collection = get_or_create_collection(settings.COLLECTION_NAME)
    n = count_documents(collection)
    print(f"Indexed items: {n}")

    preview = min(5, len(records))
    print("Sample titles:")
    for i in range(preview):
        print(f"  - {metadata[i]['title']}  (id={ids[i]})")

    print("=== Build complete ===")

if __name__ == "__main__":
    sys.exit(main())
