from typing import Iterable, List, Sequence, Optional
import time

from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError
from src.config import settings


"""
Embedding utilities for text data using OpenAI models.
Provides functions to batch embed multiple texts or single texts, with retry and exponential backoff for API robustness.
"""


_client = OpenAI()


def embed_texts(
    texts: Sequence[str],
    model: Optional[str] = None,
    batch_size: int = 64,
    max_retries: int = 3,
    backoff_seconds: float = 1.5
) -> List[List[float]]:
    """
    Generate embeddings for a sequence of texts using the specified OpenAI model.

    Args:
        texts (Sequence[str]): List of texts to embed.
        model (Optional[str]): Model name to use for embedding. Defaults to settings.EMBED_MODEL.
        batch_size (int): Number of texts per API request batch.
        max_retries (int): Maximum number of retry attempts on API errors.
        backoff_seconds (float): Base seconds for exponential backoff between retries.

    Returns:
        List[List[float]]: List of embedding vectors for each input text.

    Raises:
        ValueError: If texts is empty.
        RuntimeError: If embedding fails after max_retries attempts.
    """
    if not texts:
        raise ValueError("The function embed_texts() received an empty list of texts.")

    use_model = model or settings.EMBED_MODEL

    results: List[List[float]] = [None] * len(texts)

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]

        attempt = 0
        while True:
            try:
                response = _client.embeddings.create(model=use_model, input=list(chunk))
                for i, item in enumerate(response.data):
                    results[start + i] = item.embedding
                break
            except (RateLimitError, APITimeoutError, OpenAIError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Embedding request failed after {max_retries} attempts: {e}"
                    )
                sleep_time = backoff_seconds ** attempt
                time.sleep(sleep_time)

    return results


def embed_text(
    text: str,
    model: Optional[str] = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.5
) -> List[float]:
    """
    Generate an embedding for a single text using the specified OpenAI model.

    Args:
        text (str): The text to embed.
        model (Optional[str]): Model name to use for embedding. Defaults to settings.EMBED_MODEL.
        max_retries (int): Maximum number of retry attempts on API errors.
        backoff_seconds (float): Base seconds for exponential backoff between retries.

    Returns:
        List[float]: Embedding vector for the input text.

    Raises:
        RuntimeError: If embedding fails after max_retries attempts.
    """
    return embed_texts(
        [text],
        model = model,
        batch_size = 1,
        max_retries = max_retries,
        backoff_seconds = backoff_seconds
    )[0]


# If case for running the script directly for testing purposes
if __name__ == "__main__":
    sample_texts = [
        "This is the first text to embed.",
        "Here is another text for embedding.",
        "And this is the third one."
    ]

    embeddings = embed_texts(sample_texts)
    for i, text in enumerate(sample_texts):
        print(f"Text: {text}\nEmbedding: {embeddings[i]}\nLength: {len(embeddings[i])}\n")
