import re
from pathlib import Path


"""
Data loader for book summaries.
Reads and parses a text dataset of book summaries,
returning structured records for downstream processing.
"""


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "book_summaries.txt"


def load_book_summaries() -> list[dict[str, str]]:
    """
    Load and parse book summaries from the dataset file.

    Returns:
        list[dict[str, str]]: List of dictionaries, each containing
        'title' and 'summary' keys.
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as dataset_file:
        text = dataset_file.read()

        chunks = [
            c.strip() for c in
            re.split(r"^## Title:\s*", text, flags=re.MULTILINE)
            if c.strip()
        ]

        records = []
        for c in chunks:
            first_line, *rest = c.splitlines()
            title = first_line.strip()
            summary = "\n".join(line.strip() for line in rest if line.strip())
            if not title or not summary:
                continue
            records.append({"title": title, "summary": summary})
        return records


# If case for running the script directly for testing purposes
if __name__ == "__main__":
    summaries = load_book_summaries()
    for summary in summaries[:5]:
        print(f"Title: {summary['title']}\nSummary: {summary['summary']}\n")
