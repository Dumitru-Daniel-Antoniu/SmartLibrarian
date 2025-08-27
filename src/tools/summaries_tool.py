from __future__ import annotations

import json
from typing import Dict, List, Optional
from functools import lru_cache

from src.data_loader import load_book_summaries


"""
Toolset for retrieving book summaries by title from the loaded library.
Provides functions for exact title matching, summary lookup, and tool
dispatching for integration with chat or API workflows.
"""


def _normalize_title(s: str) -> str:
    """
   Normalize a book title by stripping whitespace and converting to lowercase.

   Args:
       s (str): The book title to normalize.

   Returns:
       str: Normalized title string.
   """
    return (s or "").strip().lower()


@lru_cache(maxsize=1)
def _summary_map() -> Dict[str, str]:
    """
    Load and cache a mapping of normalized book titles to their summaries.

    Returns:
        Dict[str, str]: Dictionary mapping normalized titles to summaries.
    """
    records = load_book_summaries()
    return {_normalize_title(r["title"]): r["summary"] for r in records}


def _find_exact_title_key(title: str) -> Optional[str]:
    """
    Find the normalized key for an exact book title match in the summary map.

    Args:
        title (str): The book title to search for.

    Returns:
        Optional[str]: Normalized title key if found, else None.
    """
    key = _normalize_title(title)
    summary = _summary_map()
    return key if key in summary.keys() else None


def get_summary_by_title(title: str) -> str:
    """
    Retrieve the full summary for a given book title.

    Args:
        title (str): The exact book title to look up.

    Returns:
        str: The book summary, or an error message if not found.
    """
    key = _find_exact_title_key(title)
    if key is None:
        return (
            f"Sorry, I couldn't find a summary for the book titled '{title}'. "
            "Please check the title and try again."
        )
    return _summary_map()[key]


TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": (
                "Return the full summary for a single book title from the "
                "library. Use the EXACT title text from the provided "
                "candidate list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact book title to retrieve "
                                       "the full summary for."
                    }
                },
                "required": ["title"],
                "additionalProperties": False
            }
        }
    }
]


FUNCTIONS = {
    "get_summary_by_title": get_summary_by_title
}


def call_tool(tool_name: str, arguments_json: str) -> str:
    """
    Dispatch a tool function call by name with JSON arguments.

    Args:
        tool_name (str): Name of the tool function to call.
        arguments_json (str): JSON string of arguments for the tool.

    Returns:
        str: Result of the tool function call or an error message.
    """
    if tool_name not in FUNCTIONS:
        return f"Unknown tool: {tool_name}"

    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return "Invalid JSON for tool arguments."

    if tool_name == "get_summary_by_title":
        title = arguments.get("title", "")
        return get_summary_by_title(title)

    return f"No dispacher implemented for tool: {tool_name}"
