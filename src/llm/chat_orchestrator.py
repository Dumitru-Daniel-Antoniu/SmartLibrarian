from typing import Any, Dict, List
from openai import OpenAI

from src.config import settings
from src.retriever import semantic_search
from src.tools.summaries_tool import TOOLS, call_tool


"""
Chat orchestrator for book recommendations using OpenAI chat completions.
Performs semantic search on book summaries, formats context and interacts
with tools to generate friendly book recommendations.
"""


_client = OpenAI()


def _one_line(
    text: str,
    max_len: int = 160
) -> str:
    """
    Extract the first line from a text and truncate it to a maximum length.

    Args:
        text (str): Input text.
        max_len (int): Maximum length of the output line.

    Returns:
        str: Truncated first line of the input text.
    """
    line = text.splitlines()[0].strip() if text else ""
    return (line[: max_len - 1] + "...") if len(line) > max_len else line


def _format_rag_context(hits: List[Dict[str, Any]]) -> str:
    """
    Format a list of semantic search hits into a readable string for context.

    Args:
        hits (List[Dict[str, Any]]): List of search result dictionaries.

    Returns:
        str: Formatted string listing candidate books with snippets
             and distances.
    """
    lines = []
    for i, h in enumerate(hits, start=1):
        title = h.get("title", "").strip()
        snippet = _one_line(h.get("summary", ""))
        distances = h.get("distances", None)
        if distances is not None:
            lines.append(
                f"{i}) {title} - {snippet} (distance={distances:.3f})"
            )
        else:
            lines.append(
                f"{i}) {title} - {snippet}"
            )
    return "\n".join(lines)


def answer_user_query(user_text: str) -> Dict[str, Any]:
    """
    Generate a book recommendation based on user query using
    semantic search and OpenAI chat completions.

    Args:
        user_text (str): The user's query about books.

    Returns:
        Dict[str, Any]: Dictionary containing the final response
        text, search hits and the original query.
    """
    rag = semantic_search(user_text, k=settings.TOP_K)
    hits = rag.get("hits", [])

    if not hits:
        return {
            "final_text": (
                "I couldn't find a good match in the library. "
                "Try different keywords."
            ),
            "hits": [],
            "query": user_text
        }

    context_block = _format_rag_context(hits)

    system_message = (
        "You are a book assistant. Use only the candidate list.\n"
        "Steps:\n"
        "1) Pick ONE title (prefer lowest distance).\n"
        "2) Call tool `get_summary_by_title` with that exact title.\n"
        "3) After tool output, send ONE message:\n"
        "   - Start with the title in **bold** and a short "
        "friendly recommendation.\n"
        "   - Add 2–3 brief reasons why it fits.\n"
        "   - End with: 'Here’s a quick summary:' + the full summary.\n\n"
        "Rules:\n"
        "- Do NOT invent titles.\n"
        "- If user asks about a title and it is in candidates, pick it. "
        "The title must match exactly (e. g. The Lord of the Rings is not "
        "the same as The Lord of the Strings).\n"
        "- If request is not about books or no candidate matches, return no "
        "content and no tool call.\n"
        "- Never explain tool mechanics."
        "- The response should be natural, friendly, engaging "
        "and conversational."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_text},
        {
            "role": "system",
            "content": (
                "Candidate books (ranked by similarity):\n"
                f"{context_block}"
            )
        }
    ]

    result = _client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=float(settings.TEMPERATURE),
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    choice = result.choices[0]
    recommendation = choice.message
    tool_calls = getattr(recommendation, "tool_calls", None)
    content = (recommendation.content or "").strip()

    if not tool_calls and not content:
        return {
            "final_text": (
                "I had trouble generating a recommendation. "
                "Try rephrasing your interests."
            ),
            "hits": hits,
            "query": user_text
        }

    if not tool_calls and content:
        return {
            "final_text": content,
            "hits": hits,
            "query": user_text
        }

    tool_messages: List[Dict[str, str]] = []
    for tool in tool_calls or []:
        tool_name = tool.function.name
        tool_arguments_json = tool.function.arguments
        tool_response = call_tool(tool_name, tool_arguments_json)
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool.id,
            "content": tool_response
        })

    messages.append(recommendation)
    messages.extend(tool_messages)

    final_result = _client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=float(settings.TEMPERATURE),
        messages=messages
    )

    final_text = final_result.choices[0].message.content \
        if result.choices \
        else "Sorry, I had trouble generating a response."

    return {
        "final_text": final_text,
        "hits": hits,
        "query": user_text
    }


# If case for running the script directly for testing purposes
if __name__ == "__main__":
    demo_query = "What is the Lord of the Rings book about?"
    output = answer_user_query(demo_query)
    print("User:", demo_query)
    print("\nAssistant:\n", output["final_text"])
    print("\nTop candidates:")
    for i, h in enumerate(output["hits"], start=1):
        print(f"{i}) {h['title']} (distance={h['distance']:.3f})")
