"""Tavily web search tool exposed to the LLM.

Tavily is designed for LLM tool-use: with `include_answer=True` it returns a
short, pre-summarized answer alongside the raw results. We hand the LLM that
answer (plus a few results as backup) so BMO has something TTS-friendly to
read aloud without needing to summarize a wall of HTML.
"""
import os

import requests

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_RESULT_COUNT = 3
REQUEST_TIMEOUT = 15  # seconds; Tavily can take a few seconds with answer generation


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information (news, facts, weather, etc.). "
            "Use this when the user asks about something you don't already know "
            "or that may have changed recently."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise search query, like one a person would type.",
                },
            },
            "required": ["query"],
        },
    },
}


def search(query: str, count: int = DEFAULT_RESULT_COUNT) -> str:
    """Run a Tavily web search and return a plain-text summary for the LLM."""
    api_key = os.environ["TAVILY_API_KEY"]
    response = requests.post(
        TAVILY_ENDPOINT,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": count,
            "include_answer": True,
            "search_depth": "basic",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    answer = (data.get("answer") or "").strip()
    results = data.get("results", [])
    if not answer and not results:
        return "No results."

    parts = []
    if answer:
        parts.append(f"Answer: {answer}")
    for i, r in enumerate(results[:count], start=1):
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        url = (r.get("url") or "").strip()
        parts.append(f"{i}. {title} — {content} ({url})")
    return "\n".join(parts)
