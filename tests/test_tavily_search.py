from unittest.mock import MagicMock, patch

import pytest

from bmo.tools import tavily_search


def _response(payload, status=200):
    r = MagicMock()
    r.json.return_value = payload
    r.status_code = status
    r.raise_for_status = MagicMock()
    return r


def test_given_answer_and_results_when_searching_then_returns_compact_summary(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    payload = {
        "answer": "It is sunny and 22C.",
        "results": [
            {"title": "Weather", "content": "Sunny today.", "url": "https://w.example/x"},
            {"title": "Forecast", "content": "Clear skies.", "url": "https://w.example/y"},
        ],
    }

    with patch.object(tavily_search.requests, "post", return_value=_response(payload)) as post:
        out = tavily_search.search("weather today")

    assert "Answer: It is sunny and 22C." in out
    assert "Weather" in out and "Sunny today." in out and "https://w.example/x" in out
    body = post.call_args.kwargs["json"]
    assert body["query"] == "weather today"
    assert body["api_key"] == "secret"
    assert body["include_answer"] is True


def test_given_only_results_when_searching_then_omits_answer_line(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    payload = {"results": [{"title": "r1", "content": "c1", "url": "https://e/1"}]}

    with patch.object(tavily_search.requests, "post", return_value=_response(payload)):
        out = tavily_search.search("x")

    assert "Answer:" not in out
    assert "1. r1" in out


def test_given_empty_response_when_searching_then_returns_no_results_string(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    with patch.object(tavily_search.requests, "post", return_value=_response({"results": []})):
        assert tavily_search.search("nothing") == "No results."


def test_given_count_when_searching_then_limits_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    payload = {"results": [
        {"title": f"r{i}", "content": "d", "url": f"https://e/{i}"} for i in range(5)
    ]}

    with patch.object(tavily_search.requests, "post", return_value=_response(payload)) as post:
        out = tavily_search.search("x", count=2)

    assert "1. r0" in out and "2. r1" in out and "3." not in out
    assert post.call_args.kwargs["json"]["max_results"] == 2


def test_given_missing_api_key_when_searching_then_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(KeyError):
        tavily_search.search("anything")
