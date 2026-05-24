from unittest.mock import MagicMock

from bmo.llm.chat import SYSTEM_PROMPT, ChatSession


def _text_response(reply):
    """Fake Groq response object for a plain-text (no tool call) reply."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=reply, tool_calls=None))]
    )


def _client(reply):
    """A fake Groq client whose completion returns `reply` as plain text."""
    client = MagicMock()
    client.chat.completions.create.return_value = _text_response(reply)
    return client


def _make_tool_call(name, arguments, call_id="call_1"):
    # MagicMock's `name` kwarg sets the mock's own repr name, not `.name` —
    # so the function attributes are assigned explicitly after construction.
    tc = MagicMock(id=call_id)
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _tool_call_response(name, arguments, call_id="call_1"):
    return MagicMock(
        choices=[MagicMock(
            message=MagicMock(content=None, tool_calls=[_make_tool_call(name, arguments, call_id)])
        )]
    )


def test_given_new_session_when_created_then_history_starts_with_system_prompt():
    chat = ChatSession(client=_client("hi"))
    assert chat._messages == [{"role": "system", "content": SYSTEM_PROMPT}]


def test_given_user_message_when_sent_then_appends_user_and_assistant_turns_stripped():
    chat = ChatSession(client=_client("  Hi there!  "))

    reply = chat.send("hello")

    assert reply == "Hi there!"
    assert chat._messages[1] == {"role": "user", "content": "hello"}
    assert chat._messages[2] == {"role": "assistant", "content": "Hi there!"}


def test_given_prior_turns_when_sending_again_then_full_history_is_sent():
    # Snapshot the roles at call time: ChatSession passes its live messages list
    # by reference, so inspecting call_args afterwards would show later mutations.
    sent_roles = []
    client = MagicMock()

    def _capture(*_, messages, **__):
        sent_roles.append([m["role"] for m in messages])
        return _text_response("ok")

    client.chat.completions.create.side_effect = _capture
    chat = ChatSession(client=client)

    chat.send("first")
    chat.send("second")

    # The second call must carry the prior turns, not just the new message.
    assert sent_roles[1] == ["system", "user", "assistant", "user"]


def test_given_tool_call_when_sent_then_runs_tool_and_returns_followup_text():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("web_search", '{"query": "weather"}'),
        _text_response("It's sunny."),
    ]
    tool = MagicMock(return_value="sunny, 22C")
    chat = ChatSession(client=client, tools={"web_search": tool}, tool_specs=[])

    reply = chat.send("what's the weather?")

    assert reply == "It's sunny."
    tool.assert_called_once_with(query="weather")
    # History should contain: system, user, assistant(tool_call), tool result, assistant(final).
    roles = [m["role"] for m in chat._messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = chat._messages[3]
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "sunny, 22C"
    assert chat._messages[-1] == {"role": "assistant", "content": "It's sunny."}


def test_given_unknown_tool_when_called_then_error_is_fed_back_not_raised():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("nope", "{}"),
        _text_response("Sorry, I can't do that."),
    ]
    chat = ChatSession(client=client, tools={}, tool_specs=[])

    reply = chat.send("do the thing")

    assert reply == "Sorry, I can't do that."
    tool_msg = chat._messages[3]
    assert tool_msg["role"] == "tool"
    assert "unknown tool" in tool_msg["content"]


def test_given_tool_raises_when_called_then_error_is_fed_back_not_raised():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("web_search", '{"query": "x"}'),
        _text_response("Something broke, sorry."),
    ]

    def _boom(**_):
        raise RuntimeError("brave is down")

    chat = ChatSession(client=client, tools={"web_search": _boom}, tool_specs=[])

    reply = chat.send("search")

    assert reply == "Something broke, sorry."
    tool_msg = chat._messages[3]
    assert tool_msg["role"] == "tool"
    assert "brave is down" in tool_msg["content"]
