from unittest.mock import MagicMock

from bmo.llm.chat import SYSTEM_PROMPT, ChatSession


def _client(reply):
    """A fake Groq client whose completion returns `reply`."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=reply))]
    )
    return client


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
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    client.chat.completions.create.side_effect = _capture
    chat = ChatSession(client=client)

    chat.send("first")
    chat.send("second")

    # The second call must carry the prior turns, not just the new message.
    assert sent_roles[1] == ["system", "user", "assistant", "user"]
