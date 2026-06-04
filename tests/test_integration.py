"""Integration tests: real BMO with real collaborators.

Unlike the unit tests (which mock each component to isolate one piece), these
build a real BMO with a real ChatSession and a real LanguageState — only the
hardware/API *boundaries* are mocked (mic, wake word, STT, TTS, the Groq
client). That lets the actual delegation chain run end to end instead of being
stubbed out.
"""

from unittest.mock import patch

from bmo.bmo import BMO


# Switching language must change the system prompt the chat will send, walking
# the whole chain for real: ChatSession._build_system_message -> BMO.reply_language()
# -> LanguageState.reply_language(). No mock stands in for the forwarding here —
# the unit tests mock ChatSession (so BMO->state is faked) or the owner (so
# ChatSession->BMO is faked); this is the only place all three are real.
@patch("bmo.llm.chat.Groq")     # real ChatSession, just no real Groq client
@patch("bmo.bmo.Speaker")
@patch("bmo.bmo.Transcriber")
@patch("bmo.bmo.WakeWordDetector")
@patch("bmo.bmo.AudioCapture")
def test_given_real_bmo_when_language_switched_then_chat_system_prompt_follows(*_):
    bmo = BMO()
    # System prompt is built through the real chain at construction (English).
    assert "English" in bmo._chat._messages[0]["content"]

    bmo.switch_language()  # flips the state and resets the chat
    # Reset rebuilds the system message, which re-walks the chain to the new state.
    assert "Spanish" in bmo._chat._messages[0]["content"]
