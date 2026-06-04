import pytest

from bmo.language.state import (
    REPLY_IN_ENGLISH,
    REPLY_IN_SPANISH,
    EnglishState,
    LanguageState,
    SpanishState,
)


def test_given_english_state_when_asked_then_returns_english_reply_clause():
    assert EnglishState().reply_language(None) == REPLY_IN_ENGLISH


def test_given_spanish_state_when_asked_then_returns_spanish_reply_clause():
    assert SpanishState().reply_language(None) == REPLY_IN_SPANISH


def test_given_abstract_template_when_instantiated_then_raises():
    # LanguageState is a template (ABC): the abstract method must be implemented.
    with pytest.raises(TypeError):
        LanguageState()
