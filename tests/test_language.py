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


def test_given_abstract_template_when_default_requested_then_returns_english():
    # The boot language is defined on the abstract type, so callers need only
    # depend on LanguageState — not on any concrete subclass.
    assert isinstance(LanguageState.default(), EnglishState)


def test_given_english_state_when_advanced_then_next_is_spanish():
    assert isinstance(EnglishState().nextLanguage(), SpanishState)


def test_given_spanish_state_when_advanced_then_next_is_english():
    assert isinstance(SpanishState().nextLanguage(), EnglishState)


def test_given_carousel_when_advanced_twice_then_returns_to_english():
    # Two languages today, so the carousel is a toggle; advancing twice is a
    # round trip. A third language would just lengthen the cycle, not break this.
    state = EnglishState()
    assert isinstance(state.nextLanguage().nextLanguage(), EnglishState)


def test_given_abstract_template_when_instantiated_then_raises():
    # LanguageState is a template (ABC): the abstract methods must be implemented.
    with pytest.raises(TypeError):
        LanguageState()
