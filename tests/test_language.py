import pytest

from bmo.language.state import (
    EnglishState,
    LanguageState,
    SpanishState,
)


def test_given_english_state_when_asked_then_returns_english_reply_clause():
    assert EnglishState().reply_language() == "Always reply in English."


def test_given_spanish_state_when_asked_then_returns_spanish_reply_clause():
    assert SpanishState().reply_language() == "Always reply in Spanish."


def test_given_english_state_when_asked_then_returns_english_stt_code():
    assert EnglishState().stt_language() == "en"


def test_given_spanish_state_when_asked_then_returns_spanish_stt_code():
    assert SpanishState().stt_language() == "es"


def test_given_concrete_state_when_constant_hook_called_on_class_then_returns_constant():
    # The hooks are classmethods, so they answer without an instance — the
    # shared instance methods above just delegate to them. Asserted against the
    # literals (not a shared constant), so a wrong value can't pass its own test.
    assert EnglishState.reply_language_constant() == "Always reply in English."
    assert EnglishState.stt_language_constant() == "en"
    assert SpanishState.reply_language_constant() == "Always reply in Spanish."
    assert SpanishState.stt_language_constant() == "es"


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
    # LanguageState is a template (ABC): the constant hooks + nextLanguage()
    # are abstract and must be implemented.
    with pytest.raises(TypeError):
        LanguageState()
