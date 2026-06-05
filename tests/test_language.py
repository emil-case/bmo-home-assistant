import pytest

from bmo.language.state import (
    EnglishState,
    LanguageState,
    SpanishState,
)


def _concrete_states(cls=LanguageState):
    """Every instantiable LanguageState subclass (recursively), skipping any
    still-abstract intermediates. New languages are picked up automatically."""
    found = []
    for sub in cls.__subclasses__():
        if not getattr(sub, "__abstractmethods__", None):
            found.append(sub)
        found.extend(_concrete_states(sub))
    return found


def test_given_english_state_when_asked_then_returns_english_reply_clause():
    assert EnglishState().reply_language() == "Always reply in English."


def test_given_spanish_state_when_asked_then_returns_spanish_reply_clause():
    assert SpanishState().reply_language() == "Always reply in Spanish."


def test_given_english_state_when_asked_then_returns_english_stt_code():
    assert EnglishState().stt_language() == "en"


def test_given_spanish_state_when_asked_then_returns_spanish_stt_code():
    assert SpanishState().stt_language() == "es"


def test_given_english_state_when_asked_then_returns_english_tts_voice():
    assert EnglishState().tts_voice() == "en"


def test_given_spanish_state_when_asked_then_returns_spanish_tts_voice():
    assert SpanishState().tts_voice() == "es"


def test_given_concrete_state_when_constant_hook_called_on_class_then_returns_constant():
    # The hooks are classmethods, so they answer without an instance — the
    # shared instance methods above just delegate to them. Asserted against the
    # literals (not a shared constant), so a wrong value can't pass its own test.
    assert EnglishState.reply_language_constant() == "Always reply in English."
    assert EnglishState.stt_language_constant() == "en"
    assert EnglishState.tts_voice_constant() == "en"
    assert SpanishState.reply_language_constant() == "Always reply in Spanish."
    assert SpanishState.stt_language_constant() == "es"
    assert SpanishState.tts_voice_constant() == "es"


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
    # LanguageState is a template (ABC): nextLanguage() is abstract and must be
    # implemented.
    with pytest.raises(TypeError):
        LanguageState()


def test_given_at_least_two_concrete_states_exist():
    # Guard the guard: if discovery found nothing, the parametrized contract test
    # below would vacuously pass. We ship English + Spanish, so expect >= 2.
    assert len(_concrete_states()) >= 2


@pytest.mark.parametrize("state_cls", _concrete_states(), ids=lambda c: c.__name__)
def test_given_concrete_state_when_inspected_then_supplies_all_required_constants(state_cls):
    # Every shipped language must define each required constant as a non-empty
    # string, and its accessor must return that value. __init_subclass__ already
    # rejects a *missing* constant at import; this also pins them to usable values
    # (e.g. an empty STT code or voice tag would break Whisper / voice matching).
    state = state_cls()
    for name in LanguageState._REQUIRED_CONSTANTS:
        value = getattr(state_cls, name)
        assert isinstance(value, str) and value, f"{state_cls.__name__}.{name} must be a non-empty string"

    assert state.reply_language() == state_cls._REPLY_CLAUSE
    assert state.stt_language() == state_cls._STT_CODE
    assert state.tts_voice() == state_cls._TTS_VOICE


def test_given_subclass_missing_a_constant_when_defined_then_raises():
    # The per-language values are plain class attributes, so ABC can't force
    # them; __init_subclass__ does — defining a concrete state without all three
    # fails at class-definition time, not later with an AttributeError.
    with pytest.raises(TypeError):
        class BrokenState(LanguageState):
            _REPLY_CLAUSE = "Always reply in Broken."
            _STT_CODE = "br"
            # _TTS_VOICE deliberately omitted

            def nextLanguage(self):
                return EnglishState()
