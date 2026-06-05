from abc import ABC, abstractmethod


class LanguageState(ABC):
    """Template for a language BMO can operate in. Concretes: English / Spanish.

    A component that needs a language-dependent value asks its owner (BMO),
    which forwards the call to the current state. The component doesn't pass
    itself — the method it calls already names the value it wants, and BMO holds
    each component, so the answer depends on the method (which value) and the
    concrete state (which language). That pairing is the double dispatch.

    The value accessors are **template methods**: the shared instance method
    (`reply_language` / `stt_language` / `tts_voice`) lives here once and
    delegates to a classmethod hook (`reply_language_constant` / …). The hook is
    concrete and also lives here once — it just reads a per-language class
    attribute (`_REPLY_CLAUSE` / `_STT_CODE` / `_TTS_VOICE`) that the concrete
    state supplies. So the per-language data sits on the subclass as a plain
    constant (no `__init__` carrying it, no accessor boilerplate repeated per
    subclass) and the lookup logic isn't duplicated. Because those constants are
    plain attributes rather than abstract methods, ABC can't enforce them, so
    `__init_subclass__` does — a concrete state missing one fails at
    class-definition time instead of with a late `AttributeError`. The values so
    far are the system-prompt reply-language clause (asked for by the
    ChatSession), the Whisper STT language code (asked for by the Transcriber),
    and the TTS voice tag (asked for by the Speaker).
    """

    # Every concrete state must supply these; checked in __init_subclass__ since
    # they're plain class attributes (no @abstractmethod to enforce them).
    _REQUIRED_CONSTANTS = ("_REPLY_CLAUSE", "_STT_CODE", "_TTS_VOICE")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [name for name in cls._REQUIRED_CONSTANTS if not hasattr(cls, name)]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define {', '.join(missing)}"
            )

    @classmethod
    def default(cls) -> "LanguageState":
        """The language BMO boots in (English). Centralizes the 'first' state
        here so callers depend only on the abstract type, not a concrete
        subclass; changing the boot language is a one-line edit."""
        return EnglishState()

    @classmethod
    def reply_language_constant(cls) -> str:
        """Hook: this language's 'Always reply in X' clause."""
        return cls._REPLY_CLAUSE

    @classmethod
    def stt_language_constant(cls) -> str:
        """Hook: this language's ISO-639-1 Whisper code."""
        return cls._STT_CODE

    @classmethod
    def tts_voice_constant(cls) -> str:
        """Hook: this language's TTS voice tag (the Piper .onnx filename prefix)."""
        return cls._TTS_VOICE

    def reply_language(self) -> str:
        """The 'Always reply in X' clause spliced into the system prompt."""
        return self.__class__.reply_language_constant()

    def stt_language(self) -> str:
        """The ISO-639-1 code Whisper is forced to transcribe in."""
        return self.__class__.stt_language_constant()

    def tts_voice(self) -> str:
        """The tag the Speaker matches against the installed Piper voices to pick
        which .onnx to speak this language in."""
        return self.__class__.tts_voice_constant()

    @abstractmethod
    def nextLanguage(self) -> "LanguageState":
        """The next state in the language carousel. Each state names its own
        successor, so cycling through languages is `state = state.nextLanguage()`
        — no central if/else that has to grow with every new language."""


class EnglishState(LanguageState):
    # English's own data — private to this class. The reply clause is the
    # language-specific slice of the system prompt (the rest is fixed English and
    # lives in bmo.llm.chat); the STT code is the ISO-639-1 language Whisper is
    # forced to transcribe in, so it doesn't mishear an accent as another language.
    _REPLY_CLAUSE = "Always reply in English."
    _STT_CODE = "en"
    # The Piper voice tag: BMO's Speaker globs resources/voices/ for an .onnx
    # whose filename starts with this (Piper names voices `en_US-amy-medium.onnx`).
    # It coincides with the STT code today but is conceptually a separate value —
    # one is a Whisper API parameter, the other selects a TTS model file.
    _TTS_VOICE = "en"

    def nextLanguage(self):
        return SpanishState()


class SpanishState(LanguageState):
    # Spanish's own data — private to this class (see EnglishState for what these are).
    _REPLY_CLAUSE = "Always reply in Spanish."
    _STT_CODE = "es"
    _TTS_VOICE = "es"

    def nextLanguage(self):
        return EnglishState()
