from abc import ABC, abstractmethod


class LanguageState(ABC):
    """Template for a language BMO can operate in. Concretes: English / Spanish.

    A component that needs a language-dependent value asks its owner (BMO),
    which forwards the call to the current state. The component doesn't pass
    itself — the method it calls already names the value it wants, and BMO holds
    each component, so the answer depends on the method (which value) and the
    concrete state (which language). That pairing is the double dispatch.

    The value accessors are **template methods**: the shared instance method
    (`reply_language` / `stt_language`) lives here once and delegates to an
    abstract *classmethod* hook (`reply_language_constant` / `stt_language_constant`)
    that each concrete state fills with its constant. So the per-language data
    sits on the subclass (no `__init__` carrying it), the lookup logic isn't
    duplicated per subclass, and ABC still forces every subclass to supply each
    constant. The values so far are the system-prompt reply-language clause
    (asked for by the ChatSession) and the Whisper STT language code (asked for
    by the Transcriber).
    """

    @classmethod
    def default(cls) -> "LanguageState":
        """The language BMO boots in (English). Centralizes the 'first' state
        here so callers depend only on the abstract type, not a concrete
        subclass; changing the boot language is a one-line edit."""
        return EnglishState()

    @classmethod
    @abstractmethod
    def reply_language_constant(cls) -> str:
        """Hook: this language's 'Always reply in X' clause."""

    @classmethod
    @abstractmethod
    def stt_language_constant(cls) -> str:
        """Hook: this language's ISO-639-1 Whisper code."""

    def reply_language(self) -> str:
        """The 'Always reply in X' clause spliced into the system prompt."""
        return self.__class__.reply_language_constant()

    def stt_language(self) -> str:
        """The ISO-639-1 code Whisper is forced to transcribe in."""
        return self.__class__.stt_language_constant()

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

    @classmethod
    def reply_language_constant(cls):
        return cls._REPLY_CLAUSE

    @classmethod
    def stt_language_constant(cls):
        return cls._STT_CODE

    def nextLanguage(self):
        return SpanishState()


class SpanishState(LanguageState):
    # Spanish's own data — private to this class (see EnglishState for what these are).
    _REPLY_CLAUSE = "Always reply in Spanish."
    _STT_CODE = "es"

    @classmethod
    def reply_language_constant(cls):
        return cls._REPLY_CLAUSE

    @classmethod
    def stt_language_constant(cls):
        return cls._STT_CODE

    def nextLanguage(self):
        return EnglishState()
