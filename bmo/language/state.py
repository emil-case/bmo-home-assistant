from abc import ABC, abstractmethod

# The language-specific slice of the system prompt. The rest of the prompt
# (BMO's persona, the "keep it short / no markdown" rules) is fixed English and
# lives in bmo.llm.chat — only this clause is chosen per language.
REPLY_IN_ENGLISH = "Always reply in English."
REPLY_IN_SPANISH = "Always reply in Spanish."

# Whisper transcription language, forced so STT doesn't auto-detect (a non-native
# accent can otherwise be misheard as another language). ISO-639-1 codes.
STT_LANGUAGE_ENGLISH = "en"
STT_LANGUAGE_SPANISH = "es"


class LanguageState(ABC):
    """Template for a language BMO can operate in. Concretes: English / Spanish.

    A component that needs a language-dependent value asks its owner (BMO),
    which forwards the call to the current state. The component doesn't pass
    itself — the method it calls already names the value it wants, and BMO holds
    each component, so the answer depends on the method (which value) and the
    concrete state (which language). That pairing is still a double dispatch,
    resolved through per-instance fields rather than an argument.

    The per-language values are plain data, so they live as fields set by each
    concrete state's constructor and are served by the shared methods below.
    The abstract `__init__` owns the fields, so every subclass must supply them;
    adding another constant-valued property is one `__init__` param, not an
    override in every subclass. The values so far are the system-prompt
    reply-language clause (asked for by the ChatSession) and the Whisper STT
    language code (asked for by the Transcriber).
    """

    def __init__(self, reply_language: str, stt_language: str):
        self._reply_language = reply_language
        self._stt_language = stt_language

    @classmethod
    def default(cls) -> "LanguageState":
        """The language BMO boots in (English). Centralizes the 'first' state
        here so callers depend only on the abstract type, not a concrete
        subclass; changing the boot language is a one-line edit."""
        return EnglishState()

    def reply_language(self) -> str:
        """The 'Always reply in X' clause spliced into the system prompt."""
        return self._reply_language

    def stt_language(self) -> str:
        """The ISO-639-1 code Whisper is forced to transcribe in."""
        return self._stt_language

    @abstractmethod
    def nextLanguage(self) -> "LanguageState":
        """The next state in the language carousel. Each state names its own
        successor, so cycling through languages is `state = state.nextLanguage()`
        — no central if/else that has to grow with every new language."""


class EnglishState(LanguageState):
    def __init__(self):
        super().__init__(REPLY_IN_ENGLISH, STT_LANGUAGE_ENGLISH)

    def nextLanguage(self):
        return SpanishState()


class SpanishState(LanguageState):
    def __init__(self):
        super().__init__(REPLY_IN_SPANISH, STT_LANGUAGE_SPANISH)

    def nextLanguage(self):
        return EnglishState()
