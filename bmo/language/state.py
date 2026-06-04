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
    which forwards the call to the current state — passing the asking component.
    There's one method per component, so the answer depends on both the
    component (which method is called) and the concrete state (which subclass
    answers): that pairing is the double dispatch. The values so far are the
    system-prompt reply-language clause (asked for by the ChatSession) and the
    Whisper STT language code (asked for by the Transcriber).
    """

    @classmethod
    def default(cls) -> "LanguageState":
        """The language BMO boots in (English). Centralizes the 'first' state
        here so callers depend only on the abstract type, not a concrete
        subclass; changing the boot language is a one-line edit."""
        return EnglishState()

    @abstractmethod
    def reply_language(self, chat) -> str:
        """The 'Always reply in X' clause spliced into the system prompt."""

    @abstractmethod
    def stt_language(self, transcriber) -> str:
        """The ISO-639-1 code Whisper is forced to transcribe in."""

    @abstractmethod
    def nextLanguage(self) -> "LanguageState":
        """The next state in the language carousel. Each state names its own
        successor, so cycling through languages is `state = state.nextLanguage()`
        — no central if/else that has to grow with every new language."""


class EnglishState(LanguageState):
    def reply_language(self, chat):
        return REPLY_IN_ENGLISH

    def stt_language(self, transcriber):
        return STT_LANGUAGE_ENGLISH

    def nextLanguage(self):
        return SpanishState()


class SpanishState(LanguageState):
    def reply_language(self, chat):
        return REPLY_IN_SPANISH

    def stt_language(self, transcriber):
        return STT_LANGUAGE_SPANISH

    def nextLanguage(self):
        return EnglishState()
