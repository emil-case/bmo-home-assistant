import io
import wave

from groq import Groq

from bmo.audio.capture import RATE

# whisper-large-v3-turbo: fastest of Groq's Whisper models (~300ms), good enough
# for short voice commands. Swap to whisper-large-v3 for higher accuracy.
DEFAULT_MODEL = "whisper-large-v3-turbo"

# Fallback transcription language when there's no owner to ask (e.g. in tests).
# In a running BMO the active language comes from the owner's LanguageState; this
# is just the floor. Forcing a language stops Whisper auto-detecting (a non-native
# accent can otherwise be misheard as another language). ISO-639-1 code.
DEFAULT_LANGUAGE = "en"


def _pcm_to_wav(pcm: bytes, rate: int = RATE) -> io.BytesIO:
    """Wrap raw 16-bit mono PCM in a WAV container in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(rate)
        wav.writeframes(pcm)
    buf.seek(0)
    return buf


class Transcriber:
    def __init__(self, model: str = DEFAULT_MODEL, language: str = DEFAULT_LANGUAGE, client=None, owner=None):
        self._owner = owner
        self._client = client or Groq()
        self._model = model
        self._language = language

    def _resolve_language(self) -> str:
        """Active STT language: ask the owner (BMO -> current LanguageState) when
        present so a language switch takes effect immediately, else fall back to
        the language fixed at construction."""
        if self._owner is not None:
            return self._owner.stt_language(self)
        return self._language

    def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw PCM audio (from AudioCapture) to text."""
        wav = _pcm_to_wav(pcm)
        result = self._client.audio.transcriptions.create(
            file=("audio.wav", wav, "audio/wav"),
            model=self._model,
            language=self._resolve_language(),
        )
        return result.text.strip()
