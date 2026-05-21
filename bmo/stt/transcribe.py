import io
import wave

from groq import Groq

from bmo.audio.capture import RATE

# whisper-large-v3-turbo: fastest of Groq's Whisper models (~300ms), good enough
# for short voice commands. Swap to whisper-large-v3 for higher accuracy.
DEFAULT_MODEL = "whisper-large-v3-turbo"


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
    def __init__(self, model: str = DEFAULT_MODEL):
        self._client = Groq()
        self._model = model

    def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw PCM audio (from AudioCapture) to text."""
        wav = _pcm_to_wav(pcm)
        result = self._client.audio.transcriptions.create(
            file=("audio.wav", wav, "audio/wav"),
            model=self._model,
        )
        return result.text.strip()
