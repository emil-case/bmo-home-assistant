import numpy as np
import sounddevice as sd
from piper import PiperVoice

from bmo.tts.voice_catalog import VoiceCatalog


class Speaker:
    """Local neural TTS via Piper. Synthesizes text and plays it on the speaker.

    Which voice speaks is language-dependent: per `say()`, the Speaker asks its
    owner (BMO -> the active LanguageState) for a voice tag, then asks its
    VoiceCatalog for the matching model — so a `switch_language()` lands on the
    next utterance, mirroring how the Transcriber resolves its STT code. Loading
    an .onnx is expensive, so each language's voice is loaded from disk once and
    cached here.
    """

    def __init__(self, owner):
        self._owner = owner
        # The Speaker makes its own catalog, rooted at the standard voices dir.
        self._catalog = VoiceCatalog.default()
        # Loaded PiperVoice objects, keyed by the voice tag a LanguageState names,
        # so each language's model loads from disk at most once.
        self._cache: dict = {}

    def _active_voice(self) -> PiperVoice:
        """The PiperVoice for the language BMO is in now, loaded once and cached."""
        tag = self._owner.tts_voice()
        if tag not in self._cache:
            self._cache[tag] = PiperVoice.load(str(self._catalog.model_for(tag)))
        return self._cache[tag]

    def say(self, text: str) -> None:
        """Synthesize `text` in the active language's voice and play it, blocking
        until playback finishes."""
        voice = self._active_voice()
        chunks = [c.audio_int16_array for c in voice.synthesize(text)]
        if not chunks:
            return
        audio = np.concatenate(chunks)
        sd.play(audio, voice.config.sample_rate)
        sd.wait()
