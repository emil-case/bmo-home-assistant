from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice

# Voice models (.onnx + .onnx.json) live here, alongside the wake-word model
# in resources/. They're gitignored.
VOICES_DIR = Path(__file__).resolve().parents[2] / "resources" / "voices"


def _default_model() -> Path:
    """First .onnx voice found in VOICES_DIR."""
    models = sorted(VOICES_DIR.glob("*.onnx"))
    if not models:
        raise FileNotFoundError(
            f"No Piper voice (.onnx) found in {VOICES_DIR}. Download a voice model "
            "and place the .onnx + .onnx.json there."
        )
    return models[0]


class Speaker:
    """Local neural TTS via Piper. Synthesizes text and plays it on the speaker.

    Which voice speaks is language-dependent: per `say()`, the Speaker asks its
    owner (BMO -> the active LanguageState) for a voice tag and plays the matching
    model — so a `switch_language()` lands on the next utterance, mirroring how
    the Transcriber resolves its STT code. Loading an .onnx is expensive, so each
    language's voice is loaded from disk once and cached.
    """

    def __init__(self, owner):
        self._owner = owner
        # Loaded PiperVoice objects, keyed by the voice tag a LanguageState names,
        # so each language's model loads from disk at most once.
        self._cache: dict = {}

    def _model_for(self, tag: str) -> Path:
        """The .onnx for `tag`: the first installed voice whose filename starts
        with it (Piper names voices `en_US-...`, `es_ES-...`). Falls back to the
        default voice, so a language with no installed voice still speaks (in that
        voice) rather than crashing."""
        matches = sorted(VOICES_DIR.glob(f"{tag}*.onnx"))
        return matches[0] if matches else _default_model()

    def _active_voice(self) -> PiperVoice:
        """The PiperVoice for the language BMO is in now, loaded once and cached."""
        tag = self._owner.tts_voice()
        if tag not in self._cache:
            self._cache[tag] = PiperVoice.load(str(self._model_for(tag)))
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
