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

    Piper's load() auto-discovers the JSON config next to the model
    (model_path + ".json"), so only the .onnx path is required.

    Which voice speaks is language-dependent: per `say()`, the Speaker asks its
    owner (BMO -> the active LanguageState) for a voice tag and uses the matching
    model — so a `switch_language()` lands on the next utterance, mirroring how
    the Transcriber resolves its STT code. Loading an .onnx is expensive, so each
    language's voice is loaded once and cached (the STT/reply values are plain
    strings resolved every call; a TTS voice is a model that must persist).
    """

    def __init__(self, model_path: Path | str | None = None, voice=None, owner=None):
        self._owner = owner
        # An explicitly injected voice (tests) bypasses file loading and language
        # resolution entirely.
        self._injected = voice
        # Explicit override for which .onnx to load when no language names one.
        self._model_path = Path(model_path) if model_path else None
        # Loaded PiperVoice objects, keyed by the voice tag a LanguageState names,
        # so each language's model loads from disk at most once.
        self._cache: dict = {}

    def _model_for(self, tag) -> Path:
        """The .onnx for `tag`: the first installed voice whose filename starts
        with it (Piper names voices `en_US-...`, `es_ES-...`). Falls back to an
        explicit model_path or the first voice found, so a language with no
        installed voice still speaks (in the default voice) rather than crashing."""
        if tag is not None:
            matches = sorted(VOICES_DIR.glob(f"{tag}*.onnx"))
            if matches:
                return matches[0]
        return self._model_path or _default_model()

    def _resolve_voice(self):
        """The PiperVoice for the active language. An injected voice (tests) wins;
        otherwise ask the owner for the tag and load+cache the matching model."""
        if self._injected is not None:
            return self._injected
        tag = self._owner.tts_voice() if self._owner is not None else None
        if tag not in self._cache:
            self._cache[tag] = PiperVoice.load(str(self._model_for(tag)))
        return self._cache[tag]

    def say(self, text: str) -> None:
        """Synthesize `text` in the active language's voice and play it, blocking
        until playback finishes."""
        voice = self._resolve_voice()
        chunks = [c.audio_int16_array for c in voice.synthesize(text)]
        if not chunks:
            return
        audio = np.concatenate(chunks)
        sd.play(audio, voice.config.sample_rate)
        sd.wait()
