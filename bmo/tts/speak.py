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
    """

    def __init__(self, model_path: Path | str | None = None, voice=None, owner=None):
        self._owner = owner
        if voice is not None:
            self._voice = voice
        else:
            model_path = Path(model_path) if model_path else _default_model()
            self._voice = PiperVoice.load(str(model_path))
        self._sample_rate = self._voice.config.sample_rate

    def say(self, text: str) -> None:
        """Synthesize `text` and play it, blocking until playback finishes."""
        chunks = [c.audio_int16_array for c in self._voice.synthesize(text)]
        if not chunks:
            return
        audio = np.concatenate(chunks)
        sd.play(audio, self._sample_rate)
        sd.wait()
