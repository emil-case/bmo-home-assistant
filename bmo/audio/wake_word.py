from pathlib import Path

import numpy as np
from openwakeword.model import Model

_RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
DEFAULT_MODEL = _RESOURCES_DIR / "hey_bmo_v2.0.onnx"


class WakeWordDetector:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL, threshold: float = 0.3):
        self._model = Model(wakeword_model_paths=[str(model_path)])
        self._threshold = threshold

    def score(self, audio: np.ndarray) -> float:
        predictions = self._model.predict(audio)
        return max(predictions.values(), default=0.0)

    def process(self, audio: np.ndarray) -> bool:
        return self.score(audio) >= self._threshold

    def reset(self) -> None:
        """Clear the model's rolling audio/feature buffers.

        Call after handling a command so leftover audio (the wake word and the
        spoken command) doesn't immediately re-trigger detection.
        """
        self._model.reset()
