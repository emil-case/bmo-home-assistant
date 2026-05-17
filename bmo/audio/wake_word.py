from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model

_MODELS_DIR = Path(openwakeword.__file__).parent / "resources" / "models"


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis_v0.1", threshold: float = 0.5):
        model_path = _MODELS_DIR / f"{model_name}.onnx"
        self._model = Model(wakeword_model_paths=[str(model_path)])
        self._threshold = threshold

    def process(self, audio: np.ndarray) -> bool:
        predictions = self._model.predict(audio)
        return any(score >= self._threshold for score in predictions.values())
