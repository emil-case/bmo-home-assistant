import numpy as np
from openwakeword.model import Model


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5):
        self._model = Model(wakeword_models=[model_name])
        self._threshold = threshold

    def process(self, audio: np.ndarray) -> bool:
        predictions = self._model.predict(audio)
        return any(score >= self._threshold for score in predictions.values())
