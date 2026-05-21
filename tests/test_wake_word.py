from unittest.mock import patch

import numpy as np

from bmo.audio.wake_word import WakeWordDetector


def _detector(score, threshold=0.3):
    """Build a detector whose underlying model predicts `score`."""
    with patch("bmo.audio.wake_word.Model") as MockModel:
        model = MockModel.return_value
        model.predict.return_value = {"hey_bmo": score}
        detector = WakeWordDetector(model_path="dummy.onnx", threshold=threshold)
    return detector, model


_CHUNK = np.zeros(1280, dtype=np.int16)


def test_given_score_above_threshold_when_processing_then_returns_true():
    detector, _ = _detector(0.9, threshold=0.3)
    assert detector.process(_CHUNK) is True


def test_given_score_below_threshold_when_processing_then_returns_false():
    detector, _ = _detector(0.1, threshold=0.3)
    assert detector.process(_CHUNK) is False


def test_given_detector_when_reset_then_resets_model():
    detector, model = _detector(0.0)
    detector.reset()
    model.reset.assert_called_once()
