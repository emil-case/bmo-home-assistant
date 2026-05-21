from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bmo.tts.speak import Speaker, _default_model


def _voice(chunks, sample_rate=22050):
    """A fake Piper voice whose synthesize() yields the given int16 chunks."""
    voice = MagicMock()
    voice.config.sample_rate = sample_rate
    voice.synthesize.return_value = [
        MagicMock(audio_int16_array=c) for c in chunks
    ]
    return voice


def test_given_text_when_say_then_synthesizes_and_plays_concatenated_audio():
    voice = _voice(
        [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5], dtype=np.int16)],
        sample_rate=22050,
    )

    with patch("bmo.tts.speak.sd") as sd:
        Speaker(voice=voice).say("hello")

    voice.synthesize.assert_called_once_with("hello")
    played, rate = sd.play.call_args.args
    assert list(played) == [1, 2, 3, 4, 5]
    assert rate == 22050
    sd.wait.assert_called_once()


def test_given_text_that_yields_no_audio_when_say_then_does_not_play():
    voice = _voice([])

    with patch("bmo.tts.speak.sd") as sd:
        Speaker(voice=voice).say("")

    sd.play.assert_not_called()
    sd.wait.assert_not_called()


def test_given_no_voice_files_when_loading_default_then_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        _default_model()


def test_given_voice_files_when_loading_default_then_returns_first_onnx(tmp_path, monkeypatch):
    (tmp_path / "b_voice.onnx").touch()
    (tmp_path / "a_voice.onnx").touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)

    assert _default_model() == tmp_path / "a_voice.onnx"
