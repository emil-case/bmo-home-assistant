from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bmo.tts.speak import Speaker, _default_model


def _loaded_voice(chunks, sample_rate=22050):
    """A fake *loaded* Piper voice (what PiperVoice.load returns): synthesize()
    yields the given int16 chunks. Piper is the boundary, so tests patch its
    load() and feed back one of these — no test-only seam in the Speaker itself."""
    voice = MagicMock()
    voice.config.sample_rate = sample_rate
    voice.synthesize.return_value = [
        MagicMock(audio_int16_array=c) for c in chunks
    ]
    return voice


def _owner(tag):
    """Stand-in for BMO: reports the active language's voice tag, like the real
    BMO.tts_voice() does."""
    return SimpleNamespace(tts_voice=lambda: tag)


def _install_voices(tmp_path, monkeypatch, *names):
    for name in names:
        (tmp_path / name).touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)


def test_given_text_when_say_then_synthesizes_and_plays_concatenated_audio(tmp_path, monkeypatch):
    _install_voices(tmp_path, monkeypatch, "en_US-amy.onnx")
    voice = _loaded_voice(
        [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5], dtype=np.int16)],
        sample_rate=22050,
    )

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd") as sd:
        Piper.load.return_value = voice
        Speaker(_owner("en")).say("hello")

    voice.synthesize.assert_called_once_with("hello")
    played, rate = sd.play.call_args.args
    assert list(played) == [1, 2, 3, 4, 5]
    assert rate == 22050
    sd.wait.assert_called_once()


def test_given_text_that_yields_no_audio_when_say_then_does_not_play(tmp_path, monkeypatch):
    _install_voices(tmp_path, monkeypatch, "en_US-amy.onnx")

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd") as sd:
        Piper.load.return_value = _loaded_voice([])
        Speaker(_owner("en")).say("")

    sd.play.assert_not_called()
    sd.wait.assert_not_called()


def test_given_owner_tag_when_say_then_loads_voice_matching_that_tag(tmp_path, monkeypatch):
    # The voice played is the one whose filename starts with the owner's tag
    # (Piper's `es_ES-...`), so BMO speaks the active language.
    _install_voices(tmp_path, monkeypatch, "en_US-amy.onnx", "es_ES-dave.onnx")

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.return_value = _loaded_voice([np.array([1], dtype=np.int16)])
        Speaker(_owner("es")).say("hola")

    Piper.load.assert_called_once_with(str(tmp_path / "es_ES-dave.onnx"))


def test_given_same_language_when_say_twice_then_voice_loaded_once(tmp_path, monkeypatch):
    # Loading an .onnx is expensive, so a language's voice is cached after the
    # first use — repeated utterances must not reload it.
    _install_voices(tmp_path, monkeypatch, "en_US-amy.onnx")

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.return_value = _loaded_voice([np.array([1], dtype=np.int16)])
        speaker = Speaker(_owner("en"))
        speaker.say("hi")
        speaker.say("again")

    Piper.load.assert_called_once()


def test_given_language_switch_when_say_then_loads_each_voice_once(tmp_path, monkeypatch):
    # A switch lands on the next utterance: the new tag loads its own voice, and
    # switching back reuses the cached one (load count stays at two).
    _install_voices(tmp_path, monkeypatch, "en_US-amy.onnx", "es_ES-dave.onnx")
    owner = MagicMock()
    owner.tts_voice.side_effect = ["en", "es", "en"]

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.side_effect = lambda p: _loaded_voice([np.array([1], dtype=np.int16)])
        speaker = Speaker(owner)
        speaker.say("hi")
        speaker.say("hola")
        speaker.say("hi again")

    assert Piper.load.call_count == 2
    loaded = {c.args[0] for c in Piper.load.call_args_list}
    assert loaded == {str(tmp_path / "en_US-amy.onnx"), str(tmp_path / "es_ES-dave.onnx")}


def test_given_no_voice_for_language_when_say_then_falls_back_to_default(tmp_path, monkeypatch):
    # Only English is installed; asking for Spanish must still speak (in the
    # default/first voice) rather than crash, so BMO keeps talking.
    _install_voices(tmp_path, monkeypatch, "en_US-amy.onnx")

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.return_value = _loaded_voice([np.array([1], dtype=np.int16)])
        Speaker(_owner("es")).say("hola")

    Piper.load.assert_called_once_with(str(tmp_path / "en_US-amy.onnx"))


def test_given_no_voice_files_when_loading_default_then_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        _default_model()


def test_given_voice_files_when_loading_default_then_returns_first_onnx(tmp_path, monkeypatch):
    (tmp_path / "b_voice.onnx").touch()
    (tmp_path / "a_voice.onnx").touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)

    assert _default_model() == tmp_path / "a_voice.onnx"
