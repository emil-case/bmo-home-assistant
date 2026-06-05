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


def test_given_owner_when_say_then_loads_voice_matching_owners_tag(tmp_path, monkeypatch):
    # With an owner, the active voice comes from it (BMO -> LanguageState): the
    # tag is matched against the installed .onnx filenames (Piper's `es_ES-...`).
    (tmp_path / "en_US-amy.onnx").touch()
    (tmp_path / "es_ES-dave.onnx").touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)
    owner = MagicMock()
    owner.tts_voice.return_value = "es"

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.return_value = _voice([np.array([1], dtype=np.int16)])
        Speaker(owner=owner).say("hola")

    Piper.load.assert_called_once_with(str(tmp_path / "es_ES-dave.onnx"))
    owner.tts_voice.assert_called_once()


def test_given_same_language_when_say_twice_then_voice_loaded_once(tmp_path, monkeypatch):
    # Loading an .onnx is expensive, so a language's voice is cached after the
    # first use — repeated utterances must not reload it.
    (tmp_path / "en_US-amy.onnx").touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)
    owner = MagicMock()
    owner.tts_voice.return_value = "en"

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.return_value = _voice([np.array([1], dtype=np.int16)])
        speaker = Speaker(owner=owner)
        speaker.say("hi")
        speaker.say("again")

    Piper.load.assert_called_once()


def test_given_language_switch_when_say_then_loads_each_voice_once(tmp_path, monkeypatch):
    # A switch lands on the next utterance: the new tag loads its own voice, and
    # switching back reuses the cached one (load count stays at two).
    (tmp_path / "en_US-amy.onnx").touch()
    (tmp_path / "es_ES-dave.onnx").touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)
    owner = MagicMock()
    owner.tts_voice.side_effect = ["en", "es", "en"]

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.side_effect = lambda p: _voice([np.array([1], dtype=np.int16)])
        speaker = Speaker(owner=owner)
        speaker.say("hi")
        speaker.say("hola")
        speaker.say("hi again")

    assert Piper.load.call_count == 2
    loaded = {c.args[0] for c in Piper.load.call_args_list}
    assert loaded == {str(tmp_path / "en_US-amy.onnx"), str(tmp_path / "es_ES-dave.onnx")}


def test_given_no_voice_for_language_when_say_then_falls_back_to_default(tmp_path, monkeypatch):
    # Only English is installed; asking for Spanish must still speak (in the
    # default/first voice) rather than crash, so BMO keeps talking.
    (tmp_path / "en_US-amy.onnx").touch()
    monkeypatch.setattr("bmo.tts.speak.VOICES_DIR", tmp_path)
    owner = MagicMock()
    owner.tts_voice.return_value = "es"

    with patch("bmo.tts.speak.PiperVoice") as Piper, patch("bmo.tts.speak.sd"):
        Piper.load.return_value = _voice([np.array([1], dtype=np.int16)])
        Speaker(owner=owner).say("hola")

    Piper.load.assert_called_once_with(str(tmp_path / "en_US-amy.onnx"))
