from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from bmo.tts.speak import Speaker


def _loaded_voice(chunks, sample_rate=22050):
    """A fake *loaded* Piper voice (what PiperVoice.load returns): synthesize()
    yields the given int16 chunks. Piper is the boundary, so tests patch its
    load() and feed back one of these."""
    voice = MagicMock()
    voice.config.sample_rate = sample_rate
    voice.synthesize.return_value = [
        MagicMock(audio_int16_array=c) for c in chunks
    ]
    return voice


def _catalog(tag_to_path):
    """A fake VoiceCatalog mapping a tag to a model path."""
    catalog = MagicMock()
    catalog.model_for.side_effect = lambda tag: tag_to_path[tag]
    return catalog


def _owner(tag):
    """Stand-in for BMO: reports the active language's voice tag, like the real
    BMO.tts_voice() does."""
    return SimpleNamespace(tts_voice=lambda: tag)


@contextmanager
def _speaker_env(catalog):
    """The Speaker builds its own VoiceCatalog, so patch that construction to hand
    it a fake (no filesystem); PiperVoice/sounddevice are patched at the engine
    boundary. Yields the PiperVoice and sounddevice mocks for assertions."""
    with patch("bmo.tts.speak.VoiceCatalog") as VC, \
         patch("bmo.tts.speak.PiperVoice") as Piper, \
         patch("bmo.tts.speak.sd") as sd:
        VC.default.return_value = catalog
        yield Piper, sd


def test_given_text_when_say_then_synthesizes_and_plays_concatenated_audio():
    voice = _loaded_voice(
        [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5], dtype=np.int16)],
        sample_rate=22050,
    )
    catalog = _catalog({"en": Path("en_US-amy.onnx")})

    with _speaker_env(catalog) as (Piper, sd):
        Piper.load.return_value = voice
        Speaker(owner=_owner("en")).say("hello")

    voice.synthesize.assert_called_once_with("hello")
    played, rate = sd.play.call_args.args
    assert list(played) == [1, 2, 3, 4, 5]
    assert rate == 22050
    sd.wait.assert_called_once()


def test_given_text_that_yields_no_audio_when_say_then_does_not_play():
    catalog = _catalog({"en": Path("en_US-amy.onnx")})

    with _speaker_env(catalog) as (Piper, sd):
        Piper.load.return_value = _loaded_voice([])
        Speaker(owner=_owner("en")).say("")

    sd.play.assert_not_called()
    sd.wait.assert_not_called()


def test_given_owner_tag_when_say_then_loads_voice_the_catalog_returns_for_it():
    # The Speaker asks its owner for the tag, the catalog for that tag's model,
    # and Piper to load exactly that file — so BMO speaks the active language.
    catalog = _catalog({"es": Path("es_ES-dave.onnx")})

    with _speaker_env(catalog) as (Piper, _sd):
        Piper.load.return_value = _loaded_voice([np.array([1], dtype=np.int16)])
        Speaker(owner=_owner("es")).say("hola")

    catalog.model_for.assert_called_once_with("es")
    Piper.load.assert_called_once_with(str(Path("es_ES-dave.onnx")))


def test_given_same_language_when_say_twice_then_voice_loaded_once():
    # Loading an .onnx is expensive, so a language's voice is cached after the
    # first use — repeated utterances must not reload it.
    catalog = _catalog({"en": Path("en_US-amy.onnx")})

    with _speaker_env(catalog) as (Piper, _sd):
        Piper.load.return_value = _loaded_voice([np.array([1], dtype=np.int16)])
        speaker = Speaker(owner=_owner("en"))
        speaker.say("hi")
        speaker.say("again")

    Piper.load.assert_called_once()


def test_given_language_switch_when_say_then_loads_each_voice_once():
    # A switch lands on the next utterance: the new tag loads its own voice, and
    # switching back reuses the cached one (load count stays at two).
    catalog = _catalog({"en": Path("en_US-amy.onnx"), "es": Path("es_ES-dave.onnx")})
    owner = MagicMock()
    owner.tts_voice.side_effect = ["en", "es", "en"]

    with _speaker_env(catalog) as (Piper, _sd):
        Piper.load.side_effect = lambda p: _loaded_voice([np.array([1], dtype=np.int16)])
        speaker = Speaker(owner=owner)
        speaker.say("hi")
        speaker.say("hola")
        speaker.say("hi again")

    assert Piper.load.call_count == 2
    loaded = {c.args[0] for c in Piper.load.call_args_list}
    assert loaded == {str(Path("en_US-amy.onnx")), str(Path("es_ES-dave.onnx"))}
