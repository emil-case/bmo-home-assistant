import pytest

from bmo.tts.voice_catalog import VoiceCatalog


# The catalog owns its directory, so tests just hand it a real tmp_path with
# voice files in it — no module globals to monkeypatch.

def test_given_voice_for_tag_when_model_for_then_returns_matching_onnx(tmp_path):
    (tmp_path / "en_US-amy.onnx").touch()
    (tmp_path / "es_ES-dave.onnx").touch()

    assert VoiceCatalog(tmp_path).model_for("es") == tmp_path / "es_ES-dave.onnx"


def test_given_multiple_matches_when_model_for_then_picks_first_sorted(tmp_path):
    (tmp_path / "en_US-amy.onnx").touch()
    (tmp_path / "en_GB-alan.onnx").touch()

    assert VoiceCatalog(tmp_path).model_for("en") == tmp_path / "en_GB-alan.onnx"


def test_given_no_voice_for_tag_when_model_for_then_falls_back_to_default(tmp_path):
    # Only English is installed; asking for Spanish still yields a usable voice
    # so BMO keeps talking rather than crashing.
    (tmp_path / "en_US-amy.onnx").touch()

    assert VoiceCatalog(tmp_path).model_for("es") == tmp_path / "en_US-amy.onnx"


def test_given_empty_directory_when_model_for_then_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VoiceCatalog(tmp_path).model_for("en")


def test_given_default_catalog_when_built_then_is_a_voice_catalog():
    # The class-side constructor roots a catalog at the standard voices dir.
    assert isinstance(VoiceCatalog.default(), VoiceCatalog)
