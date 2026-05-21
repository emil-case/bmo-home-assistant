import wave
from unittest.mock import MagicMock

from bmo.audio.capture import RATE
from bmo.stt.transcribe import DEFAULT_MODEL, Transcriber, _pcm_to_wav


def test_given_raw_pcm_when_wrapped_then_produces_valid_16khz_mono_wav():
    pcm = b"\x01\x02" * 16000  # 16000 frames of 16-bit mono
    buf = _pcm_to_wav(pcm)

    with wave.open(buf, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == RATE
        assert w.getnframes() == 16000
        assert w.readframes(16000) == pcm


def test_given_pcm_audio_when_transcribing_then_sends_wav_and_returns_stripped_text():
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(text="  hey bmo  ")

    result = Transcriber(client=client).transcribe(b"\x00\x00" * 1600)

    assert result == "hey bmo"
    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["model"] == DEFAULT_MODEL
    filename, _fileobj, mime = kwargs["file"]
    assert filename.endswith(".wav")
    assert mime == "audio/wav"
