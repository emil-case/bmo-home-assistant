from unittest.mock import patch

from bmo.audio.capture import AudioCapture


def _capture():
    """Build an AudioCapture backed by a mock PyAudio stream."""
    with patch("bmo.audio.capture.pyaudio") as mock_pyaudio:
        stream = mock_pyaudio.PyAudio.return_value.open.return_value
        capture = AudioCapture()
    return capture, stream


def test_given_active_stream_when_paused_then_stops_stream():
    capture, stream = _capture()
    stream.is_active.return_value = True
    capture.pause()
    stream.stop_stream.assert_called_once()


def test_given_stopped_stream_when_paused_then_does_nothing():
    capture, stream = _capture()
    stream.is_active.return_value = False
    capture.pause()
    stream.stop_stream.assert_not_called()


def test_given_stopped_stream_with_buffered_audio_when_resumed_then_restarts_and_flushes():
    capture, stream = _capture()
    stream.is_active.return_value = False
    stream.get_read_available.return_value = 5
    capture.resume()
    stream.start_stream.assert_called_once()
    stream.read.assert_called_once_with(5, exception_on_overflow=False)


def test_given_active_stream_with_empty_buffer_when_resumed_then_does_not_restart_or_flush():
    capture, stream = _capture()
    stream.is_active.return_value = True  # already running, no restart needed
    stream.get_read_available.return_value = 0
    capture.resume()
    stream.start_stream.assert_not_called()
    stream.read.assert_not_called()
