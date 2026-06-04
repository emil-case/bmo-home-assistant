from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bmo.bmo import BMO


@contextmanager
def _bmo():
    """Build a BMO with every component (and the beep) mocked.

    Patches the component classes in the bmo.bmo namespace so construction
    yields mocks instead of real hardware/API clients, plus play_acknowledgement
    since it's invoked while handling a command. Yields (bmo, namespace) where
    the namespace exposes both the patched classes (`.classes.*`) and the built
    instances (`.capture`, `.chat`, ...).
    """
    with patch("bmo.bmo.AudioCapture") as Capture, \
         patch("bmo.bmo.WakeWordDetector") as Detector, \
         patch("bmo.bmo.Transcriber") as Transcriber, \
         patch("bmo.bmo.ChatSession") as Chat, \
         patch("bmo.bmo.Speaker") as Speaker, \
         patch("bmo.bmo.play_acknowledgement") as beep:
        bmo = BMO()
        ns = SimpleNamespace(
            classes=SimpleNamespace(
                capture=Capture, detector=Detector, transcriber=Transcriber,
                chat=Chat, speaker=Speaker,
            ),
            capture=Capture.return_value,
            detector=Detector.return_value,
            transcriber=Transcriber.return_value,
            chat=Chat.return_value,
            speaker=Speaker.return_value,
            beep=beep,
        )
        # _handle_command does len(audio) / (2 * RATE), so this must be real bytes.
        ns.capture.record_until_silence.return_value = b"\x00\x00"
        yield bmo, ns


def test_given_bmo_when_initialized_then_each_component_is_owned_by_bmo():
    with _bmo() as (bmo, m):
        for cls in (m.classes.capture, m.classes.detector, m.classes.transcriber,
                    m.classes.chat, m.classes.speaker):
            cls.assert_called_once()
            assert cls.call_args.kwargs["owner"] is bmo


def test_given_command_when_handled_then_transcript_flows_to_chat_and_reply_to_speaker():
    with _bmo() as (bmo, m):
        m.transcriber.transcribe.return_value = "hola"
        m.chat.send.return_value = "hi"

        bmo._handle_command()

        m.chat.send.assert_called_once_with("hola")
        m.speaker.say.assert_called_once_with("hi")


def test_given_command_when_handled_then_mic_paused_around_reply_and_detector_reset():
    with _bmo() as (bmo, m):
        # Real strings, not mocks: _handle_command formats the transcript and
        # reply into print(), and formatting a mock that's attached below would
        # log a spurious nested call (e.g. send().__format__) into mock_calls.
        m.transcriber.transcribe.return_value = "hola"
        m.chat.send.return_value = "hi"

        manager = MagicMock()
        manager.attach_mock(m.capture.pause, "pause")
        manager.attach_mock(m.chat.send, "send")
        manager.attach_mock(m.speaker.say, "say")
        manager.attach_mock(m.capture.resume, "resume")
        manager.attach_mock(m.detector.reset, "reset")

        bmo._handle_command()

        # Mic must be down before BMO speaks and back up after, and the detector
        # reset last so leftover audio doesn't re-trigger the next loop.
        assert [c[0] for c in manager.mock_calls] == [
            "pause", "send", "say", "resume", "reset",
        ]


def test_given_wake_word_then_interrupt_when_running_then_handles_command_and_closes():
    with _bmo() as (bmo, m):
        m.detector.process.side_effect = [True, KeyboardInterrupt]
        m.transcriber.transcribe.return_value = "x"
        m.chat.send.return_value = "y"

        bmo.run()

        m.chat.send.assert_called_once_with("x")
        m.speaker.say.assert_called_once_with("y")
        m.capture.close.assert_called_once()


def test_given_no_wake_word_when_running_then_never_handles_command():
    with _bmo() as (bmo, m):
        m.detector.process.side_effect = [False, KeyboardInterrupt]

        bmo.run()

        m.chat.send.assert_not_called()
        m.speaker.say.assert_not_called()
        m.capture.close.assert_called_once()
