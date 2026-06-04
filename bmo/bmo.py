from bmo.audio.capture import AudioCapture, RATE
from bmo.audio.cue import play_acknowledgement
from bmo.audio.wake_word import WakeWordDetector
from bmo.llm.chat import ChatSession
from bmo.stt.transcribe import Transcriber
from bmo.tts.speak import Speaker


class BMO:
    """Owns the components and runs the main voice-assistant loop.

    Constructed once at startup. Each component is initialized with `owner=self`
    so it can later delegate shared decisions (e.g. the active language) back to
    BMO. `run()` blocks on the wake word and handles each command end to end
    (record -> STT -> LLM -> TTS).
    """

    def __init__(self):
        self._capture = AudioCapture(owner=self)
        self._detector = WakeWordDetector(owner=self)
        self._transcriber = Transcriber(owner=self)
        self._chat = ChatSession(owner=self)
        self._speaker = Speaker(owner=self)

    def run(self):
        """Listen for the wake word forever, handling one command at a time."""
        print("BMO is ready. Listening for wake word...")
        try:
            while True:
                chunk = self._capture.read_chunk()
                if self._detector.process(chunk):
                    self._handle_command()
                    print("Listening for wake word...")
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            self._capture.close()

    def _handle_command(self):
        """Record one utterance, run it through the pipeline, and speak the reply."""
        print("Wake word detected! Recording...")
        play_acknowledgement()
        audio = self._capture.record_until_silence()
        duration = len(audio) / (2 * RATE)
        print(f"Captured {duration:.1f}s of audio — transcribing...")
        transcript = self._transcriber.transcribe(audio)
        print(f"You said: {transcript}")

        # Pause the mic so nothing is captured (incl. BMO's own voice once
        # TTS lands) while handling the command, then resume.
        self._capture.pause()
        reply = self._chat.send(transcript)  # runs the tool-use loop internally
        print(f"BMO: {reply}")
        self._speaker.say(reply)  # mic is paused, so BMO won't hear itself
        self._capture.resume()
        self._detector.reset()  # clear leftover audio so it doesn't re-trigger
