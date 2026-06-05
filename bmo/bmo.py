from bmo.audio.capture import AudioCapture, RATE
from bmo.audio.cue import play_acknowledgement
from bmo.audio.wake_word import WakeWordDetector
from bmo.language.state import LanguageState
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
        # Set before building components: ChatSession asks for the prompt's
        # reply-language clause at construction, which routes back here.
        self._languageState = LanguageState.default()
        self._capture = AudioCapture(owner=self)
        self._detector = WakeWordDetector(owner=self)
        self._transcriber = Transcriber(owner=self)
        self._chat = ChatSession(owner=self)
        self._speaker = Speaker(owner=self)

    def reply_language(self):
        """A component asks for the active language's reply clause; route it to
        the current state. The method name is the dispatch — the caller doesn't
        pass itself, since BMO already holds each component."""
        return self._languageState.reply_language()

    def stt_language(self):
        """The Transcriber asks for the active language's Whisper code; route it
        to the current state (dispatch by method name, not a passed component)."""
        return self._languageState.stt_language()

    def tts_voice(self):
        """The Speaker asks for the active language's TTS voice tag; route it to
        the current state (dispatch by method name, not a passed component)."""
        return self._languageState.tts_voice()

    def switch_language(self):
        """Advance to the next language in the carousel and reseed the chat so
        its system prompt — and thus BMO's reply language — changes. Each state
        names its own successor, so BMO doesn't decide which language is next.
        Conversation history is wiped."""
        self._languageState = self._languageState.nextLanguage()
        self._chat.reset()

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
