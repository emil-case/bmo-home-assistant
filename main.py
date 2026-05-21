from dotenv import load_dotenv
load_dotenv()

from bmo.audio.capture import AudioCapture, RATE
from bmo.audio.cue import play_acknowledgement
from bmo.audio.wake_word import WakeWordDetector
from bmo.llm.chat import ChatSession
from bmo.stt.transcribe import Transcriber


def main():
    capture = AudioCapture()
    detector = WakeWordDetector()
    transcriber = Transcriber()
    chat = ChatSession()

    print("BMO is ready. Listening for wake word...")
    try:
        while True:
            chunk = capture.read_chunk()
            if detector.process(chunk):
                print("Wake word detected! Recording...")
                play_acknowledgement()
                audio = capture.record_until_silence()
                duration = len(audio) / (2 * RATE)
                print(f"Captured {duration:.1f}s of audio — transcribing...")
                transcript = transcriber.transcribe(audio)
                print(f"You said: {transcript}")

                # Pause the mic so nothing is captured (incl. BMO's own voice once
                # TTS lands) while handling the command, then resume.
                capture.pause()
                reply = chat.send(transcript)
                print(f"BMO: {reply}")
                # TODO: tool-use loop and TTS playback go here.
                capture.resume()
                print("Listening for wake word...")
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        capture.close()


if __name__ == "__main__":
    main()
