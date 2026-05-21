from dotenv import load_dotenv
load_dotenv()

import time

from bmo.audio.capture import AudioCapture, RATE
from bmo.audio.wake_word import WakeWordDetector
from bmo.stt.transcribe import Transcriber


def main():
    capture = AudioCapture()
    detector = WakeWordDetector()
    transcriber = Transcriber()

    print("BMO is ready. Listening for wake word...")
    try:
        while True:
            chunk = capture.read_chunk()
            if detector.process(chunk):
                print("Wake word detected! Recording...")
                audio = capture.record_until_silence()
                duration = len(audio) / (2 * RATE)
                print(f"Captured {duration:.1f}s of audio — transcribing...")
                transcript = transcriber.transcribe(audio)
                print(f"You said: {transcript}")

                # Placeholder for handling the command (LLM -> TTS -> playback).
                # Pause the mic so nothing is captured (incl. BMO's own voice once
                # TTS lands) while busy, then resume. Replace the sleep with the
                # real pipeline.
                capture.pause()
                print("Handling command... (listening paused for 60s)")
                time.sleep(60)
                capture.resume()
                print("Done. Listening for wake word...")
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        capture.close()


if __name__ == "__main__":
    main()
