from dotenv import load_dotenv
load_dotenv()

from bmo.audio.capture import AudioCapture, RATE
from bmo.audio.wake_word import WakeWordDetector


def main():
    capture = AudioCapture()
    detector = WakeWordDetector()

    print("BMO is ready. Listening for wake word...")
    try:
        while True:
            chunk = capture.read_chunk()
            if detector.process(chunk):
                print("Wake word detected! Recording...")
                audio = capture.record_until_silence()
                duration = len(audio) / (2 * RATE)
                print(f"Captured {duration:.1f}s of audio — ready for STT")
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        capture.close()


if __name__ == "__main__":
    main()
