import pyaudio
import numpy as np

CHUNK = 1280  # 80ms at 16kHz — required by openwakeword
RATE = 16000
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500   # RMS energy below this = silence
SILENCE_DURATION = 1.5    # seconds of silence to stop recording


class AudioCapture:
    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

    def read_chunk(self) -> np.ndarray:
        raw = self._stream.read(CHUNK, exception_on_overflow=False)
        return np.frombuffer(raw, dtype=np.int16)

    def record_until_silence(self) -> bytes:
        frames = []
        silent_chunks = 0
        min_chunks = int(RATE / CHUNK * 0.5)
        max_silent_chunks = int(RATE / CHUNK * SILENCE_DURATION)

        while True:
            raw = self._stream.read(CHUNK, exception_on_overflow=False)
            frames.append(raw)

            rms = float(np.sqrt(np.mean(
                np.frombuffer(raw, dtype=np.int16).astype(np.float32) ** 2
            )))

            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if len(frames) >= min_chunks and silent_chunks >= max_silent_chunks:
                break

        return b"".join(frames)

    def close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
