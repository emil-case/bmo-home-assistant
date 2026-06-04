import pyaudio
import numpy as np

CHUNK = 1280  # 80ms at 16kHz — required by openwakeword
RATE = 16000
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 2000  # RMS energy below this = silence
SILENCE_DURATION = 1.5    # seconds of silence to stop recording


class AudioCapture:
    def __init__(self, owner=None):
        self._owner = owner
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

            #print(f"rms={rms:7.1f}  silent_chunks={silent_chunks}") debug command

            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if len(frames) >= min_chunks and silent_chunks >= max_silent_chunks:
                break

        return b"".join(frames)

    def pause(self):
        """Stop capturing so no audio buffers while BMO handles a command."""
        if self._stream.is_active():
            self._stream.stop_stream()

    def resume(self):
        """Resume capturing, discarding any frames buffered around the restart."""
        if not self._stream.is_active():
            self._stream.start_stream()
        # Drop straggler frames so the wake-word detector doesn't act on stale audio.
        available = self._stream.get_read_available()
        if available > 0:
            self._stream.read(available, exception_on_overflow=False)

    def close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
