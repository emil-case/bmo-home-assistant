import numpy as np
import sounddevice as sd

_SR = 44100


def play_acknowledgement() -> None:
    """Play a short beep so the user knows the wake word registered.

    Placeholder — swap this out for a BMO voice clip later.
    """
    t = np.linspace(0, 0.15, int(_SR * 0.15), endpoint=False)
    beep = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    sd.play(beep, _SR)
    sd.wait()
