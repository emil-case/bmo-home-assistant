from pathlib import Path

# Voice models (.onnx + .onnx.json) live here, alongside the wake-word model
# in resources/. They're gitignored.
VOICES_DIR = Path(__file__).resolve().parents[2] / "resources" / "voices"


class VoiceCatalog:
    """The set of Piper voices BMO has on disk, and how to pick one.

    Owns the voices directory so nothing else reaches into the filesystem: a
    language tag goes in (`model_for`), a model file comes out. A language with
    no installed voice falls back to the default one, so BMO keeps talking rather
    than crashing. The Speaker is given a catalog to collaborate with, so swapping
    where voices live (or faking them in a test) is a different catalog, not a
    patched module global.
    """

    @classmethod
    def default(cls) -> "VoiceCatalog":
        """A catalog rooted at the standard resources/voices/ directory — the one
        BMO uses in production."""
        return cls(VOICES_DIR)

    def __init__(self, voices_dir: Path):
        self._dir = voices_dir

    def model_for(self, tag: str) -> Path:
        """The .onnx whose filename starts with `tag` (Piper names voices
        `en_US-...`, `es_ES-...`), or the default voice if none matches."""
        matches = sorted(self._dir.glob(f"{tag}*.onnx"))
        return matches[0] if matches else self._default_model()

    def _default_model(self) -> Path:
        """The first .onnx in the directory; raises if there are none at all."""
        models = sorted(self._dir.glob("*.onnx"))
        if not models:
            raise FileNotFoundError(
                f"No Piper voice (.onnx) found in {self._dir}. Download a voice "
                "model and place the .onnx + .onnx.json there."
            )
        return models[0]
