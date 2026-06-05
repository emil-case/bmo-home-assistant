# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BMO is a local voice assistant inspired by the character from Adventure Time. Designed to run on cheap Raspberry Pi hardware with minimal API spend. The full pipeline is:

wake word → audio capture → STT (Groq Whisper) → LLM tool-use loop (Groq Llama) → TTS (Piper) → audio playback

Web search (Tavily API) is available as an LLM tool.

## Architecture decisions

**Guiding principle:** run everything possible locally; only hit APIs for what's too heavy.

| Component | Choice | Reason |
|---|---|---|
| Wake word | OpenWakeWord | Free, local, runs on Pi Zero 2W |
| STT | Groq Whisper API | ~300ms transcription, generous free tier |
| LLM | Groq + Llama 4 Scout 17B | Free tier, fast, reliable structured tool calls (Llama 3.3 70B mis-formats them as `<function=...>` and Groq rejects with `tool_use_failed`) |
| TTS | Piper TTS | Designed for edge devices, very low RAM |
| Web search | Tavily API | Free tier (~1000 queries/month), LLM-shaped results with built-in answer |

**LLM loop is recursive, not single-pass.** The pattern is:
```
while AI returns a tool call:
    run the tool
    send result back to AI
speak the final text response
```
This allows tool chaining (e.g. search → summarize). Loop until model returns plain text.

**Conversation history is session-scoped.** Keep a `messages[]` list and send it each turn so BMO feels coherent within a session. Wipe on restart.

**Language is a State, resolved by double dispatch.** BMO holds a `LanguageState`
(`EnglishState` / `SpanishState` in `bmo/language/state.py`), created via
`LanguageState.default()` so BMO depends only on the abstract type, not a concrete
subclass. A component that needs a language-dependent value asks its owner (BMO), which
forwards the call to the current state. The component does **not** pass itself: the
method it calls already names the value it wants (`reply_language()` / `stt_language()`),
and BMO holds each component — so the answer depends on the method (which value) and the
concrete state (which language). That pairing is the double dispatch, resolved not by a
passed argument but by which method runs on which subclass. The value accessors are
**template methods**: the shared instance method (`reply_language()` / `stt_language()` /
`tts_voice()`) lives once on the abstract class and delegates to a *classmethod* hook
(`reply_language_constant()` / `stt_language_constant()` / `tts_voice_constant()`). That
hook is concrete and also lives once on the abstract class — it simply reads a per-language
class attribute (`_REPLY_CLAUSE` / `_STT_CODE` / `_TTS_VOICE`) that the concrete state
supplies. So the per-language data sits on the subclass as a plain constant (no `__init__`
carrying it, no accessor boilerplate repeated per subclass) and the lookup logic isn't
duplicated. Because those constants are plain attributes rather than abstract methods, ABC
can't enforce them, so `__init_subclass__` does: a concrete state missing one fails at
class-definition time instead of with a late `AttributeError`. The values so far
are the system prompt's
reply-language clause ("Always reply in English/Spanish"), asked for by the `ChatSession`,
and the Whisper STT language code (`en`/`es`), asked for by the `Transcriber` (the rest of
the system prompt is fixed English and lives in `chat.py`'s `build_system_prompt`). The
`ChatSession` resolves its clause once per reset (so a switch takes effect on the reseed);
the `Transcriber` resolves its code per `transcribe()` call, so a switch lands on the very
next utterance.
`BMO.switch_language()` advances the carousel — `state = state.nextLanguage()`, where
each state names its own successor (`EnglishState` ↔ `SpanishState`), so BMO doesn't
decide which language is next — then calls `ChatSession.reset()`, which wipes
history and reseeds the prompt in the new language. The TTS voice now switches too:
the `Speaker` asks BMO for the active `tts_voice()` tag (the Piper `.onnx` filename
prefix, `en`/`es`) once per `say()` — so a switch lands on the next spoken reply —
then asks its own `VoiceCatalog` (`tts/voice_catalog.py`), which it builds at
construction via `VoiceCatalog.default()`, for the matching model file. The catalog
owns `resources/voices/`, so nothing else touches the filesystem;
if a language has no installed voice it falls back to the default so BMO keeps
talking. Because loading an `.onnx` is expensive (unlike the reply/STT *strings*,
which are resolved every call), the `Speaker` loads each language's `PiperVoice` once
and caches it.

### Language switching — status

What a `switch_language()` call changes today, and what it doesn't:

| Capability | State | Where |
|---|---|---|
| LLM reply language (system-prompt clause) | ✅ Done — flips on the chat reset | `llm/chat.py`, `language/state.py` |
| STT input language (Whisper code) | ✅ Done — flips on the next utterance | `stt/transcribe.py`, `language/state.py` |
| Carousel + boot state (`nextLanguage()` / `default()`) | ✅ Done — states own succession; BMO depends only on the abstract type | `language/state.py` |
| TTS output voice | ✅ Done — `Speaker` resolves the active `tts_voice()` tag per `say()`, asks its own `VoiceCatalog` (built at construction) for the matching `.onnx` in `resources/voices/`, and caches each language's voice (needs a per-language voice installed; falls back to the default otherwise) | `tts/speak.py`, `tts/voice_catalog.py`, `language/state.py` |
| **A trigger that calls `switch_language()`** | ❌ Missing — the method exists but nothing invokes it (planned: a GPIO button on the Pi) | — |

Adding a language = add a concrete `LanguageState` (its constants + `nextLanguage()`)
and splice it into the carousel, then install a matching Piper voice in
`resources/voices/`. The reply/STT/voice values come for free via the abstract
methods; the trigger is the one piece still unbuilt.

## Development environment

**Develop on Linux, not Windows.** The target is Raspberry Pi OS (Debian-based Linux). Developing on Linux avoids Windows-specific package issues (e.g. `tflite-runtime` has no Windows wheels, `pyaudio` requires workarounds on Windows).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Pi
pip install -r requirements.txt

cp .env.example .env             # then fill in API keys
```

Required env vars (`.env`):
- `GROQ_API_KEY` — used for both STT (Whisper) and LLM inference
- `TAVILY_API_KEY` — used for the web search tool

Piper TTS requires downloading a voice model (`.onnx` + `.onnx.json`) and placing it under `resources/voices/` (alongside the wake-word model in `resources/`). `Speaker` auto-loads the first `.onnx` it finds there. These files are gitignored.

## Testing

```bash
pip install -r requirements-dev.txt   # minimal test-only deps (dev/CI, not the Pi)
pytest
```

`requirements-dev.txt` is the minimal set the tests import (pytest + numpy, pyaudio,
groq, openwakeword, onnxruntime, sounddevice, piper-tts, requests) — deliberately
*not* the full runtime `requirements.txt`. The mocked objects' imports still resolve
at module load, so those packages must be installed; python-dotenv is not.

Tests live in `tests/` and run with no hardware or network — the boundaries are
mocked, so nothing hits a real mic, speaker, or the Groq API:
- **Pure logic** (`_pcm_to_wav`) is tested directly.
- **API clients** (`Transcriber`, `ChatSession`) take an optional `client=` arg so
  tests inject a fake Groq client instead of patching imports.
- **Hardware wrappers** (`AudioCapture`, `WakeWordDetector`) are tested by patching
  `pyaudio` / the openwakeword `Model`.
- **Collaborators** (`VoiceCatalog`) are tested directly against a real temp
  directory — no globals to patch. The `Speaker` builds its own `VoiceCatalog`, so
  its tests patch `VoiceCatalog.default()` to hand it a *fake* (no filesystem)
  while patching `PiperVoice`/`sounddevice` at the engine boundary.

The main loop now lives in `BMO` (`bmo/bmo.py`), not `main.py`, and is unit-tested
(`tests/test_bmo.py`) by patching the component classes in the `bmo.bmo` namespace —
no hardware or network. `main.py` is just `BMO().run()` glue and isn't tested. Test
names follow the `given_when_then` convention.

**Integration tests** (`tests/test_integration.py`) build a *real* BMO with a real
`ChatSession` and real `LanguageState` — only the hardware/API boundaries (mic, wake
word, STT, TTS, the Groq client) are mocked. The unit tests isolate one piece at a
time (and so each mock out a link of the `ChatSession → BMO.reply_language() →
LanguageState` chain); the integration test runs that whole delegation chain for
real, e.g. asserting `switch_language()` actually changes the chat's system prompt.
Keep the two layers complementary: unit tests localize failures, integration tests
prove the wiring.

CI runs the suite on every push and PR to `main` via `.github/workflows/tests.yml`
(installs PortAudio for pyaudio, then `requirements-dev.txt`, then `pytest`). No
secrets needed — tests never reach Groq.

## Package structure

```
main.py              # entry point — loads .env, builds BMO, runs it (thin)
bmo/
  bmo.py             # BMO: orchestrator; owns the components, the main loop, and the LanguageState
  audio/
    capture.py       # owns PyAudio stream; read_chunk(), record_until_silence(), pause()/resume()
    wake_word.py     # wraps openwakeword Model; process(chunk) -> bool
    cue.py           # play_acknowledgement() — beep when wake word fires (placeholder for BMO voice)
  language/
    state.py         # LanguageState template + EnglishState/SpanishState; serves the reply-language clause + STT code + TTS voice tag via template-method classmethod hooks, plus the carousel (nextLanguage/default)
  stt/
    transcribe.py    # Transcriber: PCM -> in-memory WAV -> Groq Whisper -> text
  llm/
    chat.py          # ChatSession: session history + Groq Llama + recursive tool-use loop; build_system_prompt() + reset()
  tts/
    speak.py         # Speaker: Piper synthesis -> sounddevice playback; per say() asks BMO for the active tts_voice tag, asks its VoiceCatalog for the model, and caches each loaded voice
    voice_catalog.py # VoiceCatalog: owns resources/voices/; resolves a language tag -> .onnx model file (falls back to the default voice); the Speaker builds one via VoiceCatalog.default()
  tools/
    tavily_search.py # Tavily web search tool exposed to the LLM
tests/               # pytest suite (mocks hardware + APIs, see Testing)
```

`README.md` is the public-facing overview (project pitch, status, services, setup).

**Always update `README.md` before pushing to the remote.** Whenever you push, first
review what changed in the commits being pushed and bring the README in sync — its
status checklist, services, structure, and setup — alongside the **Current state**
section below. Include the README update in the push.

## Current state

- [x] Wake word detection + audio capture pipeline (`bmo/audio/`)
- [x] Main loop scaffold (`main.py`) — detects wake word, plays a beep, records until silence
- [x] STT — captured audio sent to Groq Whisper, returns transcript (`bmo/stt/`)
- [x] Half-duplex mic — `pause()`/`resume()` stop capture while BMO handles a command
- [x] TTS — final text synthesized by Piper and played back (`bmo/tts/speak.py`)
- [x] Session conversation history — `ChatSession` keeps `messages[]` for the session
- [x] LLM tool-use loop — `ChatSession.send()` runs tool calls recursively until the model returns plain text (`bmo/llm/chat.py`)
- [x] Web search tool — `bmo/tools/tavily_search.py` (Tavily API), registered as a default LLM tool
- [x] `BMO` orchestrator (`bmo/bmo.py`) — owns the components and the main loop; `main.py` is thin glue
- [x] Component ownership — every component takes `owner=` and stores it (`self._owner`); BMO sets itself as owner at init, so components can later delegate shared decisions (e.g. active language) back to BMO
- [x] Bilingual reply language (EN/ES) — `LanguageState` (`bmo/language/state.py`) supplies the system prompt's reply-language clause; `BMO.switch_language()` advances the carousel and resets the chat so replies switch language (see **Language switching — status** above)
- [x] Bilingual STT (EN/ES) — `Transcriber` asks BMO for the active Whisper language code via the same double dispatch (`stt_language`), resolved per `transcribe()` call so a switch lands on the next utterance (`bmo/stt/transcribe.py`)
- [x] TTS voice language switch — `Speaker` asks BMO for the active language's `tts_voice` tag per `say()`, asks its own `VoiceCatalog` (built at construction) for the matching `.onnx` in `resources/voices/`, and caches each loaded voice; a switch lands on the next spoken reply (falls back to the default voice if that language has no installed voice — install a second voice model to hear the switch)
- [ ] Language switch trigger — `switch_language()` exists but nothing calls it yet (planned: a GPIO button on the Pi)

## Key dependencies

| Package | Role |
|---|---|
| `openwakeword` | Always-on wake word detection |
| `onnxruntime` | ONNX inference backend for openwakeword (use instead of tflite on Linux/Pi) |
| `pyaudio` | Microphone capture and speaker output |
| `numpy` | Audio chunk processing |
| `groq` | Whisper STT + LLM completions |
| `piper-tts` | Local neural TTS |
| `requests` | HTTP calls for Tavily search |
| `python-dotenv` | Load `.env` API keys |

## Wake word notes

- Default model: custom `hey_bmo_v2.0.onnx`, shipped in `resources/` at the repo root
- openwakeword chunk size must be **1280 samples at 16kHz** (80ms) — do not change `CHUNK` in `capture.py`
- Silence threshold (RMS `500` in `capture.py`) may need tuning per microphone
- To swap models, pass a `model_path` to `WakeWordDetector(...)` in `main.py`
- Call `detector.reset()` after handling a command — otherwise leftover audio (the
  wake word + the spoken command) lingers in the model's buffer and re-triggers
  detection on the next loop
