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
(`EnglishState` / `SpanishState` in `bmo/language/state.py`). A component that needs a
language-dependent value asks its owner (BMO), which forwards the call to the current
state, *passing the component* — so the answer depends on both the component (which
method it calls) and the concrete state (which subclass answers). Today the only such
value is the system prompt's reply-language clause ("Always reply in English/Spanish");
the rest of the prompt is fixed English and lives in `chat.py` (`build_system_prompt`).
`BMO.switch_language()` flips the state and calls `ChatSession.reset()`, which wipes
history and reseeds the prompt in the new language. STT language and TTS voice are
**not** switched yet (see Current state).

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

The main loop now lives in `BMO` (`bmo/bmo.py`), not `main.py`, and is unit-tested
(`tests/test_bmo.py`) by patching the component classes in the `bmo.bmo` namespace —
no hardware or network. `main.py` is just `BMO().run()` glue and isn't tested. Test
names follow the `given_when_then` convention.

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
    state.py         # LanguageState template + EnglishState/SpanishState; supplies the reply-language clause
  stt/
    transcribe.py    # Transcriber: PCM -> in-memory WAV -> Groq Whisper -> text
  llm/
    chat.py          # ChatSession: session history + Groq Llama + recursive tool-use loop; build_system_prompt() + reset()
  tts/
    speak.py         # Speaker: Piper synthesis -> sounddevice playback (voice from resources/voices/)
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
- [x] Bilingual reply language (EN/ES) — `LanguageState` (`bmo/language/state.py`) supplies the system prompt's reply-language clause via double dispatch; `BMO.switch_language()` flips the state and resets the chat so replies switch language
- [ ] STT / TTS language switch — Whisper is still forced to English (`bmo/stt/transcribe.py`) and the Speaker loads the first voice it finds; switching the input/output *audio* language is not wired yet
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
