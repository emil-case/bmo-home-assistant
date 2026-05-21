# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BMO is a local voice assistant inspired by the character from Adventure Time. Designed to run on cheap Raspberry Pi hardware with minimal API spend. The full pipeline is:

wake word → audio capture → STT (Groq Whisper) → LLM tool-use loop (Groq Llama) → TTS (Piper) → audio playback

Web search (Brave API) is available as an LLM tool.

## Architecture decisions

**Guiding principle:** run everything possible locally; only hit APIs for what's too heavy.

| Component | Choice | Reason |
|---|---|---|
| Wake word | OpenWakeWord | Free, local, runs on Pi Zero 2W |
| STT | Groq Whisper API | ~300ms transcription, generous free tier |
| LLM | Groq + Llama 3.3 70B | Free tier, tool calling support, 1-2s responses |
| TTS | Piper TTS | Designed for edge devices, very low RAM |
| Web search | Brave Search API | Free tier, 2000 queries/month |

**LLM loop is recursive, not single-pass.** The pattern is:
```
while AI returns a tool call:
    run the tool
    send result back to AI
speak the final text response
```
This allows tool chaining (e.g. search → summarize). Loop until model returns plain text.

**Conversation history is session-scoped.** Keep a `messages[]` list and send it each turn so BMO feels coherent within a session. Wipe on restart.

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
- `BRAVE_API_KEY` — used for web search tool

Piper TTS requires downloading a voice model (`.onnx` + `.onnx.json`) and placing it under `bmo/audio/voices/`. These files are gitignored.

## Testing

```bash
pip install -r requirements-dev.txt   # minimal test-only deps (dev/CI, not the Pi)
pytest
```

`requirements-dev.txt` is the minimal set the tests import (pytest + numpy, pyaudio,
groq, openwakeword, onnxruntime) — deliberately *not* the full runtime
`requirements.txt`. The mocked objects' imports still resolve at module load, so those
packages must be installed; sounddevice/piper-tts/requests/python-dotenv are not.

Tests live in `tests/` and run with no hardware or network — the boundaries are
mocked, so nothing hits a real mic, speaker, or the Groq API:
- **Pure logic** (`_pcm_to_wav`) is tested directly.
- **API clients** (`Transcriber`, `ChatSession`) take an optional `client=` arg so
  tests inject a fake Groq client instead of patching imports.
- **Hardware wrappers** (`AudioCapture`, `WakeWordDetector`) are tested by patching
  `pyaudio` / the openwakeword `Model`.

`main.py`'s loop is glue and isn't unit-tested — verify it manually. Test names follow
the `given_when_then` convention.

CI runs the suite on every push and PR to `main` via `.github/workflows/tests.yml`
(installs PortAudio for pyaudio, then `requirements-dev.txt`, then `pytest`). No
secrets needed — tests never reach Groq.

## Package structure

```
main.py              # entry point, main loop
bmo/
  audio/
    capture.py       # owns PyAudio stream; read_chunk(), record_until_silence(), pause()/resume()
    wake_word.py     # wraps openwakeword Model; process(chunk) -> bool
    cue.py           # play_acknowledgement() — beep when wake word fires (placeholder for BMO voice)
  stt/
    transcribe.py    # Transcriber: PCM -> in-memory WAV -> Groq Whisper -> text
  llm/
    chat.py          # ChatSession: session history + Groq Llama request (tool loop TBD)
  tools/             # pluggable LLM tools (e.g. web search)
tests/               # pytest suite (mocks hardware + APIs, see Testing)
```

`README.md` is the public-facing overview (project pitch, status, services, setup). Keep
its status checklist in sync with the **Current state** section below when features land.

## Current state

- [x] Wake word detection + audio capture pipeline (`bmo/audio/`)
- [x] Main loop scaffold (`main.py`) — detects wake word, plays a beep, records until silence
- [x] STT — captured audio sent to Groq Whisper, returns transcript (`bmo/stt/`)
- [x] Half-duplex mic — `pause()`/`resume()` stop capture while BMO handles a command (mic currently paused via a 60s `time.sleep` placeholder)
- [~] LLM tool-use loop — basic Groq Llama request + session history done (`bmo/llm/chat.py`); recursive tool calls TBD
- [ ] Brave search tool
- [ ] TTS — send final text to Piper, play audio
- [ ] Session conversation history

## Key dependencies

| Package | Role |
|---|---|
| `openwakeword` | Always-on wake word detection |
| `onnxruntime` | ONNX inference backend for openwakeword (use instead of tflite on Linux/Pi) |
| `pyaudio` | Microphone capture and speaker output |
| `numpy` | Audio chunk processing |
| `groq` | Whisper STT + LLM completions |
| `piper-tts` | Local neural TTS |
| `requests` | HTTP calls for Brave search |
| `python-dotenv` | Load `.env` API keys |

## Wake word notes

- Default model: custom `hey_bmo_v2.0.onnx`, shipped in `resources/` at the repo root
- openwakeword chunk size must be **1280 samples at 16kHz** (80ms) — do not change `CHUNK` in `capture.py`
- Silence threshold (RMS `500` in `capture.py`) may need tuning per microphone
- To swap models, pass a `model_path` to `WakeWordDetector(...)` in `main.py`
- Call `detector.reset()` after handling a command — otherwise leftover audio (the
  wake word + the spoken command) lingers in the model's buffer and re-triggers
  detection on the next loop
