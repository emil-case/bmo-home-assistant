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

## Package structure

```
main.py              # entry point, main loop
bmo/
  audio/
    capture.py       # owns PyAudio stream; read_chunk() + record_until_silence()
    wake_word.py     # wraps openwakeword Model; process(chunk) -> bool
  tools/             # pluggable LLM tools (e.g. web search)
```

## Current state

- [x] Wake word detection + audio capture pipeline (`bmo/audio/`)
- [x] Main loop scaffold (`main.py`) — detects wake word, records until silence, prints duration
- [ ] STT — send captured audio to Groq Whisper, get transcript
- [ ] LLM tool-use loop — send transcript to Groq Llama, handle tool calls recursively
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
