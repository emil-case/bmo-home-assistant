# BMO 🎮

A local-first voice assistant inspired by **BMO** from *Adventure Time*, built to run on
cheap Raspberry Pi hardware with minimal API spend.

[![tests](https://github.com/emil-case/bmo-home-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/emil-case/bmo-home-assistant/actions/workflows/tests.yml)

> **Guiding principle:** run everything possible locally; only hit external APIs for
> what's too heavy for the hardware.

## How it works

BMO listens for a wake word, records what you say, understands it, and talks back:

```
wake word → audio capture → STT → LLM (tool-use loop) → TTS → audio playback
```

The LLM step is a **recursive tool-use loop**, not a single pass:

```
while the model returns a tool call:
    run the tool
    feed the result back to the model
speak the final text response
```

This lets BMO chain tools (e.g. web search → summarize) before answering. Conversation
history is **session-scoped** — kept in memory so BMO stays coherent within a session,
and wiped on restart.

## Services & tech choices

| Component   | Choice              | Why |
|-------------|---------------------|-----|
| Wake word   | OpenWakeWord        | Free, fully local, runs on a Pi Zero 2W |
| STT         | Groq Whisper API    | ~300ms transcription, generous free tier |
| LLM         | Groq + Llama 3.3 70B | Free tier, tool calling, 1–2s responses |
| TTS         | Piper TTS           | Designed for edge devices, very low RAM |
| Web search  | Brave Search API    | Free tier, 2000 queries/month |

Only **STT** and **LLM** (both via Groq) and **web search** (Brave) leave the device.
Wake word and TTS run entirely locally.

## Project status

This is a work in progress. Here's how far it's come:

- [x] **Wake word detection** + audio capture pipeline (`bmo/audio/`) — custom
  "Hey BMO" model
- [x] **Acknowledgement beep** when the wake word fires (placeholder for a BMO voice clip)
- [x] **Half-duplex mic** — capture pauses while BMO handles a command, then resumes
  (so it won't hear itself once TTS lands)
- [x] **STT** — captured audio → Groq Whisper → transcript (`bmo/stt/`)
- [x] **LLM request** + session history (`bmo/llm/`) — *basic request done; recursive
  tool-use loop still pending*
- [x] **Test suite + CI** — pytest with mocked hardware/APIs, runs on GitHub Actions
- [ ] **LLM tool-use loop** — handle tool calls recursively
- [ ] **Brave web search tool**
- [ ] **TTS** — final text → Piper → audio playback

The current `main.py` loop: detect wake word → beep → record until silence → transcribe
→ send to the LLM → print the reply. A `TODO` marks where the tool-use loop and TTS will
slot in.

## Setup

Develop on **Linux** (the target is Raspberry Pi OS, Debian-based). This avoids
Windows-specific package pain (`tflite-runtime` has no Windows wheels, `pyaudio` needs
workarounds).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your API keys
```

Required env vars in `.env`:

- `GROQ_API_KEY` — used for both STT (Whisper) and LLM inference
- `BRAVE_API_KEY` — used for the web search tool

Piper TTS needs a voice model (`.onnx` + `.onnx.json`) placed under
`bmo/audio/voices/` (gitignored).

Then run:

```bash
python main.py
```

## Project structure

```
main.py              # entry point, main loop
bmo/
  audio/
    capture.py       # PyAudio stream; read_chunk(), record_until_silence(), pause()/resume()
    wake_word.py     # wraps openwakeword Model; process(chunk) -> bool
    cue.py           # acknowledgement beep
  stt/
    transcribe.py    # PCM -> in-memory WAV -> Groq Whisper -> text
  llm/
    chat.py          # session history + Groq Llama request (tool loop TBD)
  tools/             # pluggable LLM tools (e.g. web search)
tests/               # pytest suite (mocks hardware + APIs)
```

## Testing

Tests run with **no hardware and no network** — the mic, speaker, and Groq API are all
mocked, so nothing real is touched:

```bash
pip install -r requirements-dev.txt   # minimal test-only deps
pytest
```

CI runs the suite on every push and PR to `main` via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml). No secrets needed — the
tests never reach Groq. Test names follow the `given_when_then` convention.

## Wake word notes

- Default model: custom `hey_bmo_v2.0.onnx`, shipped in `resources/`
- openwakeword requires **1280-sample chunks at 16kHz** (80ms) — don't change `CHUNK`
- The silence threshold (RMS) in `capture.py` may need tuning per microphone
- After handling a command, the detector is reset so leftover audio doesn't re-trigger it
