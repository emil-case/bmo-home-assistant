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
| LLM         | Groq + Llama 4 Scout 17B | Free tier, fast, reliable structured tool calls |
| TTS         | Piper TTS           | Designed for edge devices, very low RAM |
| Web search  | Tavily API          | Free tier (~1000 queries/month), LLM-shaped results with a built-in answer |

Only **STT** and **LLM** (both via Groq) and **web search** (Tavily) leave the device.
Wake word and TTS run entirely locally.

## Project status

This is a work in progress. Here's how far it's come:

- [x] **Wake word detection** + audio capture pipeline (`bmo/audio/`) — custom
  "Hey BMO" model
- [x] **Acknowledgement beep** when the wake word fires (placeholder for a BMO voice clip)
- [x] **Half-duplex mic** — capture pauses while BMO handles a command, then resumes
  (so it won't hear itself once TTS lands)
- [x] **STT** — captured audio → Groq Whisper → transcript (`bmo/stt/`); the language
  is forced (not auto-detected) so accents aren't misheard as another language
- [x] **LLM request** + session history (`bmo/llm/`) — full recursive **tool-use
  loop**: `ChatSession.send()` runs any tool calls and only returns once the model
  produces plain text
- [x] **TTS** — final text → Piper → audio playback (`bmo/tts/`)
- [x] **Test suite + CI** — pytest with mocked hardware/APIs, runs on GitHub Actions
- [x] **Tavily web search tool** (`bmo/tools/tavily_search.py`) — registered as a
  default LLM tool, so BMO can look things up when it doesn't already know
- [x] **`BMO` orchestrator** (`bmo/bmo.py`) — owns the components and the main loop;
  each component is initialized with its `owner` so it can delegate shared decisions
  (e.g. active language) back to BMO. `main.py` is now thin glue
- [x] **Bilingual replies (English / Spanish)** — `LanguageState` (`bmo/language/state.py`)
  holds the language; the LLM system prompt's "Always reply in …" clause is chosen per
  language and the rest stays fixed. The states form a carousel — each names its own
  successor via `nextLanguage()` and the boot language comes from `LanguageState.default()`,
  so BMO never references a concrete state. `BMO.switch_language()` advances it and resets the chat
- [x] **Bilingual STT (English / Spanish)** — the `Transcriber` asks BMO for the active
  Whisper language code through the same double dispatch (`stt_language`), resolved on each
  transcription so a switch takes effect on the next utterance
- [x] **Bilingual TTS voice (English / Spanish)** — the `Speaker` asks BMO for the active
  language's voice tag (`tts_voice`) on each reply and plays the matching Piper voice from
  `resources/voices/`, caching each loaded voice; a switch lands on the next spoken reply.
  Install a per-language voice to hear it — with only one installed, BMO falls back to it
- [ ] **A trigger to switch language** — `switch_language()` exists but isn't wired to
  anything yet (planned: a GPIO button on the Pi)

The main loop lives in the **`BMO` orchestrator** (`bmo/bmo.py`): detect wake word →
beep → record until silence → transcribe → send to the LLM (which runs the recursive
tool-use loop internally, calling tools like Tavily search as needed) → speak the reply
with Piper. `BMO` owns each component and sets itself as their `owner`, so a component
can later delegate shared decisions (e.g. the active language) back to BMO.

### Language switching at a glance

BMO can run in **English or Spanish**. Where a switch lands today:

| Part of the pipeline | Switches? |
|---|---|
| LLM reply language | ✅ yes — on the chat reset |
| STT (Whisper) input language | ✅ yes — on the next utterance |
| TTS output voice | ✅ yes — on the next spoken reply (matches a per-language voice in `resources/voices/`; falls back to the default if that language has none installed) |
| Something that *triggers* the switch | ❌ not yet — `switch_language()` exists but nothing calls it (planned: a GPIO button) |

So the whole **pipeline** — understanding you, choosing the reply language, and BMO's
spoken voice — is bilingual; what's left is a real-world **trigger** to flip languages.

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
- `TAVILY_API_KEY` — used for the web search tool

Piper TTS needs a voice model (`.onnx` + `.onnx.json`) placed under
`resources/voices/` (gitignored). BMO picks the voice per language by matching the
active language's tag against the filename prefix (Piper names voices `en_US-…`,
`es_ES-…`), so installing `en_*` and `es_*` voices gives BMO a distinct voice per
language; with only one installed it uses that for every language.

Then run:

```bash
python main.py
```

## Project structure

```
main.py              # entry point — loads .env, builds BMO, runs it (thin glue)
bmo/
  bmo.py             # BMO orchestrator — owns the components, the main loop, and the language state
  audio/
    capture.py       # PyAudio stream; read_chunk(), record_until_silence(), pause()/resume()
    wake_word.py     # wraps openwakeword Model; process(chunk) -> bool
    cue.py           # acknowledgement beep
  language/
    state.py         # LanguageState (English/Spanish) — supplies the reply-language clause + STT code + TTS voice tag
  stt/
    transcribe.py    # PCM -> in-memory WAV -> Groq Whisper -> text
  llm/
    chat.py          # session history + Groq Llama + recursive tool-use loop
  tts/
    speak.py         # Piper synthesis -> sounddevice playback; voice chosen per language, cached
    voice_catalog.py # owns resources/voices/; resolves a language tag -> .onnx model (the Speaker builds its own)
  tools/
    tavily_search.py # Tavily web search tool exposed to the LLM
tests/               # pytest suite (mocks hardware + APIs)
```

## Testing

Tests run with **no hardware and no network** — the mic, speaker, and Groq API are all
mocked, so nothing real is touched:

```bash
pip install -r requirements-dev.txt   # minimal test-only deps
pytest
```

Two layers: **unit tests** isolate each component (mocking its collaborators), and an
**integration test** (`tests/test_integration.py`) builds a real BMO with a real
`ChatSession` and `LanguageState` — only the mic/STT/TTS/Groq boundaries mocked — to
prove the real delegation chain works (e.g. switching language actually changes the
chat's system prompt).

CI runs the suite on every push and PR to `main` via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml). No secrets needed — the
tests never reach Groq. Test names follow the `given_when_then` convention.

## Wake word notes

- Default model: custom `hey_bmo_v2.0.onnx`, shipped in `resources/`
- openwakeword requires **1280-sample chunks at 16kHz** (80ms) — don't change `CHUNK`
- The silence threshold (RMS) in `capture.py` may need tuning per microphone
- After handling a command, the detector is reset so leftover audio doesn't re-trigger it
