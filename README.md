# AI Gun — Rick and Morty Edition

Rick's AI gun that can automatically detect and shoot aliens *(gun **NOT** included)*.

A fully local, voice-driven agent: say the wake word, tell it what to do, and an LLM decides
which tools to run — grab a camera frame, detect faces, classify each one as human or alien,
paint crosshairs on the aliens, fire the laser (sound only, we promise), and show you the proof.

Everything runs on your machine. Speech-to-text, text-to-speech, vision models and the LLM
are all local — no cloud inference required.

---

## How it works

```
 mic ──► wake word ──► VAD record ──► Whisper STT ──► "take a picture and shoot the aliens"
 (openWakeWord)      (webrtcvad)    (faster-whisper)             │
                                                                 ▼
                                                  ┌──────────────────────────────┐
                                                  │  LangChain agent (Ollama)    │
                                                  │  plans → calls tools → answers│
                                                  └──────────────┬───────────────┘
                                                                 │
        ┌──────────────────┬───────────────────────┬─────────────┴─────────┐
        ▼                  ▼                       ▼                       ▼
     camera          find_aliens_and_shoot     display_image        time / date /
  (static or        GroundingDINO → boxes      (PIL viewer)       calculator / stop
   OpenCV cam)      CLIP → human vs alien
                    draw crosshair + laser 🔊
                    Piper TTS narrates
```

1. **`inputLoop`** (`tools/audio_io.py`) holds an always-on 16 kHz mic stream. openWakeWord
   scans a rolling window for `hey_jarvis`. A 0.2 s pre-roll buffer is kept so the first
   syllable after the wake word is never clipped.
2. Once woken, webrtcvad decides frame by frame whether you are still talking. Recording
   stops after 1 s of silence, 15 s hard cap, or 3 s of nothing at all (false wake).
3. The utterance goes to `faster-whisper` (`small.en`, int8 on CPU) and the transcript is
   handed to the callback — `llm()` in `main.py`.
4. `llm()` builds a LangChain agent against a local Ollama model and invokes it with the
   transcript. A random "thinking" word from `src/thinking.txt` is printed on a side thread
   so you know it is alive.
5. The agent picks tools. `find_aliens_and_shoot` is the interesting one:
   **GroundingDINO-tiny** does zero-shot detection of `face`, each crop goes through
   **CLIP ViT-B/32** against three prompts (real human face / cartoon-or-alien face / not a
   face), aliens get a red box plus a scoped crosshair on the forehead, humans get a green
   box. The annotated image is written to `.tmp/output/<uuid>.png`, a laser sound fires once
   per alien, and Piper TTS narrates.
6. The agent is instructed to always call `display_image` afterwards, so the result pops open.

If the audio queue backs up more than ~1.2 s (models hogging the CPU), the loop drains the
backlog and resets the wake model instead of drifting further behind.

---

## Requirements

- **Python 3.13+** (pinned via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **[Ollama](https://ollama.com/)** running locally with the model you intend to use
- A microphone; speakers if you want TTS and laser sounds
- A webcam only if you pass `--use_real_camera`
- Windows is the pinned platform (`tool.uv.environments = sys_platform == 'win32'`) and the
  torch index is pinned to **CUDA 13.0** wheels. CUDA is optional — the vision models fall
  back to CPU when `torch.cuda.is_available()` is false.

> `boundingBox.py` loads `arialbd.ttf` by name for labels, which resolves out of the box on
> Windows. On another OS, point that to a font you have.

## Setup

```bash
uv sync                      # install everything, including dev tools
ollama pull granite4.1:3b    # or whichever model you plan to use
```

First run downloads a few things automatically into `.tmp/` and the HuggingFace cache:
the Piper voice (`en_US-hfc_female-medium`), the Whisper `small.en` weights, GroundingDINO-tiny
and CLIP. openWakeWord models may need a one-time fetch:

```bash
uv run python -c "import openwakeword.utils as u; u.download_models()"
```

## Usage

```bash
uv run main                          # voice loop, static test image, audio on
uv run main --use_real_camera        # capture from webcam 0 instead
uv run main --model qwen3:4b         # override the Ollama model
uv run main --disable_audio_output   # mute TTS and laser sounds
uv run main -d                       # LangChain agent debug output
```

Then say **"hey jarvis"**, wait for the `[wake]` line, and give it a command — for example
*"take a picture using the camera, find the alien faces, and shoot them."*
Say "stop" / "quit" / "exit", or hit `ctrl+c`, to shut down.

Run the detector directly on the bundled test image, no voice or LLM involved:

```bash
uv run check
```

### CLI flags

| Flag | Effect |
| --- | --- |
| `image_path` (positional, optional) | Accepted by the parser; the agent currently sources images through the `camera` tool. |
| `--use_real_camera` | Use OpenCV device 0 instead of the static `.tmp/alien.png`. |
| `--model MODEL` | Ollama model tag for the agent. |
| `--disable_audio_output` | Silence Piper TTS and the laser SFX. |
| `-d`, `--debug` | Enable LangChain agent debug logging. |

---

## Configuration

Runtime config lives in `src/rnm/config.py` as module-level constants (imported everywhere as
`C`), and CLI flags mutate it in `process_args()` before anything else runs.

| Name | Meaning |
| --- | --- |
| `STATIC_IMAGE_PATH` | `.tmp/alien.png` — the fake camera feed. Must exist; import fails otherwise. |
| `DYNAMIC_IMAGE_PATH` | `.tmp/tmp.png` — where webcam captures land. |
| `OUTPUT_IMAGE_DIR` | `.tmp/output/` — annotated result images, one UUID-named PNG per shot. |
| `TTS_MODEL_DIR` | `.tmp/tts_model/` — downloaded Piper voice. |
| `THINKING_WORDS` | `src/thinking.txt` — comma-separated loader words. |
| `LASER_PATH` | `src/laser.raw` — raw int16 mono @ 44.1 kHz laser SFX. |
| `model` | Default Ollama model (`granite4.1:3b`). Models known to work are listed in comments. |
| `DEBUG`, `USE_REAL_CAMERA`, `AUDIO_OUTPUT` | Flags set from the CLI. |

Audio tuning constants (wake word, thresholds, silence window, VAD aggressiveness) are at the
top of `src/rnm/tools/audio_io.py`.

`.env` is loaded via `python-dotenv` at startup and is used for optional
[LangSmith](https://smith.langchain.com/) tracing (`LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). It is gitignored — create your own if you want traces.

---

## Project layout

```
src/
  rnm/
    main.py              CLI parsing, agent construction + system prompt, Loader, entrypoint
    config.py            paths, directory creation, runtime flags, model choice
    tools/
      audio_io.py        wake word + VAD + Whisper input loop, PiperTTS, laser SFX
      visual_io.py       camera (static/webcam) and display_image tools
      boundingBox.py     GroundingDINO + CLIP detection, crosshairs, find_aliens_and_shoot tool
      misc_tools.py      time, date, calculator, stop
  thinking.txt           loader vocabulary
  laser.raw              laser sound effect (raw PCM)
tests/                   pytest suite
.tmp/                    models, static image, generated output (mostly gitignored)
```

### Models

| Role | Model |
| --- | --- |
| Wake word | openWakeWord `hey_jarvis` (ONNX) |
| Voice activity | webrtcvad, aggressiveness 2 |
| Speech to text | faster-whisper `small.en`, CPU / int8 |
| Text to speech | Piper `en_US-hfc_female-medium` |
| Object detection | `IDEA-Research/grounding-dino-tiny` |
| Classification | `openai/clip-vit-base-patch32` |
| Agent | Ollama, default `granite4.1:3b` |


## License

MIT — see [LICENSE](LICENSE). © 2026 Deekshant Wadhwa
