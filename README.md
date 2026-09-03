# AI Clipping Studio

Turns one long video into a folder of ready-to-review short-form clips — actual
rendered MP4s, not timestamps.

Everything runs **locally**. No API keys, no cloud calls: speech recognition uses
faster-whisper on your machine, and all reasoning runs through
[Ollama](https://ollama.com).

```
Upload  →  Transcribe  →  Find moments  →  Validate  →  Render  →  Hooks + Captions  →  Export
```

For each selected moment the app produces a vertical 1080×1920 MP4 with burned-in
captions, 13 ranked hook options, one finished caption, and a metadata file.

---

## What it actually does

The user uploads a video and nothing else. The system generates its own
transcript, speaker turns, scene list and audio events, proposes candidate
moments, has a local LLM judge which ones stand on their own, cuts them with
FFmpeg, and packages the results.

**The design rule is: the AI decides *what*, code decides *how*.** The model
never touches the filesystem or the video. It returns JSON, that JSON is
validated and clamped, and a deterministic renderer performs every cut.

### Pipeline

| Stage | What it does | Cost |
|---|---|---|
| `INGEST` | ffprobe metadata, builds a 480p/8fps analysis proxy | cheap |
| `EXTRACT_AUDIO` | 16 kHz mono WAV for ASR and analysis | cheap |
| `TRANSCRIBE` | faster-whisper with **word-level** timestamps | moderate |
| `DIARIZE` | MFCC embeddings + agglomerative clustering → Speaker A/B/C | cheap |
| `ANALYZE_SCENES` | PySceneDetect shot boundaries on the proxy | cheap |
| `ANALYZE_AUDIO` | loudness, silence, emphasis, laughter/applause | cheap |
| `ANALYZE_VISUALS` | OpenCV faces, motion, exposure across every sampled frame | cheap |
| `GENERATE_CANDIDATES` | 30–100 windows, multi-factor scored | cheap |
| `LLM_EVALUATE` | vision montages on the top candidates, then LLM judgement | **expensive** |
| `VALIDATE` | context check, filler trim, dedupe, word-boundary snap | moderate |
| `GENERATE_COPY` | 13 hooks + caption, fact-checked against the transcript | moderate |
| `RENDER` | trim → smart 9:16 crop → loudnorm → burn captions → H.264 | moderate |

Only the strongest candidates reach the expensive stages. That staging is what
keeps a 90-minute video tractable on a desktop GPU.

---

## Requirements

- **Python 3.11+** (developed on 3.13)
- **Node 18+** (only to build the UI)
- **FFmpeg** — auto-detected, or install it locally with the bundled script
- **Ollama** running, with two models pulled

No GPU is required, but one helps a great deal.

---

## Setup

```bash
# 1. Python environment
python -m venv venv
venv\Scripts\python.exe -m pip install -r backend/requirements.txt

# 2. FFmpeg (skip if it is already on your PATH)
python scripts/setup_ffmpeg.py

# 3. Local models
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5vl:7b

# 4. Configuration
copy .env.example .env

# 5. Build the UI
cd frontend && npm install && npm run build && cd ..
```

Then run it:

```bash
venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```

Open <http://127.0.0.1:8000>.

To develop the UI with hot reload, run `npm run dev` in `frontend/` (port 5173);
it proxies `/api` to the backend.

### Check your setup

```bash
curl http://127.0.0.1:8000/api/health
```

This reports FFmpeg and every provider, with the exact remedy for anything
missing. The upload screen shows the same information, so a broken setup is
visible *before* you wait through an upload.

---

## Configuration

Everything lives in `.env` (see `.env.example` for the annotated list).

The settings you are most likely to change:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_LLM_MODEL` | `qwen2.5:7b-instruct` | Any instruct model with good JSON adherence |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Set `VISION_ENABLED=false` to skip vision entirely |
| `WHISPER_MODEL` | `small` | `base` is ~3× faster, `medium` more accurate |
| `VISION_MAX_CANDIDATES` | `20` | The main speed/quality lever — vision dominates runtime |
| `RENDER_WORKERS` | `2` | Parallel clip renders |
| `CLIP_MIN_DURATION` / `CLIP_MAX_DURATION` | `10` / `60` | Also settable per project in the UI |

Scoring weights live in `backend/config/scoring.json` and can be tuned without
touching code.

### Swapping providers

Providers sit behind protocols in `backend/app/ai/base.py`
(`TranscriptionProvider`, `DiarizationProvider`, `LLMProvider`,
`VisionProvider`). Adding a backend means writing one class and changing one
environment variable — no pipeline changes. Set any provider to `null` to
disable it; the pipeline degrades with a visible warning rather than failing
silently.

---

## Output

`Export all` produces exactly this structure, as a ZIP:

```
<SourceVideoName>/
├── Clip_01/
│   ├── Clip_01.mp4      vertical, captioned, loudness-normalised
│   ├── Hooks.txt        BEST HOOK + 13 categorised hooks + ranking
│   ├── Caption.txt      the finished caption, nothing else
│   └── Info.txt         source range, speakers, topic, transcript, analysis
├── Clip_02/
└── ...
```

---

## Accuracy

Clip selection can mislead by omission, so several guards are deterministic
rather than left to the model:

- **Boundaries** snap to word and sentence timings; a clip can never open or
  close mid-word.
- **Context validation** asks whether trimming changed the speaker's meaning,
  and expands the window when it did.
- **Filler trimming** removes logistics and small-talk openings regardless of
  how the model scored them.
- **Hooks and captions are fact-checked in code** against the clip transcript.
  A hook citing a number or quote that was never said is rejected and replaced.
- **Fewer clips is a valid answer.** If only 6 moments are strong, you get 6 and
  a message saying so. The app never pads the list to hit a requested count.

For political or contested material the system transcribes, clips and describes
faithfully. It does not do voter targeting, demographic persuasion, or
optimisation of political messaging.

---

## Performance

Measured on a Ryzen-class desktop with an RX 6650 XT (8 GB, ROCm):

| Stage | Throughput |
|---|---|
| Transcription (`small`, CPU int8) | ~5–7× realtime |
| Scene + audio + visual analysis | ~15× realtime |
| Vision reasoning | ~25 s per candidate |
| LLM evaluation | ~5 s per candidate |
| Rendering | ~6× realtime per clip, 2 in parallel |

Vision dominates. Lower `VISION_MAX_CANDIDATES`, or set `VISION_ENABLED=false`,
to trade some selection quality for a large speed-up.

The text model (4.7 GB) and vision model (6 GB) do not fit in 8 GB of VRAM
together, so the pipeline deliberately runs all vision calls in one batch before
any text calls, avoiding repeated model swaps.

---

## Development

```bash
venv\Scripts\python.exe -m pytest backend/tests -q      # unit tests
python scripts/make_test_video.py                       # synthetic test fixture
python scripts/run_e2e.py [video.mp4] --clips 10        # full pipeline, no HTTP
```

`make_test_video.py` builds a ~4 minute interview using Windows SAPI voices,
with scene changes, audience reactions and deliberately planted strong moments
among filler — useful for checking selection quality against known ground truth.

### Layout

```
backend/app/
├── api/routes/     HTTP endpoints
├── ai/             providers, prompts, evaluation, copy generation
├── analysis/       candidates, boundaries, scenes, audio, visual, selection
├── video/          ffmpeg wrapper, renderer, reframing, subtitles
├── jobs/           queue, pipeline orchestration, state machine
├── models/         SQLAlchemy rows, domain models, API schemas
├── exporters/      Clip_NN folders and ZIP packaging
└── services/       project storage and path safety
frontend/src/
├── screens/        Upload, Processing, Results, ClipDetail
├── components/     shared UI
└── api/            typed client
```

Analysis artifacts are written as JSON under
`data/projects/<id>/analysis/`, so any stage can be inspected or resumed
without rerunning the ones before it.
