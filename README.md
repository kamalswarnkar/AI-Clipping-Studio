# AI Clipping Studio

Turns one long video into a folder of ready-to-review short-form clips — actual
rendered MP4s, not timestamps.

Everything runs **locally**. No API keys, no cloud calls: speech recognition uses
faster-whisper on your machine, and all reasoning runs through
[Ollama](https://ollama.com).

```
Upload  →  Transcribe  →  Understand  →  Find moments  →  Validate  →  Render  →  Export
```

For each selected moment the app produces an MP4 with burned-in subtitles, plus
an `Info.txt` holding that clip's transcript. Clips keep the source aspect
ratio, so nothing that was on screen gets cropped away; 9:16 is available as an
option.

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
| `SUMMARIZE` | Describes the whole video from its transcript, third person | moderate |
| `ANALYZE_SCENES` | PySceneDetect shot boundaries on the proxy | cheap |
| `ANALYZE_AUDIO` | loudness, silence, emphasis, laughter/applause | cheap |
| `ANALYZE_VISUALS` | OpenCV faces, motion, exposure, and when the source shows captions of its own | cheap |
| `GENERATE_CANDIDATES` | 30–100 windows, multi-factor scored | cheap |
| `LLM_EVALUATE` | LLM judgement, batched and run concurrently | **expensive** |
| `VALIDATE` | context check, filler trim, dedupe, sentence-boundary snap | moderate |
| `DESCRIBE_CLIPS` | Describes each clip against the video's own context | moderate |
| `RENDER` | trim → optional 9:16 reframe → loudnorm → burn captions → H.264 | moderate |

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

Steps 3 and 5 are optional: the launcher below does both if you skip them.

---

## Running it

Double-click **`run.bat`** in the project folder (or `./run.sh` on
macOS/Linux). That is the whole thing — no terminal commands, no separate
windows to keep open.

```bash
run.bat
```

You never open the Python files yourself, and you do not need an editor or IDE:
`run.bat` finds the Python inside the project's own `venv` folder and runs
[`scripts/launch.py`](scripts/launch.py) with it. A console window opens and
stays open while the app runs — that window *is* the app.

It checks the machine and fills in whatever is missing, in order:

| | |
|---|---|
| **Ollama** | started in the background if it is not already listening |
| **Models** | verified against `ollama list`, and offered as a download if absent |
| **Interface** | `npm install` + `npm run build` on the first run only |
| **Server** | started in the foreground, so `Ctrl+C` stops it |
| **Browser** | opened once the app actually answers, not before |

Launch it a second time while it is already running and it just opens the
browser again instead of failing on a busy port. Ollama is deliberately left
running on exit: it may have been started by its own tray app, and a warm model
makes the next launch much faster.

Pass `--no-browser` to skip the browser. Anything that is genuinely missing
— Ollama not installed, no `venv`, dependencies absent — stops the launcher
with the exact command that fixes it.

### Stopping it

Press **Ctrl+C** in the window `run.bat` opened, or just close that window.
Either one stops the server and every render underneath it.

If that window is gone, or you started the app some other way, double-click
**`stop.bat`**:

```bash
stop.bat
```

It stops the launcher recorded in `data/.launcher.pid`, which takes the server
with it and closes the window `run.bat` opened. If there is no launcher — the
server was started some other way — it falls back to whatever is listening on
the app's port, and refuses to touch it if that turns out to be an unrelated
program.

Add `--ollama` to stop Ollama as well. Without it Ollama is left running, which
is usually what you want: a warm model makes the next launch much faster.

To develop the UI with hot reload, run `npm run dev` in `frontend/` (port 5173);
it proxies `/api` to the backend. To start the server on its own, without the
launcher's checks:

```bash
venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```

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
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Only used when `VISION_ENABLED=true` |
| `VISION_ENABLED` | `false` | Vision costs ~25s per candidate. Conflict is detected far more cheaply from audio + speaker turns |
| `WHISPER_MODEL` | `base` | ~2.8× faster than `small` with near-identical output. A bigger model does **not** reliably fix proper nouns — see below |
| `WHISPER_VOCABULARY` | *(empty)* | Names and terms recurring across your videos, comma separated. Per-video names go in the upload screen instead |
| `LLM_PARALLEL` | `3` | Concurrent Ollama requests |
| `OPENING_WEIGHT` | `0.22` | How much the first 3 seconds count toward selection |
| `CONFLICT_WEIGHT` | `0.18` | Weight for shouting / interruption / confrontation |
| `RENDER_WORKERS` | `3` | Parallel clip renders |
| `SUBTITLE_FONT_SIZE` | `120` | Caption size for a 1920-tall frame, scaled to the real output. About 6% of frame height |
| `RENDER_WIDTH` / `RENDER_HEIGHT` | `1080` / `1920` | Only used when a project asks for the 9:16 crop |
| `REMOVE_SOURCE_SUBTITLES` | `true` | Detect captions already burned into the source. The 9:16 crop removes that band; at the original ratio the app's own captions are placed above it |
| `CLIP_MIN_DURATION` / `CLIP_MAX_DURATION` | `20` / `60` | Also settable per project in the UI |

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
├── About.txt            what the source video is
├── Clip_01/
│   ├── Clip_01.mp4      subtitled, loudness-normalised, source aspect ratio
│   ├── Context.txt      what the clip is, in the third person
│   └── Info.txt         the clip transcript, nothing else
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
- **Fewer clips is a valid answer.** If only 6 moments are strong, you get 6 and
  a message saying so. The app never pads the list to hit a requested count.
- **Openings are protected.** A clip may not start mid-sentence or on a greeting,
  logistics or channel intro, because the first three seconds decide whether a
  short-form clip is watched at all.

For political or contested material the system transcribes, clips and describes
faithfully. It does not do voter targeting, demographic persuasion, or
optimisation of political messaging.

---

## Where a clip starts and stops

A clip that opens halfway through a sentence, or stops before the thought
lands, is unusable however good the moment was. Two guards run after the
boundaries are otherwise settled, because the duration clamp can reintroduce
both problems after the snapping is done:

- **A dependent opening is pushed back.** "Well because it was one of the
  founding principles..." is an answer to a question the viewer never heard.
  Discourse markers are stripped before the test, so `well`, `yeah` and `okay`
  do not hide the `because` behind them, and the start moves back to the
  sentence that stands on its own.
- **An unfinished ending is completed.** The end extends to the next full stop
  when there is room inside the maximum duration. Failing that it pulls back to
  the previous one — fast speech can run a long way without Whisper punctuating
  anything. Failing that too, trailing connectives are dropped, so a clip ends
  on "...they're making it illegal to expose the fraud" rather than on
  "...expose the fraud But".

Only punctuation counts as a sentence ending for these repairs. Whisper ends a
*segment* wherever its decoding window ran out, which is frequently mid-clause.

## Context

Each clip is described in the third person, and the description is written
against a description of the whole video rather than the clip alone. That is
what lets a clip's notes name the bill, the city or the person the clip itself
only calls "it" or "he".

`SUMMARIZE` builds the video-level description from the full transcript. Long
transcripts do not fit in a local model's context window, so it is map-reduce:
each section is summarised on its own, then the section summaries are
synthesised into one description — setting, participants, subject, what
happens, and the terms that recur.

`DESCRIBE_CLIPS` then describes each clip against that. The result appears on
the clip screen, and in the export as `Context.txt` beside each clip, with the
video-level description as `About.txt` at the top of the folder. `Info.txt`
stays the transcript verbatim: one file is what was said, the other is what it
was about.

Both stages are optional. If Ollama is unavailable the clips are still cut
correctly, just undescribed.

## Getting names right

Speech recognition mishears names it has no reason to know. On the test footage
"Stop Nick Shirley Act" came out as *"Stop and make Shirley act"*, and a larger
model did not save it — `small` produced *"stop and make surely act"* for 2.4×
the transcription time.

What does fix it is telling the recogniser the words exist. Whisper accepts
biasing terms, and with `Nick Shirley, Stop Nick Shirley Act` supplied, `base`
transcribes the line correctly with no speed cost at all.

So there are two places to put them:

- **Names and terms in this video** on the upload screen, for names specific to
  one video.
- `WHISPER_VOCABULARY` in `.env`, for names that recur across everything you
  cut — a channel's regular subjects, your own brand names.

Both are used together. Add a term whenever you see a name come out wrong;
correcting the transcript fixes the burned-in captions and `Info.txt` at once,
since both come from it.

## Aspect ratio

Clips keep the shape of the source. A 16:9 conversation stays 16:9, so two
people talking are still two people talking.

The 9:16 crop is available per project (**Crop to vertical 9:16** on the upload
screen) and does what the name says: it keeps the full height and discards most
of the width. Smart reframing tracks faces to choose *where* that window sits,
but on a wide two-shot there is no window that holds both speakers — something
has to go. That trade is worth making for a short-form platform and not worth
making otherwise, which is why it is off by default.

Captions scale with the frame rather than assuming 1080×1920. `SUBTITLE_FONT_SIZE`
is given for a 1920-tall frame and scaled to whatever is actually rendered, so
captions keep the same share of the picture at any resolution. Line length is
then derived from that size and the frame width — necessary rather than tidy,
since ASS `WrapStyle: 2` means libass breaks lines only where the app puts a
break, so a line allowed to be too long runs off the edge instead of wrapping.

## Working with sources that already have captions

Plenty of footage arrives with captions already burned in. `ANALYZE_VISUALS`
looks for the signature of outlined caption text (a very bright pixel with a
very dark pixel a few pixels away, centred horizontally, low in the frame).

What happens next depends on the output shape. Cropping to 9:16 removes the
band outright — it would otherwise be sliced down the middle — so only one set
of captions survives.

At the source aspect ratio nothing is cut, so the source keeps its own
captions and two more things follow. `ANALYZE_VISUALS` also records, per
sampled frame, whether captions are on screen *right now*, because burned-in
captions usually cover part of a video rather than all of it. Where the source
is showing its own, the app drops its cues rather than stacking a second set on
top; everywhere else it captions normally, positioned above the band. A single
on/off flag would have been wrong either way: captions lost on the uncaptioned
parts, or doubled up on the captioned ones.

Sources without burned-in captions are left untouched. Set
`REMOVE_SOURCE_SUBTITLES=false` to skip all of this.

---

## Performance

Measured on an i5-14400F with an RX 6650 XT (8 GB, ROCm), on a **16.9-minute**
source producing **8 clips**:

| Stage | Time |
|---|---|
| Ingest + proxy | 22 s |
| Transcription (`base`, CPU int8) | 41 s |
| Diarization | 1 s |
| Video summary | 24 s |
| Scene detection | 9 s |
| Audio analysis | 1 s |
| Visual analysis (1013 frames) | 25 s |
| Candidate generation | 3 s |
| LLM evaluation (32 candidates) | 85 s |
| Context validation | 18 s |
| Clip descriptions (8 clips) | 23 s |
| Rendering (8 clips) | 36 s |
| **Total** | **4.8 min** |

These are stage timings from one real run, not a projection. Expect the LLM
stages to move around: evaluation alone has been measured anywhere from 74 s to
185 s on the same source, depending on what else is holding the GPU and whether
the model was already warm.

LLM evaluation is the largest stage. Lower `CANDIDATE_LLM_MAX` to trade some
selection quality for speed.

If you enable `VISION_ENABLED=true`, expect roughly +25 s per analysed
candidate. The text model (4.7 GB) and vision model (6 GB) do not both fit in
8 GB of VRAM, so the pipeline runs all vision calls in one batch before any text
calls, avoiding repeated model swaps.

---

## Development

```bash
venv\Scripts\python.exe -m pytest backend/tests -q      # unit tests
python scripts/make_test_video.py                       # synthetic test fixture
python scripts/run_e2e.py [video.mp4] --clips 10        # full pipeline, no HTTP
python scripts/run_e2e.py sample.mp4 --vocabulary "Nick Shirley"
```

`make_test_video.py` builds a ~4 minute interview using Windows SAPI voices,
with scene changes, audience reactions and deliberately planted strong moments
among filler — useful for checking selection quality against known ground truth.

### Layout

```
backend/app/
├── api/routes/     HTTP endpoints
├── ai/             providers, prompts, clip evaluation
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
scripts/
├── launch.py       what run.bat runs: Ollama, build, server, browser
├── stop.py         what stop.bat runs
├── run_e2e.py      full pipeline without the HTTP layer
├── make_test_video.py
└── setup_ffmpeg.py
```

Analysis artifacts are written as JSON under
`data/projects/<id>/analysis/`, so any stage can be inspected or resumed
without rerunning the ones before it.
