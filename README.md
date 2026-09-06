# AI Clipping Studio

**Turns one long video into a folder of ready-to-publish short-form clips —
actual rendered MP4s, not a list of timestamps.**

Upload a two-hour interview and get back cut, subtitled, loudness-normalised
clips, each with a description of what it is, thirteen ranked hooks and a
finished caption. Nothing leaves the machine: speech recognition runs on the CPU
via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and every piece
of reasoning runs through a local [Ollama](https://ollama.com) model. There are
no API keys, no cloud calls and no per-minute costs.

```
Upload → Transcribe → Understand → Find moments → Validate → Render → Write copy → Export
```

---

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
  - [The pipeline](#the-pipeline)
  - [The design rule](#the-design-rule)
- [The hard parts](#the-hard-parts)
  - [Choosing which moments to cut](#choosing-which-moments-to-cut)
  - [Where a clip starts and stops](#where-a-clip-starts-and-stops)
  - [Understanding the video before describing a clip](#understanding-the-video-before-describing-a-clip)
  - [Hooks and captions](#hooks-and-captions)
  - [Getting names right](#getting-names-right)
  - [Aspect ratio](#aspect-ratio)
  - [Sources that already have captions](#sources-that-already-have-captions)
- [Configuration](#configuration)
- [Performance](#performance)
- [Accuracy and limits](#accuracy-and-limits)
- [Development](#development)

---

## What you get

`Export all` produces exactly this, as a ZIP:

```
<SourceVideoName>/
├── About.txt              what the source video is, in the third person
├── Clip_01/
│   ├── Clip_01.mp4        subtitled, loudness-normalised, source aspect ratio
│   ├── Caption.txt        finished Reel caption: headline, three beats, a
│   │                      forced-choice question, location, five hashtags
│   ├── Context.txt        what this clip is, written against the whole video
│   ├── Hooks.txt          best hook, 13 categorised hooks, final ranking
│   └── Info.txt           the clip transcript, verbatim, nothing else
├── Clip_02/
└── ...
```

Each file answers a different question. `Info.txt` is **what was said**.
`Context.txt` is **what it was about**. `Hooks.txt` and `Caption.txt` are
**how to publish it**.

---

## Quick start

### Requirements

| | |
|---|---|
| **Python 3.11+** | developed on 3.13 |
| **Node 18+** | only to build the interface |
| **FFmpeg** | auto-detected, or installed locally by a bundled script |
| **Ollama** | running, with `qwen2.5:7b-instruct` and `qwen2.5vl:7b` pulled |

No GPU is required, but one helps a great deal. Developed against an 8 GB AMD
card under ROCm.

### Setup

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

# 5. Build the interface
cd frontend && npm install && npm run build && cd ..
```

Steps 3 and 5 are optional — the launcher does both if you skip them.

### Running it

Double-click **`run.bat`** in the project folder, or `./run.sh` on macOS/Linux.
That is the whole procedure: no terminal commands, no separate windows.

```bash
run.bat
```

You never open the Python files yourself and you do not need an editor:
`run.bat` finds the Python inside the project's own `venv` and runs
[`scripts/launch.py`](scripts/launch.py) with it. A console window opens and
stays open while the app runs — that window *is* the app.

The launcher fills in whatever is missing, in dependency order:

| | |
|---|---|
| **Ollama** | started in the background if it is not already listening |
| **Models** | verified, and offered as a download if absent |
| **Interface** | `npm install` + `npm run build` on the first run only |
| **Server** | started in the foreground, so `Ctrl+C` stops it |
| **Browser** | opened once the app actually answers, not before |

Launch it again while it is already running and it just opens the browser rather
than failing on a busy port. Anything genuinely missing — Ollama not installed,
no `venv`, dependencies absent — stops the launcher with the exact command that
fixes it. Pass `--no-browser` to skip the browser.

### Stopping it

Press **Ctrl+C** in the window `run.bat` opened, or close that window. Either
stops the server and every render underneath it.

If that window is gone, double-click **`stop.bat`**. It stops the launcher
recorded in `data/.launcher.pid`, which takes the server with it. If there is no
launcher it falls back to whatever is listening on the app's port — and refuses
to touch it when that turns out to be an unrelated program. Add `--ollama` to
stop Ollama too; without it Ollama is left running, because a warm model makes
the next launch much faster.

### Checking the setup

```bash
curl http://127.0.0.1:8000/api/health
```

Reports FFmpeg and every provider with the exact remedy for anything missing.
The upload screen shows the same information, so a broken setup is visible
*before* you wait through an upload.

To develop the interface with hot reload, run `npm run dev` in `frontend/`
(port 5173); it proxies `/api` to the backend.

---

## How it works

### The pipeline

Fifteen stages, each a resumable job with its own progress. Artifacts are
written to disk as each completes, so a retry resumes from the last good state
instead of re-transcribing.

| Stage | What it does | Cost |
|---|---|---|
| `INGEST` | ffprobe metadata, builds a 480p/8fps analysis proxy | cheap |
| `EXTRACT_AUDIO` | 16 kHz mono WAV for recognition and analysis | cheap |
| `TRANSCRIBE` | faster-whisper with **word-level** timestamps | moderate |
| `DIARIZE` | MFCC embeddings + agglomerative clustering → Speaker A/B/C | cheap |
| `SUMMARIZE` | describes the whole video from its transcript, third person | moderate |
| `REFINE_TRANSCRIPT` | transcribes again, biased toward the names that summary found | moderate |
| `ANALYZE_SCENES` | PySceneDetect shot boundaries on the proxy | cheap |
| `ANALYZE_AUDIO` | loudness, silence, emphasis, laughter, applause | cheap |
| `ANALYZE_VISUALS` | OpenCV faces, motion, exposure, and when the source shows captions of its own | cheap |
| `GENERATE_CANDIDATES` | 30–100 windows, multi-factor scored | cheap |
| `LLM_EVALUATE` | model judgement on the strongest candidates, batched and concurrent | **expensive** |
| `VALIDATE` | context check, filler trim, dedupe, sentence-boundary snap | moderate |
| `DESCRIBE_CLIPS` | describes each clip against the video's own context | moderate |
| `RENDER` | trim → optional 9:16 reframe → loudnorm → burn captions → H.264 | moderate |
| `WRITE_COPY` | watches each rendered clip, then writes hooks and a caption | **expensive** |

Only the strongest candidates reach the expensive stages. That staging is what
keeps a 90-minute video tractable on a desktop GPU.

Stages marked optional in `jobs/states.py` — diarization, the analyses, the
summary, the refine pass, the descriptions and the copy — degrade the result
without killing the run. If Ollama is unavailable you still get correctly cut
clips, just undescribed.

### The design rule

**The AI decides *what*; code decides *how*.**

The model never touches the filesystem or the video. It returns JSON, that JSON
is re-validated and clamped, and deterministic code performs every cut, every
render and every file write. Concretely:

- Proposed boundaries are snapped to real word and sentence timings, and a model
  that proposes a boundary more than 15 s from its candidate window is treated
  as having hallucinated a timestamp.
- Hook and caption *layouts* are assembled in code. The model supplies the
  writing; the emoji headers, separators, numbered categories and ranking block
  are not its problem.
- Every checkable rule is checked: hook length, banned phrases, first person,
  duplicates, hashtag count, caption word count, whether a "location" is really
  a location.

This is why the output stays consistent on a 7B model that is, left alone,
inconsistent.

---

## The hard parts

Everything below exists because the obvious approach failed on real footage.

### Choosing which moments to cut

Candidate windows are generated cheaply from the transcript, audio and visual
signals — 30 to 100 of them, scored on speech density, emphasis, conflict
intensity, face presence, scene stability and how strong the opening three
seconds are. Short-form retention is decided in those three seconds, so they are
scored separately and weighted into selection (`OPENING_WEIGHT`), as is conflict
— shouting, interruption, rapid exchange (`CONFLICT_WEIGHT`).

Only the top `CANDIDATE_LLM_MAX` reach the model. It judges whether each stands
on its own, and its verdicts are then deduplicated by time overlap and text
similarity.

### Where a clip starts and stops

A clip that opens halfway through a sentence, or stops before the thought lands,
is unusable however good the moment was. Two repairs run after the boundaries
are otherwise settled, because the duration clamp can reintroduce both problems
*after* the snapping is done:

- **A dependent opening is pushed back.** "Well because it was one of the
  founding principles…" is an answer to a question the viewer never heard.
  Discourse markers are stripped before the test, so `well`, `yeah` and `okay`
  do not hide the `because` behind them.
- **An unfinished ending is completed.** The end extends to the next full stop
  if there is room inside the maximum duration; failing that it pulls back to
  the previous one; failing that, trailing connectives are dropped, so a clip
  ends on "…making it illegal to expose the fraud" rather than "…expose the
  fraud But".

Only punctuation counts as a sentence ending for these repairs. Whisper ends a
*segment* wherever its decoding window ran out, which is frequently mid-clause.

### Understanding the video before describing a clip

A clip cut out of an hour of footage loses everything that made it legible: who
is speaking, where, and what they are arguing about.

`SUMMARIZE` rebuilds that once, for the whole video. A 17-minute transcript does
not fit in an 8k context window, so it is map-reduce: sections are summarised
concurrently, then the section summaries are synthesised into one description —
setting, participants, subject, what happens, and the terms that recur. Past 14
sections the middle is sampled rather than included whole, so the reduce step
does not hit the same wall.

`DESCRIBE_CLIPS` then describes each clip **against** that description. This is
what lets a clip's notes name the bill, the city or the person that the clip
itself only calls "it" — the transcript alone cannot supply that.

### Hooks and captions

Every clip gets a `Hooks.txt` and a `Caption.txt` in the exact formats specified by
`hooks.txt` and `caption.txt` — the prompt briefs kept in the project root: a best hook,
thirteen hooks by category, a final ranking, and a caption built as
TRIGGER → ESCALATION → DIVISION → DEBATE with a headline, a forced-choice
question, an optional location and exactly five hashtags. Third person
throughout, capped at 190 words.

**The clip itself is the primary source**, as those specs require. `WRITE_COPY`
runs *after* `RENDER`, so the clip exists as a file: the vision model watches
frames spread across it and describes what is visible, and that description
reaches the writing prompts ahead of the transcript. It shows — hooks like
"He interrupted her mid-sentence, grabbing the microphone" describe things the
transcript never mentions.

Watching and writing are kept in separate passes. The text model is 4.7 GB and
the vision model 6 GB, which do not both fit in 8 GB of VRAM: interleaving them
per clip would swap models on every call, so every clip is watched first and
then every clip is written. Within the vision pass clips *are* watched
concurrently — the model is already loaded, so that costs no swap, and it
matters:

| | 4 clips |
|---|---|
| 3 frames, one clip at a time | 143 s |
| 2 frames, one at a time | 84 s |
| **2 frames, 2 at once** (the default) | **29 s** |

Model output is treated as a proposal:

- Hooks outside 4–12 words, carrying a banned generic phrase, written in the
  first person, or duplicating another are rejected. First person is checked
  *outside* quotation marks, so a Quote-Inspired hook can still quote someone
  saying "I".
- **Every hook carries an emoji** — 🚨 👀 😳 😶 and friends, at the start, the
  end, or both. Where the model forgets, one suited to the category is added.
  Emoji are excluded from the word count and from duplicate comparison, because
  they are decoration, not words.
- Missing categories are re-requested, up to three follow-up rounds. A 7B model
  rarely delivers all thirteen in one answer.
- The ranking is repaired if it is not a permutation of the hooks that survived.
- Captions are trimmed by dropping whole sentences, never mid-sentence, and
  forced to exactly five hashtags. A location prints only if it matches
  "City, State" — never guessed.
- Output that drifts out of English is discarded and retried. Small models
  switch language near the token limit, and nothing else would catch it: one
  caption ended "…or was he just trying to draw 注意力". A question that comes
  back without its question mark is treated the same way, because it means the
  answer was cut off mid-word.

### Getting names right

Speech recognition mishears names it has no reason to know, and this matters
more than it looks: the transcript feeds the subtitles, the summary, the clip
descriptions, the hooks and the caption. One wrong name appears in all of them.

Three mechanisms, in order of how much they ask of you.

**1. The video corrects itself.** A name mangled in one sentence is usually
right in another, because Whisper decodes each window independently. The
video-level summary recovers the correct form from the first pass — it produced
`Stop Nick Shirley Act AB-26-24` from a transcript that also contained "the
Stopnic Shirley Act" — and `REFINE_TRANSCRIPT` decodes once more with those
names supplied. On the test video that corrected 728 words with nothing typed
in. Cost is one extra transcription pass; `WHISPER_REFINE_PASS=false` skips it.

A larger model does **not** fix this. Measured on the failing passage:

| Setting | Time | Result |
|---|---|---|
| `base`, greedy | 2.7 s | "Stop and make Shirley act" |
| `small`, beam 5 | 6.8 s | "stop and make surely act" |
| **`base` + the derived names** | **1.8 s** | **"Stop Nick Shirley Act"** ✓ |

**2. Truncations are merged.** Recognition drops leading syllables, so the same
transcript can say "Antifa" three times and "Tifa" once — worse than a plain
mistake, because the summary then reports two entities ("Participants: Nick
Shirley, **Tifa**, … and members of **Antifa**") and every caption inherits a
person who does not exist. A rare token that is the *tail* of a common one,
where both are capitalised mid-sentence, is merged into it. All three conditions
carry weight:

- *Suffix, not prefix* — recognition mishears the **start** of an unknown word.
- *Capitalised mid-sentence* — without this the same rule proposes
  `public → republic`, `rally → literally` and `ever → never`: all real suffix
  matches, all nonsense.
- *The long form at least twice as common*, and not itself a one-off.

Across five real transcripts this fires on exactly one thing. The summary is
rewritten with the same mapping rather than regenerated, since it was built
before the repair ran.

Asking the model to correct names instead was tried and rejected: it turned
"Tifa" into "Tia" — still wrong — and "Nick Shirley" into "Nickolas J. Shirley",
which is invented. A pass that damages names that were already right is worse
than no pass.

**3. You supply them.** For a name the video never says clearly anywhere, use
**Names and terms in this video** on the upload screen, or `WHISPER_VOCABULARY`
in `.env` for names that recur across everything you cut. Both are used, and
alongside the derived ones.

### Aspect ratio

Clips keep the shape of the source. A 16:9 conversation stays 16:9, so two
people talking are still two people talking.

The 9:16 crop is available per project and does what the name says: it keeps the
full height and discards most of the width. Smart reframing tracks faces to
choose *where* that window sits, but on a wide two-shot there is no window that
holds both speakers — on the test footage the crop lost both and centred on a
flagpole. That trade is worth making for a short-form platform and not worth
making otherwise, which is why it is off by default.

Captions scale with the frame rather than assuming 1080×1920. `SUBTITLE_FONT_SIZE`
is given for a 1920-tall frame and scaled to whatever is rendered. Line length is
then derived from that size and the frame width — necessary rather than tidy,
since ASS `WrapStyle: 2` means libass breaks lines only where the app puts a
break, so a line allowed to be too long runs off the edge instead of wrapping.

### Sources that already have captions

Plenty of footage arrives with captions burned in. `ANALYZE_VISUALS` looks for
the signature of outlined caption text — a very bright pixel with a very dark
one a few pixels away, centred horizontally, low in the frame.

What happens next depends on the output shape. Cropping to 9:16 removes the band
outright, since it would otherwise be sliced down the middle. At the source
aspect ratio nothing is cut, so the source keeps its own captions — and the same
pass records, per sampled frame, whether captions are on screen *right now*,
because burned-in captions usually cover part of a video rather than all of it
(76 s of 649 s, in 9 spans, on the test source). Where the source is showing its
own, the app drops its cues rather than stacking a second set on top; everywhere
else it captions normally, positioned above the band.

`REMOVE_SOURCE_SUBTITLES=false` skips all of this.

---

## Configuration

Everything lives in `.env` (see [`.env.example`](.env.example) for the annotated
list). The settings most worth changing:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_LLM_MODEL` | `qwen2.5:7b-instruct` | Any instruct model with good JSON adherence |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Used for watching clips, and for candidate scoring when enabled |
| `WHISPER_MODEL` | `base` | ~2.8× faster than `small` with near-identical output. A bigger model does **not** reliably fix proper nouns |
| `WHISPER_VOCABULARY` | *(empty)* | Names recurring across your videos. Per-video names go in the upload screen |
| `WHISPER_REFINE_PASS` | `true` | Second transcription pass using the names the summary found |
| `LLM_PARALLEL` | `3` | Concurrent Ollama requests |
| `CANDIDATE_LLM_MAX` | `32` | Candidates sent to the model. The main quality/speed lever |
| `CLIP_COUNT_DEFAULT` | `15` | Also set per project in the UI |
| `CLIP_MIN_DURATION` / `CLIP_MAX_DURATION` | `20` / `60` | Also set per project |
| `OPENING_WEIGHT` | `0.22` | How much the first 3 seconds count toward selection |
| `CONFLICT_WEIGHT` | `0.18` | Weight for shouting / interruption / confrontation |
| `COPY_VISION_ENABLED` | `true` | Watch each rendered clip before writing its hooks and caption |
| `COPY_VISION_FRAMES` / `COPY_VISION_PARALLEL` | `2` / `2` | Frames per clip, and clips watched at once |
| `VISION_ENABLED` | `false` | Vision for *candidate scoring*. Costs ~25 s per candidate; conflict is detected far more cheaply from audio |
| `SUBTITLE_FONT_SIZE` | `120` | For a 1920-tall frame, scaled to the real output. About 6% of frame height |
| `RENDER_WIDTH` / `RENDER_HEIGHT` | `1080` / `1920` | Only used when a project asks for the 9:16 crop |
| `REMOVE_SOURCE_SUBTITLES` | `true` | Detect and handle captions already burned into the source |
| `RENDER_WORKERS` | `3` | Parallel clip renders |

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

## Performance

Measured on an i5-14400F with an RX 6650 XT (8 GB, ROCm), on a **16.9-minute**
source producing **9 clips**:

| Stage | Time |
|---|---|
| Ingest + proxy | 22 s |
| Transcription (`base`, CPU int8) | 45 s |
| Diarization | 1 s |
| Video summary | 41 s |
| Name correction (second pass) | 45 s |
| Scene detection | 8 s |
| Audio analysis | 1 s |
| Visual analysis (1013 frames) | 25 s |
| Candidate generation | 2 s |
| LLM evaluation (32 candidates) | 118 s |
| Context validation | 28 s |
| Clip descriptions | 36 s |
| Rendering | 43 s |
| Hooks and captions, clips watched | 500 s |
| **Total** | **15.3 min** |

These are stage timings from one real run, not a projection. Expect the model
stages to move: evaluation alone has measured anywhere from 74 s to 185 s on the
same source, depending on what else is holding the GPU and whether the model was
warm.

**`WRITE_COPY` dominates, and it is linear in clip count** — roughly 7 s to
watch each clip against 48 s to write it. If you need shorter runs, ask for
fewer clips; that is the lever with the least quality cost.

---

## Accuracy and limits

Clip selection can mislead by omission, so several guards are deterministic
rather than left to the model:

- **Boundaries** snap to word and sentence timings; a clip can never open or
  close mid-word.
- **Context validation** asks whether trimming changed the speaker's meaning,
  and expands the window when it did.
- **Filler trimming** removes logistics and small-talk openings regardless of
  how the model scored them.
- **Openings are protected.** A clip may not start mid-sentence or on a
  greeting, logistics or channel intro.
- **Fewer clips is a valid answer.** If only 6 moments are strong you get 6, and
  a message saying so. The app never pads the list to hit a requested count.

For political or contested material the system transcribes, clips and describes
faithfully. It does not do voter targeting, demographic persuasion, or
optimisation of political messaging.

Known limits worth stating plainly:

- Hooks and captions are written for one genre — street interviews,
  confrontations, protests and debates — because that is what the prompts in
  the prompt briefs specify.
- Diarization identifies *distinct voices*, not people. "Speaker C" is an
  internal label and never reaches anything a viewer or publisher reads.
- A name the video never says clearly anywhere needs supplying by hand.

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

The tests cover the logic that must be correct regardless of what any model
says: boundary snapping and repair, truncation merging, duplicate detection,
subtitle timing and wrapping, crop expressions, hook and caption validation, and
path safety.

### Layout

```
backend/app/
├── api/routes/     HTTP endpoints
├── ai/             providers, prompts, evaluation, context, copy
├── analysis/       candidates, boundaries, scenes, audio, visual, terms, selection
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

**Stack:** FastAPI + SQLAlchemy + SQLite (WAL) with an in-process thread-pool job
queue — no Redis, no Celery, because a local single-user tool does not need
them. The interface is a Vite + React + TypeScript + Tailwind SPA served as
static files by the same server.

Analysis artifacts are written as JSON under `data/projects/<id>/analysis/`, so
any stage can be inspected or resumed without rerunning the ones before it.

The database migrates itself on startup: `init_db` adds columns a model has
gained but the file lacks. This is a double-click app — there is no migration
command for a user to run, and without it every query fails with "no such
column" after an upgrade.

---

> **AI can make mistakes.** Everything this produces — the clip boundaries, the
> subtitles, the descriptions, the hooks and the captions — comes from models
> that are wrong often enough to matter. Review every clip and every line of
> copy before publishing it anywhere.
