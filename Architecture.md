# AI Short-Form Clipping Studio --- Architecture

## 1. Project Goal

Build a desktop/web application that accepts a single long-form video as
the primary input and automatically turns it into a collection of
high-quality, self-contained short-form videos.

The user should **not** need to provide: - a transcript - speaker
notes - timestamps - manually selected moments - manually downloaded
clips

The system should analyze the uploaded video itself, identify strong
self-contained moments, physically render the selected clips, generate
multiple hook options and an appropriate caption for every clip, and
export everything in an organized folder structure.

The intended workflow is:

**Upload one long video → Analyze → Find candidate moments → Validate
context → Render 10+ clips → Generate hooks/captions → Review → Export**

The application is intended for short-form content production,
particularly interviews, podcasts, debates, speeches, commentary, public
interactions, and other long-form spoken-video content.

For political content, the system must remain accurate and
non-misleading. It may identify compelling moments and generate faithful
descriptive copy, but must not fabricate claims, distort context, or
optimize political messaging through voter targeting or demographic
persuasion.

------------------------------------------------------------------------

## 2. Core Product Requirements

### Input

Required: - One video file.

Supported input should ideally include: - MP4 - MOV - MKV - WebM

Optional future inputs: - campaign/brand brief - content rules -
preferred duration - platform - visual style

The application itself generates the transcript and other analysis data.

### Output

The application must generate actual video files, not merely timestamps.

For each selected clip: - 10--60 seconds by default - vertical 9:16
output - original audio preserved and normalized - optional/built-in
subtitles - intelligent speaker framing/reframing - multiple hook
options - one finished caption - metadata

Recommended export structure:

``` text
<SourceVideoName>/
├── Clip_01/
│   ├── Clip_01.mp4
│   ├── Hooks.txt
│   ├── Caption.txt
│   └── Info.txt
│
├── Clip_02/
│   ├── Clip_02.mp4
│   ├── Hooks.txt
│   ├── Caption.txt
│   └── Info.txt
│
├── Clip_03/
│   ├── Clip_03.mp4
│   ├── Hooks.txt
│   ├── Caption.txt
│   └── Info.txt
│
└── ...
```

`Info.txt` is optional but recommended. It can contain: - source
filename - source start/end time - duration - detected speaker(s) -
topic - transcript - analysis notes - generation metadata

------------------------------------------------------------------------

# 3. High-Level Architecture

``` text
                         ┌──────────────────────┐
                         │      USER UPLOAD     │
                         │      Long Video      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    INGESTION LAYER   │
                         │ validation + metadata│
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             ┌────────────┐  ┌────────────┐  ┌────────────┐
             │    AUDIO   │  │    VIDEO   │  │   FRAMES   │
             │ extraction │  │ inspection │  │ sampling   │
             └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
                    │               │                │
                    ▼               ▼                ▼
             ┌────────────┐  ┌────────────┐  ┌────────────┐
             │ TRANSCRIPT │  │   SCENES   │  │   VISUAL   │
             │ + word     │  │ + cuts     │  │  analysis  │
             │ timestamps │  │ + shots    │  │ + OCR      │
             └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
                    │               │                │
                    └───────────────┼────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  CANDIDATE GENERATOR │
                         │  30–100 candidates   │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   LLM ANALYSIS       │
                         │ context + coherence  │
                         │ + quality evaluation │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ VALIDATION + RANKING │
                         │ dedupe + context     │
                         │ + duration + quality │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   FINAL CLIP SET     │
                         │      10–20 clips     │
                         └──────────┬───────────┘
                                    │
                   ┌────────────────┼─────────────────┐
                   ▼                ▼                 ▼
            ┌────────────┐  ┌────────────┐   ┌────────────┐
            │ VIDEO      │  │ COPY       │   │ METADATA   │
            │ RENDERER   │  │ GENERATOR  │   │ GENERATOR  │
            └──────┬─────┘  └──────┬─────┘   └──────┬─────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   OUTPUT MANAGER     │
                         │ folders + ZIP/export │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │    REVIEW UI         │
                         │ preview/edit/select  │
                         └──────────────────────┘
```

------------------------------------------------------------------------

# 4. Processing Pipeline

## Stage 1 --- Video Ingestion

When a user uploads a video:

1.  Validate file format.
2.  Extract:
    -   duration
    -   resolution
    -   frame rate
    -   audio properties
    -   codec
3.  Create a project ID.
4.  Store the source video.
5.  Generate a low-resolution proxy if useful for analysis.
6.  Create a processing job.

The original source must remain untouched.

------------------------------------------------------------------------

## Stage 2 --- Audio Extraction

Extract a clean audio stream for speech and audio analysis.

Use FFmpeg.

Outputs: - WAV/PCM analysis audio - normalized analysis representation -
optional waveform data

------------------------------------------------------------------------

# 5. Automatic Transcription

Use an automatic speech recognition model such as
Whisper/faster-whisper.

The system must generate:

-   full transcript
-   sentence segments
-   word-level timestamps
-   confidence where available
-   detected language

Example internal representation:

``` json
{
  "start": 125.42,
  "end": 131.17,
  "text": "This is the part that people keep missing.",
  "words": [
    {"word": "This", "start": 125.42, "end": 125.71},
    {"word": "is", "start": 125.71, "end": 125.82}
  ]
}
```

The user should never have to supply this manually.

------------------------------------------------------------------------

# 6. Speaker Detection

Speaker diarization should be used where practical.

Goal: - determine speaker turns - identify conversational exchanges -
detect interruptions - associate transcript sections with speakers

Speaker labels can initially be:

``` text
Speaker A
Speaker B
Speaker C
```

Optional future enhancement: - associate a speaker with a visible face
using active-speaker detection.

Do not require user-provided speaker notes.

------------------------------------------------------------------------

# 7. Visual Analysis

Transcript-only clipping is insufficient.

Analyze the actual video for:

-   scene changes
-   shot boundaries
-   camera changes
-   face presence
-   number of people
-   active speaker
-   facial expressions
-   gestures
-   pointing
-   movement
-   crowd reactions
-   audience reactions
-   on-screen text
-   graphics
-   visual interruptions
-   important objects/events
-   composition

Use frame sampling rather than sending every frame to an expensive
vision model.

A practical strategy:

1.  Detect scenes cheaply.
2.  Sample representative frames.
3.  Analyze high-value regions more densely.
4.  Run expensive vision reasoning only on candidate segments.

------------------------------------------------------------------------

# 8. Audio Analysis

Analyze: - loudness - sudden volume changes - laughter - applause -
cheering - silence - interruptions - speech intensity - pauses -
overlapping speech

Audio should be treated as a supporting signal, not the sole reason to
select a clip.

------------------------------------------------------------------------

# 9. Candidate Moment Generation

Do not immediately ask an LLM to select the final 10 clips from an
entire long video.

First generate a large candidate pool.

For example:

``` text
90-minute video
        ↓
30–100 candidate windows
        ↓
context validation
        ↓
LLM evaluation
        ↓
20 strong candidates
        ↓
deduplication
        ↓
10–15 final clips
```

Candidate windows should be created around:

-   complete statements
-   stories
-   questions + answers
-   disagreements
-   surprising statements
-   strong explanations
-   reactions
-   memorable quotes
-   demonstrations
-   emotionally significant moments
-   clear conclusions
-   natural conversational beats

Avoid selecting clips merely because they contain emotionally charged
words.

------------------------------------------------------------------------

# 10. Intelligent Start/End Detection

This is one of the most important parts of the system.

The system should not blindly cut at arbitrary fixed intervals.

It should locate:

### Start boundary

Prefer: - beginning of a sentence - beginning of a thought - beginning
of a question - beginning of a meaningful exchange - enough setup to
understand the moment

Avoid: - mid-word - mid-sentence - excessive dead air - irrelevant setup

### End boundary

Prefer: - completion of thought - punchline/payoff - answer completion -
natural conversational pause

Avoid: - cutting off the conclusion - ending immediately before the
important statement - excessive trailing silence

Word-level transcript timestamps should be used to refine boundaries.

------------------------------------------------------------------------

# 11. Context Validation

Every candidate must be checked for standalone comprehensibility.

Questions:

-   Can a viewer understand what is happening without watching the full
    video?
-   Does the clip require missing context?
-   Is the opening understandable?
-   Is the ending complete?
-   Has an important qualifier been removed?
-   Does trimming change the apparent meaning?
-   Does the clip contain a complete thought?

If necessary, expand the candidate window.

This stage is critical for avoiding misleading clips.

------------------------------------------------------------------------

# 12. Candidate Scoring

Use a multi-factor score.

Example:

``` text
Content clarity              20%
Standalone completeness      20%
Narrative structure          15%
Visual interest              10%
Audio dynamics               10%
Emotional/reaction value     10%
Information density           5%
Opening strength              5%
Ending/payoff                 5%
```

These weights should be configurable.

Do NOT describe this as a guaranteed "virality score." It is a
content-quality and short-form suitability score.

------------------------------------------------------------------------

# 13. Duplicate / Overlap Prevention

The final set must avoid:

-   duplicate clips
-   near-identical clips
-   heavily overlapping timestamps
-   multiple clips representing the same moment

Use: - timestamp overlap checks - transcript similarity - semantic
similarity - optional perceptual video similarity

If Clip 01 and Clip 08 are essentially the same moment, retain only the
stronger candidate.

------------------------------------------------------------------------

# 14. Final Clip Selection

Default target:

``` text
10–20 clips
```

Do not force poor clips merely to reach an arbitrary number.

If only 8 genuinely strong clips exist, the UI should say so rather than
fabricate weak selections.

Allow user configuration:

``` text
Number of clips:
[ 10 ] [ 15 ] [ 20 ]

Minimum duration:
[ 10 sec ]

Maximum duration:
[ 60 sec ]
```

------------------------------------------------------------------------

# 15. Actual Video Rendering

The system must physically generate the clips.

Use FFmpeg as the rendering backbone.

Pipeline:

``` text
source.mp4
    ↓
trim to selected boundaries
    ↓
audio normalization
    ↓
vertical crop/reframe
    ↓
speaker tracking
    ↓
subtitle rendering
    ↓
encoding
    ↓
Clip_01.mp4
```

No manual timestamp work should be required from the user.

------------------------------------------------------------------------

# 16. Smart 9:16 Reframing

For landscape source video:

-   output 1080×1920 where practical
-   detect faces/active speakers
-   keep primary subject centered
-   dynamically shift crop when the active speaker changes
-   avoid cutting faces
-   preserve important visual information

For two-person conversations: - use a crop that keeps both people where
possible - otherwise use active-speaker framing - optionally support
split-screen in a future version

The original landscape video must remain available.

------------------------------------------------------------------------

# 17. Captions/Subtitles

Generate accurate subtitles from the transcript.

Requirements: - word/sentence timing - readable line length - safe
margins - high contrast - no excessive animation - correct spelling -
speaker changes handled cleanly

Visual styling should remain minimal and configurable.

------------------------------------------------------------------------

# 18. LLM Integration

An LLM is required for semantic reasoning.

The LLM should receive structured evidence such as:

``` text
Candidate transcript
Speaker turns
Candidate timestamps
Scene information
Visual observations
Audio observations
Surrounding transcript context
```

The LLM should NOT be expected to directly edit video.

Its job is to reason about:

-   context
-   completeness
-   meaning
-   clip boundaries
-   quality
-   copy generation
-   validation

The renderer performs the deterministic video operations.

------------------------------------------------------------------------

# 19. Copy Generation

For each final clip generate:

### Hooks

Generate multiple options.

Recommended default: - 13 hook options - one marked as BEST HOOK -
ranked strongest → weakest

Hooks must: - accurately represent the clip - be specific to the
moment - avoid generic filler - avoid unsupported claims - avoid
revealing the entire payoff unnecessarily - remain concise - use proper
spelling and grammar

The system should never invent something that did not occur in the
source.

### Caption

Generate one finished caption based strictly on the clip and verified
context.

For political content: - no fabricated claims - no misleading framing -
no invented quotes - no unsupported accusations - no deceptive omission
of critical qualifiers - no targeting of protected groups - no
voter-targeting or demographic persuasion

The copy-generation layer should support configurable campaign rules in
a future version, but safety and factuality rules always take
precedence.

------------------------------------------------------------------------

# 20. Output Folder Generation

For every selected clip:

``` text
Clip_01/
    Clip_01.mp4
    Hooks.txt
    Caption.txt
    Info.txt
```

### Hooks.txt

Contains:

``` text
BEST HOOK

...

1. Shock
...

2. Curiosity
...

3. Conflict
...

...

13. Controversial
...

RANKING

1. ...
2. ...
3. ...
```

### Caption.txt

Contains only the final caption.

### Info.txt

Contains:

``` text
Source:
Interview_01.mp4

Source Start:
00:32:14

Source End:
00:32:57

Duration:
43 seconds

Topic:
...

Speakers:
Speaker A, Speaker B

Transcript:
...
```

------------------------------------------------------------------------

# 21. UI / UX

The UI should be sleek, minimal, aesthetic, and professional.

## Color system

Maximum 2--3 colors.

Preferred:

``` text
Black
White
One restrained accent
```

Avoid: - gradients - excessive neon - rainbow AI aesthetics - excessive
shadows - clutter - unnecessary decorative elements

The product should feel closer to a premium creative tool than a generic
AI dashboard.

------------------------------------------------------------------------

# 22. Main Screens

## Upload Screen

Minimal:

``` text
CLIPPER

Drop your video here

MP4 · MOV · MKV · WebM

[ Upload Video ]

Number of clips [ 15 ]
Min duration   [ 10s ]
Max duration   [ 60s ]

☑ 9:16
☑ Captions
☑ Smart reframing

[ Generate Clips ]
```

## Processing Screen

Show real progress:

``` text
Uploading                         ✓
Extracting audio                 ✓
Transcribing                     ✓
Detecting scenes                 ✓
Analyzing video                  ✓
Finding moments                  ✓
Validating context               ✓
Generating copy                  ✓
Rendering clips                  ███████░░ 72%
```

## Results Screen

Show clip cards with:

-   thumbnail
-   play button
-   duration
-   clip number
-   processing status

## Clip Detail Screen

Two-column layout:

Left: - video player

Right: - hook options - caption - clip information - copy buttons -
download button

Actions:

``` text
[ Copy Hook ]
[ Copy Caption ]
[ Download Clip ]
[ Regenerate Copy ]
[ Adjust Clip ]
```

## Export

``` text
[ Export Selected ]
[ Export All ]
```

Export should produce the folder structure described above, optionally
packaged as ZIP.

------------------------------------------------------------------------

# 23. Performance Strategy

Long videos can be expensive to analyze.

Use a staged pipeline.

### Cheap processing first

-   FFprobe
-   audio extraction
-   scene detection
-   transcript
-   basic audio analysis

### Expensive AI second

Only send candidate regions to expensive vision/LLM analysis.

This avoids processing every frame with a multimodal model.

### Parallelization

Where possible:

``` text
Transcript ─────────────┐
Scene detection ────────┤
Audio analysis ─────────┼──→ candidate generation
Visual sampling ────────┘
```

Then:

``` text
candidate generation
        ↓
LLM evaluation
        ↓
rendering
```

Rendering of independent clips should be parallelized where hardware
permits.

------------------------------------------------------------------------

# 24. Suggested Technology Stack

The exact stack can be changed after implementation constraints are
known.

## Frontend

Recommended: - Next.js / React - TypeScript - Tailwind CSS or a minimal
custom CSS system

## Backend

Recommended: - Python - FastAPI - background job system

Python is preferred for the AI/video pipeline because of its ecosystem.

## Video

-   FFmpeg
-   OpenCV
-   PySceneDetect where useful

## Speech

-   faster-whisper / Whisper

## Speaker diarization

-   pyannote.audio or equivalent

## Vision

Potential components: - OpenCV - face/person detection - tracking -
OCR - multimodal LLM/vision model for higher-level reasoning

## AI reasoning

Use an LLM API with structured JSON output.

Keep the LLM provider behind an abstraction layer so the provider/model
can be changed without rewriting the application.

## Storage

MVP: - local filesystem

Production: - S3-compatible object storage such as Cloudflare R2 or AWS
S3

## Database

MVP: - SQLite

Production: - PostgreSQL

------------------------------------------------------------------------

# 25. Data Model

Core entities:

``` text
Project
 ├── SourceVideo
 ├── Transcript
 ├── Scenes
 ├── Speakers
 ├── Candidates
 ├── FinalClips
 │    ├── RenderedVideo
 │    ├── Hooks
 │    ├── Caption
 │    └── Metadata
 └── Export
```

Example FinalClip:

``` json
{
  "id": "clip_01",
  "source_id": "video_001",
  "start": 1934.2,
  "end": 1977.8,
  "duration": 43.6,
  "topic": "...",
  "speakers": ["speaker_a"],
  "transcript": "...",
  "score": 0.91,
  "render_status": "completed",
  "video_path": "...",
  "hooks_path": "...",
  "caption_path": "..."
}
```

------------------------------------------------------------------------

# 26. API Shape

Potential endpoints:

``` text
POST   /projects
POST   /projects/{id}/upload
POST   /projects/{id}/process
GET    /projects/{id}/status
GET    /projects/{id}/clips
GET    /projects/{id}/clips/{clip_id}
POST   /projects/{id}/clips/{clip_id}/regenerate-copy
POST   /projects/{id}/clips/{clip_id}/rerender
GET    /projects/{id}/export
```

Use asynchronous processing for long-running jobs.

------------------------------------------------------------------------

# 27. Job Architecture

Use a queue for:

``` text
INGEST
TRANSCRIBE
DIARIZE
ANALYZE_SCENES
ANALYZE_AUDIO
ANALYZE_VISUALS
GENERATE_CANDIDATES
LLM_EVALUATE
VALIDATE
RENDER
GENERATE_COPY
PACKAGE_EXPORT
```

Each job should be resumable.

If rendering Clip 09 fails, the entire project should not need to
restart.

------------------------------------------------------------------------

# 28. Error Handling

Handle:

-   unsupported file
-   corrupt video
-   missing audio
-   speech recognition failure
-   LLM timeout
-   API rate limit
-   rendering failure
-   insufficient candidate moments
-   low-confidence transcript
-   storage failure

The UI should expose useful errors instead of generic:

> Something went wrong.

------------------------------------------------------------------------

# 29. No Fine-Tuning Requirement for V1

Do not build the first version around fine-tuning.

Use:

-   strong prompting
-   structured outputs
-   deterministic validators
-   candidate scoring
-   multimodal evidence
-   user feedback

Fine-tuning or a custom ranking model can be evaluated later after
collecting sufficient examples of: - selected clips - rejected clips -
manually adjusted boundaries - preferred hooks - preferred captions -
campaign acceptance/rejection where legally and operationally
appropriate

The first goal is a reliable end-to-end pipeline.

------------------------------------------------------------------------

# 30. Feedback Loop

Future versions should learn from user behavior.

Track:

``` text
Clip selected
Clip rejected
Clip boundary adjusted
Hook selected
Hook rejected
Caption edited
Clip downloaded
```

This can improve ranking and copy generation without requiring immediate
model fine-tuning.

------------------------------------------------------------------------

# 31. Security and Privacy

-   Never expose uploaded files publicly by default.
-   Use project-scoped storage.
-   Validate filenames.
-   Sanitize paths.
-   Do not execute arbitrary uploaded files.
-   Clean temporary files after configurable retention.
-   Protect API keys server-side.
-   Never place provider API keys in frontend code.

------------------------------------------------------------------------

# 32. MVP Definition

The first working version is complete when a user can:

1.  Open the application.
2.  Upload one long video.
3.  Start processing.
4.  Wait while the application automatically:
    -   extracts audio
    -   transcribes
    -   detects scenes
    -   analyzes visual/audio signals
    -   finds candidate moments
    -   validates them
    -   selects final clips
    -   physically renders clips
    -   generates hooks
    -   generates captions
5.  View generated clips.
6.  Preview each clip.
7.  Read and copy hooks/captions.
8.  Download individual clips.
9.  Export all clips in organized folders.

The user must not need to manually: - transcribe - find timestamps - cut
videos - reframe videos - create subtitles - write hooks - write
captions - organize output folders

------------------------------------------------------------------------

# 33. Guiding Principle

The application should behave like a **creative production assistant**,
not a black-box generator.

It should automate the tedious work while keeping the user in control of
the final selection.

The most important product promise is:

> **One long video in. A folder of ready-to-review short-form clips
> out.**
