"""Generate a synthetic long-form interview video for pipeline testing.

Uses Windows SAPI voices so the test asset contains *real speech* (exercising
transcription and diarization) plus scene changes and audience reactions
(exercising scene and audio analysis).

The script deliberately plants a handful of genuinely strong, self-contained
moments among filler, so clip selection can be graded against ground truth.

Usage:  python scripts/make_test_video.py [--out data/samples/interview.mp4]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.config import resolve_binary  # noqa: E402

SR = 22050

HOST = "Microsoft Zira Desktop"
GUEST = "Microsoft David Desktop"
THIRD = "Microsoft Hazel Desktop"

# (voice, text, gap_after_seconds, reaction)
# reaction: None | "laughter" | "applause"
SCRIPT: list[tuple[str, str, float, str | None]] = [
    (HOST, "Welcome back to the show. Before we start, a quick reminder that we record every Tuesday, and thank you to everyone who sent in questions last week.", 0.6, None),
    (GUEST, "Happy to be here. Thanks for having me.", 0.7, None),
    (HOST, "So, let us get the boring logistics out of the way. You flew in this morning, is that right?", 0.5, None),
    (GUEST, "I did. The flight was delayed about an hour, but it was fine.", 0.9, None),

    # --- Planted strong moment 1: story with setup, tension and payoff --------
    (HOST, "I want to start with the mistake you say cost you two years.", 0.5, None),
    (GUEST, "So here is what happened. For the first two years we spent every dollar we had on getting new people in the door. We were obsessed with reach. We hit a million impressions in one month and we celebrated.", 0.4, None),
    (GUEST, "Then I actually looked at the numbers. Ninety one percent of those people never came back a second time. We had built a bucket with no bottom. We were pouring money straight through it.", 0.4, None),
    (GUEST, "So we stopped all paid acquisition for one full quarter. Everyone told me it was suicide. We spent that entire quarter fixing the first week of the product experience, nothing else.", 0.4, None),
    (GUEST, "Revenue tripled in the next quarter, with zero acquisition spend. That was the moment I understood that retention is not a metric you optimize after growth. Retention is the growth.", 0.5, "applause"),

    (HOST, "That is a strong claim though.", 0.4, None),

    # --- Planted strong moment 2: disagreement and partial concession ---------
    (GUEST, "It is, and I will defend it.", 0.4, None),
    (HOST, "But surely that only works if you already have a product people want. If nobody knows you exist, retention of zero users is still zero.", 0.5, None),
    (GUEST, "That is fair, and I think you are right for the very earliest stage. What I would say is that most companies stay in acquisition mode long after they should have switched. The switch should happen much earlier than people think.", 0.4, None),
    (HOST, "So you are conceding the point for pre launch companies.", 0.3, None),
    (GUEST, "For pre launch, yes. After your first hundred real users, no. After that, if your retention curve does not flatten, more traffic just makes the leak louder.", 0.6, None),

    (HOST, "Let us take a short break and talk about the schedule for the rest of the season.", 0.5, None),
    (HOST, "We have three more episodes recorded and two of them will go out in January.", 0.8, None),

    # --- Planted strong moment 3: crisp question and answer ------------------
    (HOST, "If you could only look at one number every morning, what would it be?", 0.4, None),
    (GUEST, "Week four retention. Not day one, not day seven. Week four. Day one tells you your onboarding is pretty. Week four tells you whether the product actually earned a place in someone's routine.", 0.5, None),
    (HOST, "Why week four specifically?", 0.3, None),
    (GUEST, "Because by week four the novelty is completely gone. Nobody is using you because you are new. If they are still there at week four, they are there because it works.", 0.6, "laughter"),

    (HOST, "Interesting. I had not heard it framed that way.", 0.7, None),
    (GUEST, "Most people have not. It is not a popular thing to say in a fundraising meeting.", 0.9, None),

    (THIRD, "Sorry to interrupt, we need to swap the microphone battery.", 0.8, None),
    (HOST, "Of course, let us pause for a second.", 1.0, None),

    # --- Planted strong moment 4: memorable claim with a concrete number -----
    (HOST, "You mentioned a number earlier that surprised me. Say it again.", 0.4, None),
    (GUEST, "We cut our support tickets by sixty percent without hiring a single person. We did it by fixing the three screens where people got stuck. Three screens. That was the whole project.", 0.4, None),
    (GUEST, "Every company I talk to wants to add a chatbot. Almost none of them have looked at where people actually get confused. The fix is usually embarrassing and cheap.", 0.6, "applause"),

    (HOST, "That is a good place to talk about tooling.", 0.5, None),
    (GUEST, "Sure, although I do not think tools matter as much as people say.", 0.7, None),
    (HOST, "We will come back to that after the break. Thanks everyone for listening, and we will see you next week.", 0.5, None),
]


def synth_line(voice: str, text: str, out_wav: Path) -> None:
    """Render one line of speech with the Windows SAPI engine."""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); "
        "$s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{out_wav}'); "
        f"$s.Speak(@'\n{text}\n'@); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        check=True,
        capture_output=True,
    )


def make_applause(duration: float, *, kind: str) -> np.ndarray:
    """Synthesise a crowd reaction: broadband noise with an amplitude envelope.

    Not a real crowd, but it has the loudness burst and spectral flatness the
    audio analyser keys on, which is what we need to test the signal path.
    """
    n = int(duration * SR)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, n)

    if kind == "applause":
        # Dense random claps: modulate noise with sharp random spikes.
        spikes = (rng.random(n) < 0.004).astype(float)
        env = np.convolve(spikes, np.exp(-np.linspace(0, 8, 400)), mode="same")
        sig = noise * (0.3 + env)
    else:  # laughter: slower amplitude wobble
        t = np.arange(n) / SR
        wobble = 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t) ** 2
        sig = noise * wobble

    # Fade in/out so the burst does not click.
    fade = int(0.08 * SR)
    if n > 2 * fade:
        sig[:fade] *= np.linspace(0, 1, fade)
        sig[-fade:] *= np.linspace(1, 0, fade)

    peak = np.abs(sig).max() or 1.0
    return (sig / peak * 0.35).astype(np.float32)


def build_audio(tmp: Path) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Render every line, returning the mixed track and the speaker timeline."""
    chunks: list[np.ndarray] = []
    timeline: list[tuple[str, float, float]] = []
    cursor = 0.0

    for i, (voice, text, gap, reaction) in enumerate(SCRIPT):
        wav = tmp / f"line_{i:03d}.wav"
        synth_line(voice, text, wav)
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            # Linear resample is fine for a test fixture.
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * SR / sr))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

        start = cursor
        chunks.append(audio)
        cursor += len(audio) / SR
        timeline.append((voice, start, cursor))

        if reaction:
            burst = make_applause(2.2, kind=reaction)
            chunks.append(burst)
            cursor += len(burst) / SR

        silence = np.zeros(int(gap * SR), dtype=np.float32)
        chunks.append(silence)
        cursor += gap

        print(f"  line {i + 1}/{len(SCRIPT)}  t={cursor:7.1f}s", flush=True)

    track = np.concatenate(chunks)
    # Light room tone so the file is never digitally silent.
    track += np.random.default_rng(3).normal(0, 0.0012, len(track)).astype(np.float32)
    return np.clip(track, -1.0, 1.0), timeline


def build_video(timeline, audio_path: Path, out_path: Path, duration: float) -> None:
    """Compose a 1280x720 video whose background changes on every speaker turn.

    That gives the scene detector real cuts to find and gives the reframer a
    landscape source to convert.
    """
    colours = {
        HOST: "0x1E3A5F",
        GUEST: "0x5F2E1E",
        THIRD: "0x1E5F35",
    }
    # Each background segment must run until the *next* speaker starts, not just
    # to the end of the line: gaps and reaction bursts live in between, and
    # sizing the video from speech alone lets -shortest truncate the audio.
    segments = []
    for i, (voice, start, _end) in enumerate(timeline):
        next_start = timeline[i + 1][1] if i + 1 < len(timeline) else duration
        segments.append((colours.get(voice, "0x333333"), max(0.35, next_start - start)))

    ffmpeg = resolve_binary("ffmpeg")
    inputs: list[str] = []
    filters: list[str] = []
    for i, (colour, dur) in enumerate(segments):
        inputs += [
            "-f", "lavfi",
            "-t", f"{dur:.3f}",
            "-i", f"color=c={colour}:s=1280x720:r=25",
        ]
        filters.append(f"[{i}:v]")

    concat = "".join(filters) + f"concat=n={len(segments)}:v=1:a=0[v]"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-i", str(audio_path),
        "-filter_complex", concat,
        "-map", "[v]",
        "-map", f"{len(segments)}:a",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "data" / "samples" / "interview.mp4"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("Synthesising speech...")
        track, timeline = build_audio(tmp)
        audio_path = tmp / "full.wav"
        sf.write(audio_path, track, SR)
        duration = len(track) / SR
        print(f"Audio ready: {duration:.1f}s")

        print("Composing video...")
        build_video(timeline, audio_path, out, duration)

    print(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {duration / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
