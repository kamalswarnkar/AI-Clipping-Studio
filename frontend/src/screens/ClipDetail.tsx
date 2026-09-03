import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Clip } from "../api/types";
import { CopyButton, ErrorPanel, Spinner } from "../components/Common";
import { timecode } from "../lib/format";

function HookRow({
  hook,
  onCopy,
}: {
  hook: { category: string; text: string; rank: number; is_best: boolean };
  onCopy: () => void;
}) {
  return (
    <li
      className={`group flex items-start gap-3 rounded-md px-3 py-2.5 transition-colors hover:bg-ink-850 ${
        hook.is_best ? "bg-accent/[0.07]" : ""
      }`}
    >
      <span className="mt-0.5 w-5 shrink-0 text-right text-[11px] tabular-nums text-ink-600">
        {hook.rank}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm leading-relaxed text-ink-100">
          {hook.text}
        </span>
        <span className="mt-0.5 block text-[11px] uppercase tracking-wider text-ink-600">
          {hook.category}
          {hook.is_best && <span className="ml-2 text-accent">Best</span>}
        </span>
      </span>
      <CopyButton
        text={hook.text}
        label="Copy"
        className="btn-ghost shrink-0 px-2 py-1 text-xs opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
        onCopied={onCopy}
      />
    </li>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-4 py-1.5 text-[13px]">
      <span className="w-28 shrink-0 text-ink-500">{label}</span>
      <span className="min-w-0 flex-1 text-ink-300">{value}</span>
    </div>
  );
}

export default function ClipDetail({
  projectId,
  clipId,
  onBack,
}: {
  projectId: string;
  clipId: string;
  onBack: () => void;
}) {
  const [clip, setClip] = useState<Clip | null>(null);
  const [busy, setBusy] = useState<"copy" | "adjust" | null>(null);
  const [error, setError] = useState<string>("");
  const [showAdjust, setShowAdjust] = useState(false);
  const [range, setRange] = useState({ start: 0, end: 0 });
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    api
      .getClip(projectId, clipId)
      .then((c) => {
        setClip(c);
        setRange({ start: c.start, end: c.end });
      })
      .catch((err) => setError((err as Error).message));
  }, [projectId, clipId]);

  const regenerate = async () => {
    setBusy("copy");
    setError("");
    try {
      setClip(await api.regenerateCopy(projectId, clipId));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const applyAdjust = async () => {
    setBusy("adjust");
    setError("");
    try {
      const updated = await api.adjustClip(projectId, clipId, range.start, range.end);
      setClip(updated);
      setRange({ start: updated.start, end: updated.end });
      setShowAdjust(false);
      // Force the player to reload the re-rendered file.
      if (videoRef.current) videoRef.current.load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  if (!clip) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        {error ? (
          <ErrorPanel error={{ message: error, stage: "", remedy: "" }} onRetry={onBack} />
        ) : (
          <Spinner />
        )}
      </div>
    );
  }

  const ranked = [...clip.hooks].sort((a, b) => a.rank - b.rank);

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <button className="btn-ghost mb-6 -ml-2" onClick={onBack}>
        ← All clips
      </button>

      <div className="grid gap-10 lg:grid-cols-[minmax(0,380px)_1fr]">
        {/* --- left: player --- */}
        <div>
          <div
            className="overflow-hidden rounded-lg border border-ink-800 bg-black"
            style={{ aspectRatio: "9 / 16" }}
          >
            {clip.has_video ? (
              <video
                ref={videoRef}
                key={clip.id}
                src={api.videoUrl(projectId, clip.id)}
                poster={
                  clip.has_thumbnail ? api.thumbnailUrl(projectId, clip.id) : undefined
                }
                controls
                playsInline
                className="h-full w-full"
              />
            ) : (
              <div className="flex h-full items-center justify-center px-6 text-center text-sm text-ink-500">
                {clip.render_error ?? "This clip has not been rendered."}
              </div>
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <a
              className="btn-primary flex-1"
              href={api.downloadUrl(projectId, clip.id)}
              download
              aria-disabled={!clip.has_video}
            >
              Download clip
            </a>
            <button
              className="btn-secondary"
              onClick={() => setShowAdjust((v) => !v)}
            >
              Adjust
            </button>
          </div>

          {showAdjust && (
            <div className="panel mt-3 space-y-3 p-4">
              <p className="text-xs leading-relaxed text-ink-500">
                Times are in source-video seconds. They are snapped to word
                boundaries, so the clip will not cut mid-word.
              </p>
              <div className="flex gap-3">
                <label className="flex-1">
                  <span className="label">Start</span>
                  <input
                    type="number"
                    step="0.1"
                    className="field mt-1 tabular-nums"
                    value={range.start}
                    onChange={(e) =>
                      setRange({ ...range, start: Number(e.target.value) })
                    }
                  />
                </label>
                <label className="flex-1">
                  <span className="label">End</span>
                  <input
                    type="number"
                    step="0.1"
                    className="field mt-1 tabular-nums"
                    value={range.end}
                    onChange={(e) =>
                      setRange({ ...range, end: Number(e.target.value) })
                    }
                  />
                </label>
              </div>
              <button
                className="btn-primary w-full"
                disabled={busy === "adjust" || range.end <= range.start}
                onClick={applyAdjust}
              >
                {busy === "adjust" ? "Re-rendering…" : "Apply and re-render"}
              </button>
            </div>
          )}
        </div>

        {/* --- right: copy and metadata --- */}
        <div className="min-w-0">
          <header className="mb-6">
            <h1 className="text-lg font-medium tracking-tight text-ink-50">
              {clip.name.replace("_", " ")}
            </h1>
            <p className="mt-1 text-sm text-ink-400">
              {clip.topic || "No topic recorded"}
            </p>
          </header>

          {error && (
            <p className="mb-4 rounded-md border border-red-900/50 bg-red-950/20 px-3 py-2 text-[13px] text-red-300">
              {error}
            </p>
          )}

          {/* best hook */}
          {clip.best_hook && (
            <section className="mb-6">
              <div className="label mb-2">Best hook</div>
              <div className="flex items-start gap-3 rounded-md border border-accent-dim/40 bg-accent/[0.07] px-4 py-3">
                <p className="min-w-0 flex-1 text-[15px] leading-relaxed text-ink-50">
                  {clip.best_hook}
                </p>
                <CopyButton
                  text={clip.best_hook}
                  className="btn-secondary shrink-0 px-2.5 py-1 text-xs"
                  onCopied={() =>
                    api.feedback(projectId, clip.id, "hook_selected", clip.best_hook)
                  }
                />
              </div>
            </section>
          )}

          {/* caption */}
          <section className="mb-6">
            <div className="mb-2 flex items-center justify-between">
              <span className="label">Caption</span>
              <CopyButton
                text={clip.caption}
                label="Copy caption"
                className="btn-ghost px-2 py-1 text-xs"
              />
            </div>
            <p className="panel px-4 py-3 text-sm leading-relaxed text-ink-200">
              {clip.caption || "No caption generated."}
            </p>
          </section>

          {/* hooks */}
          <section className="mb-6">
            <div className="mb-2 flex items-center justify-between">
              <span className="label">All hooks ({ranked.length})</span>
              <button
                className="btn-ghost px-2 py-1 text-xs"
                disabled={busy === "copy"}
                onClick={regenerate}
              >
                {busy === "copy" ? "Regenerating…" : "Regenerate copy"}
              </button>
            </div>
            <ul className="panel divide-y divide-ink-850 py-1">
              {ranked.map((hook, i) => (
                <HookRow
                  key={i}
                  hook={hook}
                  onCopy={() =>
                    api.feedback(projectId, clip.id, "hook_selected", hook.text)
                  }
                />
              ))}
            </ul>
          </section>

          {/* metadata */}
          <section>
            <div className="label mb-2">Clip information</div>
            <div className="panel divide-y divide-ink-850 px-4 py-2">
              <InfoRow
                label="Source range"
                value={`${timecode(clip.start)} – ${timecode(clip.end)} (${Math.round(clip.duration)}s)`}
              />
              <InfoRow
                label="Speakers"
                value={clip.speakers.join(", ") || "Not identified"}
              />
              <InfoRow label="Quality score" value={clip.score.toFixed(2)} />
              <InfoRow label="Standalone" value={clip.context_dependency} />
              {clip.reason && <InfoRow label="Why selected" value={clip.reason} />}
              {clip.analysis_notes && (
                <InfoRow label="Validation" value={clip.analysis_notes} />
              )}
            </div>

            <details className="mt-3 group">
              <summary className="cursor-pointer list-none text-xs text-ink-500 hover:text-ink-300">
                Transcript
              </summary>
              <p className="panel mt-2 max-h-64 overflow-y-auto px-4 py-3 text-[13px] leading-relaxed text-ink-300">
                {clip.transcript || "No transcript."}
              </p>
            </details>
          </section>
        </div>
      </div>
    </div>
  );
}
