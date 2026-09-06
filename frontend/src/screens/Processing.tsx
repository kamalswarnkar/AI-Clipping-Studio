import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Job, ProjectStatus } from "../api/types";
import { ErrorPanel, Logo, ProgressBar, WarningList } from "../components/Common";

/** Poll interval. Fast enough to feel live, slow enough to be free. */
const POLL_MS = 1200;

function StatusMark({ job }: { job: Job }) {
  if (job.status === "completed")
    return <span className="text-accent">✓</span>;
  if (job.status === "skipped")
    return <span className="text-ink-600">–</span>;
  if (job.status === "failed")
    return <span className="text-red-400">✕</span>;
  if (job.status === "running")
    return (
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-ink-700 border-t-accent" />
    );
  return <span className="text-ink-700">·</span>;
}

function JobRow({ job }: { job: Job }) {
  const running = job.status === "running";
  const done = job.status === "completed";

  return (
    <li className="flex items-center gap-3 py-2">
      <span className="flex h-4 w-4 shrink-0 items-center justify-center text-sm">
        <StatusMark job={job} />
      </span>

      <span
        className={`flex-1 truncate text-sm ${
          running
            ? "text-ink-100"
            : done
              ? "text-ink-300"
              : job.status === "failed"
                ? "text-red-300"
                : "text-ink-600"
        }`}
      >
        {job.label}
      </span>

      {/* Real per-stage progress, not a simulated bar. */}
      {running && job.progress > 0 && job.progress < 1 && (
        <span className="w-24">
          <ProgressBar value={job.progress} />
        </span>
      )}

      <span className="w-40 shrink-0 truncate text-right text-xs text-ink-500">
        {job.status === "failed" ? job.error : job.message}
      </span>

      {job.duration_seconds != null && job.duration_seconds > 1 && (
        <span className="w-12 shrink-0 text-right text-xs tabular-nums text-ink-600">
          {job.duration_seconds < 60
            ? `${job.duration_seconds.toFixed(0)}s`
            : `${(job.duration_seconds / 60).toFixed(1)}m`}
        </span>
      )}
    </li>
  );
}

export default function Processing({
  projectId,
  onComplete,
  onCancel,
}: {
  projectId: string;
  onComplete: () => void;
  onCancel: () => void;
}) {
  const [status, setStatus] = useState<ProjectStatus | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const startedAt = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let active = true;
    let timer: number;

    const poll = async () => {
      try {
        const next = await api.status(projectId);
        if (!active) return;
        setStatus(next);
        setPollError(null);
        if (next.status === "completed") {
          onComplete();
          return;
        }
        if (next.status === "failed" || next.status === "cancelled") return;
      } catch (err) {
        if (active) setPollError((err as Error).message);
      }
      if (active) timer = window.setTimeout(poll, POLL_MS);
    };

    poll();
    const ticker = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - startedAt.current) / 1000)),
      1000,
    );

    return () => {
      active = false;
      clearTimeout(timer);
      clearInterval(ticker);
    };
  }, [projectId, onComplete]);

  const failed = status?.status === "failed";
  const cancelled = status?.status === "cancelled";

  return (
    <div className="mx-auto flex min-h-[calc(100vh-4.5rem)] max-w-3xl flex-col justify-center px-6 py-16">
      <header className="mb-10 flex items-baseline justify-between">
        <Logo subtitle={status?.current_stage || "Preparing"} />
        <span className="text-xs tabular-nums text-ink-500">
          {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
        </span>
      </header>

      {status && !failed && !cancelled && (
        <div className="mb-8">
          <div className="mb-2.5 flex items-baseline justify-between">
            <span className="text-sm text-ink-300">
              {status.current_stage || "Working"}
            </span>
            <span className="text-sm tabular-nums text-ink-400">
              {Math.round(status.overall_progress * 100)}%
            </span>
          </div>
          <ProgressBar value={status.overall_progress} />
          {status.clips_total > 0 && (
            <p className="mt-2.5 text-xs text-ink-500">
              {status.clips_ready} of {status.clips_total} clips rendered
            </p>
          )}
        </div>
      )}

      <ul className="panel divide-y divide-ink-850 px-4 py-1">
        {(status?.jobs ?? []).map((job) => (
          <JobRow key={job.id} job={job} />
        ))}
        {!status && (
          <li className="py-3 text-sm text-ink-500">Starting…</li>
        )}
      </ul>

      {status && status.warnings.length > 0 && (
        <div className="mt-6">
          <WarningList warnings={status.warnings} />
        </div>
      )}

      {pollError && (
        <p className="mt-6 text-sm text-ink-500">
          Lost contact with the backend — retrying. ({pollError})
        </p>
      )}

      {failed && status?.error && (
        <div className="mt-6">
          <ErrorPanel error={status.error} onRetry={onCancel} />
        </div>
      )}

      {cancelled && (
        <p className="mt-6 text-sm text-ink-400">This run was cancelled.</p>
      )}

      <div className="mt-8 flex gap-3">
        {!failed && !cancelled ? (
          <button
            className="btn-ghost"
            onClick={async () => {
              await api.cancel(projectId);
            }}
          >
            Cancel
          </button>
        ) : (
          <button className="btn-secondary" onClick={onCancel}>
            Start over
          </button>
        )}
        {status && status.clips_ready > 0 && !failed && (
          <button className="btn-secondary" onClick={onComplete}>
            View {status.clips_ready} ready clip
            {status.clips_ready === 1 ? "" : "s"}
          </button>
        )}
      </div>
    </div>
  );
}
