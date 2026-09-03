import { useEffect, useState } from "react";
import type { PipelineErrorDetail, PipelineWarning } from "../api/types";
import { copyText } from "../lib/format";

export function Logo({ subtitle }: { subtitle?: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="text-[15px] font-semibold uppercase tracking-[0.2em] text-ink-100">
        Clipper
      </span>
      {subtitle && (
        <span className="text-[13px] text-ink-500">{subtitle}</span>
      )}
    </div>
  );
}

/** A failure the user can act on: what happened, and what to do about it. */
export function ErrorPanel({
  error,
  onRetry,
}: {
  error: PipelineErrorDetail;
  onRetry?: () => void;
}) {
  return (
    <div className="panel border-red-900/50 bg-red-950/20 p-5">
      <div className="flex items-start gap-3">
        <span
          aria-hidden
          className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-red-500/60 text-[11px] font-bold text-red-400"
        >
          !
        </span>
        <div className="min-w-0 flex-1">
          {error.stage && (
            <div className="label mb-1 text-red-400/80">{error.stage}</div>
          )}
          <p className="text-sm leading-relaxed text-ink-100">{error.message}</p>
          {error.remedy && (
            <p className="mt-2 text-sm leading-relaxed text-ink-400">
              {error.remedy}
            </p>
          )}
          {onRetry && (
            <button className="btn-secondary mt-4" onClick={onRetry}>
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function WarningList({ warnings }: { warnings: PipelineWarning[] }) {
  if (!warnings.length) return null;
  return (
    <div className="space-y-2">
      {warnings.map((w, i) => (
        <div
          key={i}
          className={`flex items-start gap-2.5 rounded-md border px-3 py-2.5 text-[13px] leading-relaxed ${
            w.severity === "warning"
              ? "border-accent-dim/40 bg-accent/[0.06] text-ink-200"
              : "border-ink-800 bg-ink-900 text-ink-400"
          }`}
        >
          <span
            aria-hidden
            className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
              w.severity === "warning" ? "bg-accent" : "bg-ink-500"
            }`}
          />
          <span>{w.message}</span>
        </div>
      ))}
    </div>
  );
}

/** Copy-to-clipboard button that confirms it worked. */
export function CopyButton({
  text,
  label = "Copy",
  className = "btn-secondary",
  onCopied,
}: {
  text: string;
  label?: string;
  className?: string;
  onCopied?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1600);
    return () => clearTimeout(timer);
  }, [copied]);

  return (
    <button
      className={className}
      disabled={!text}
      onClick={async () => {
        if (await copyText(text)) {
          setCopied(true);
          onCopied?.();
        }
      }}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={`inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-ink-600 border-t-accent ${className}`}
    />
  );
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div
      className="h-1 w-full overflow-hidden rounded-full bg-ink-800"
      role="progressbar"
      aria-valuenow={Math.round(value * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
        style={{ width: `${Math.max(2, value * 100)}%` }}
      />
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <p className="text-sm font-medium text-ink-200">{title}</p>
      {detail && (
        <p className="mt-2 max-w-md text-sm leading-relaxed text-ink-500">
          {detail}
        </p>
      )}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}
