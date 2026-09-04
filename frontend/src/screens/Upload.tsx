import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import type { Health, ProjectSettings } from "../api/types";
import {
  ErrorPanel,
  Logo,
  LogoWatermark,
  ProgressBar,
  Spinner,
} from "../components/Common";
import { fileSize } from "../lib/format";

const ACCEPTED = [".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"];

const DEFAULTS: ProjectSettings = {
  clip_count: 15,
  min_duration: 10,
  max_duration: 60,
  vertical: true,
  captions: true,
  smart_reframe: true,
};

function NumberField({
  label,
  value,
  min,
  max,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  suffix?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-4 py-2.5">
      <span className="text-sm text-ink-300">{label}</span>
      <span className="relative">
        <input
          type="number"
          className="field w-24 pr-8 text-right tabular-nums"
          value={value}
          min={min}
          max={max}
          onChange={(e) => {
            const next = Number(e.target.value);
            if (!Number.isNaN(next)) onChange(next);
          }}
        />
        {suffix && (
          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-ink-500">
            {suffix}
          </span>
        )}
      </span>
    </label>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 py-2.5">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer appearance-none rounded border border-ink-600 bg-ink-900 transition-colors checked:border-accent checked:bg-accent"
      />
      <span className="min-w-0">
        <span className="block text-sm text-ink-200">{label}</span>
        <span className="block text-xs leading-relaxed text-ink-500">{hint}</span>
      </span>
    </label>
  );
}

export default function Upload({
  onStarted,
}: {
  onStarted: (projectId: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [settings, setSettings] = useState<ProjectSettings>(DEFAULTS);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<{ message: string; stage: string; remedy: string } | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Surface a broken setup before the user waits through a long upload.
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const chooseFile = useCallback((picked: File) => {
    const extension = picked.name.slice(picked.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED.includes(extension)) {
      setError({
        message: `"${extension || "unknown"}" is not a supported video format.`,
        stage: "upload",
        remedy: `Supported formats: ${ACCEPTED.join(", ")}`,
      });
      return;
    }
    setError(null);
    setFile(picked);
  }, []);

  const start = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const project = await api.createProject(file.name.replace(/\.[^.]+$/, ""));
      await api.uploadVideo(project.id, file, settings, setUploadProgress);
      await api.process(project.id);
      onStarted(project.id);
    } catch (err) {
      const apiError = err as ApiError;
      setError({
        message: apiError.message ?? "Something failed while starting the job.",
        stage: apiError.stage ?? "upload",
        remedy: apiError.remedy ?? "",
      });
      setBusy(false);
    }
  };

  const blockers: string[] = [];
  if (health && !health.ffmpeg.available) blockers.push(health.ffmpeg.error ?? "FFmpeg is missing.");
  if (health && !health.providers.transcription?.available)
    blockers.push(health.providers.transcription.reason);
  const llmDown = health && !health.providers.llm?.available;

  return (
    <div className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center px-6 py-16">
      <LogoWatermark />

      <header className="mb-12">
        <Logo />
        <h1 className="mt-6 text-2xl font-medium tracking-tight text-ink-50">
          One long video in. A folder of short clips out.
        </h1>
      </header>

      {/* --- drop zone --- */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) chooseFile(dropped);
        }}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        className={`cursor-pointer rounded-lg border border-dashed px-8 py-14 text-center transition-colors ${
          dragging
            ? "border-accent bg-accent/[0.04]"
            : file
              ? "border-ink-700 bg-ink-900"
              : "border-ink-700 hover:border-ink-600 hover:bg-ink-900/50"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="hidden"
          onChange={(e) => {
            const picked = e.target.files?.[0];
            if (picked) chooseFile(picked);
          }}
        />
        {file ? (
          <>
            <p className="truncate text-sm font-medium text-ink-100">{file.name}</p>
            <p className="mt-1.5 text-xs text-ink-500">
              {fileSize(file.size)} · click to choose a different file
            </p>
          </>
        ) : (
          <>
            <p className="text-sm text-ink-200">Drop your video here</p>
            <p className="mt-2 text-xs uppercase tracking-wider text-ink-500">
              MP4 · MOV · MKV · WebM
            </p>
          </>
        )}
      </div>

      {/* --- settings --- */}
      <div className="mt-8 divide-y divide-ink-800">
        <div className="pb-2">
          <NumberField
            label="Number of clips"
            value={settings.clip_count}
            min={1}
            max={50}
            onChange={(v) => setSettings({ ...settings, clip_count: v })}
          />
          <NumberField
            label="Minimum duration"
            value={settings.min_duration}
            min={3}
            max={180}
            suffix="s"
            onChange={(v) => setSettings({ ...settings, min_duration: v })}
          />
          <NumberField
            label="Maximum duration"
            value={settings.max_duration}
            min={5}
            max={300}
            suffix="s"
            onChange={(v) => setSettings({ ...settings, max_duration: v })}
          />
        </div>
        <div className="pt-2">
          <Toggle
            label="Vertical 9:16"
            hint="Convert to 1080x1920 for short-form platforms"
            checked={settings.vertical}
            onChange={(v) => setSettings({ ...settings, vertical: v })}
          />
          <Toggle
            label="Auto captions"
            hint="Burn word-timed subtitles into each clip"
            checked={settings.captions}
            onChange={(v) => setSettings({ ...settings, captions: v })}
          />
          <Toggle
            label="Smart reframing"
            hint="Track faces so speakers stay in frame when cropping"
            checked={settings.smart_reframe}
            onChange={(v) => setSettings({ ...settings, smart_reframe: v })}
          />
        </div>
      </div>

      {settings.max_duration <= settings.min_duration && (
        <p className="mt-4 text-xs text-red-400">
          Maximum duration must be greater than the minimum.
        </p>
      )}

      {/* --- setup problems --- */}
      {blockers.length > 0 && (
        <div className="mt-6">
          <ErrorPanel
            error={{
              message: blockers.join(" "),
              stage: "setup",
              remedy: "Fix this before uploading, or processing will fail.",
            }}
          />
        </div>
      )}

      {llmDown && blockers.length === 0 && (
        <p className="mt-6 rounded-md border border-accent-dim/40 bg-accent/[0.06] px-3 py-2.5 text-[13px] leading-relaxed text-ink-200">
          {health?.providers.llm.reason} Clips can still be generated using signal
          analysis, but selection and copy quality will be noticeably lower.
        </p>
      )}

      {error && (
        <div className="mt-6">
          <ErrorPanel error={error} />
        </div>
      )}

      {/* --- action --- */}
      <div className="mt-8">
        {busy ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm text-ink-300">
              <span className="flex items-center gap-2">
                <Spinner />
                {uploadProgress < 1 ? "Uploading" : "Starting analysis"}
              </span>
              <span className="tabular-nums text-ink-500">
                {Math.round(uploadProgress * 100)}%
              </span>
            </div>
            <ProgressBar value={uploadProgress} />
          </div>
        ) : (
          <button
            className="btn-primary w-full py-2.5"
            disabled={!file || settings.max_duration <= settings.min_duration || blockers.length > 0}
            onClick={start}
          >
            Generate clips
          </button>
        )}
      </div>
    </div>
  );
}
