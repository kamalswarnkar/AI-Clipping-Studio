import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Clip, Project } from "../api/types";
import { EmptyState, Logo, Spinner, WarningList } from "../components/Common";
import { duration, timecode } from "../lib/format";

function ClipCard({
  clip,
  projectId,
  selected,
  onToggle,
  onOpen,
}: {
  clip: Clip;
  projectId: string;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const ready = clip.render_status === "completed" && clip.has_video;

  return (
    <div className="group relative">
      <button
        onClick={onOpen}
        disabled={!ready}
        className={`relative block w-full overflow-hidden rounded-lg border bg-ink-900 transition-colors ${
          selected ? "border-accent" : "border-ink-800 hover:border-ink-600"
        } ${ready ? "cursor-pointer" : "cursor-default"}`}
        style={{ aspectRatio: "9 / 16" }}
      >
        {clip.has_thumbnail ? (
          <img
            src={api.thumbnailUrl(projectId, clip.id)}
            alt=""
            loading="lazy"
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-ink-850">
            {clip.render_status === "failed" ? (
              <span className="px-4 text-center text-xs text-red-400">
                Render failed
              </span>
            ) : (
              <Spinner />
            )}
          </div>
        )}

        {ready && (
          <span className="absolute inset-0 flex items-center justify-center bg-ink-950/0 transition-colors group-hover:bg-ink-950/30">
            <span className="flex h-11 w-11 items-center justify-center rounded-full bg-ink-950/70 opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
              <span className="ml-0.5 text-sm text-white">▶</span>
            </span>
          </span>
        )}

        <span className="absolute left-2 top-2 rounded bg-ink-950/75 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-ink-100 backdrop-blur-sm">
          {clip.name.replace("_", " ")}
        </span>
        <span className="absolute bottom-2 right-2 rounded bg-ink-950/75 px-1.5 py-0.5 text-[11px] tabular-nums text-ink-200 backdrop-blur-sm">
          {duration(clip.duration)}
        </span>
      </button>

      {/* Selection control sits outside the open-clip button. */}
      <label
        className="absolute right-2 top-2 flex h-6 w-6 cursor-pointer items-center justify-center rounded bg-ink-950/75 backdrop-blur-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          aria-label={`Select ${clip.name}`}
          className="h-3.5 w-3.5 cursor-pointer appearance-none rounded-sm border border-ink-500 bg-transparent checked:border-accent checked:bg-accent"
        />
      </label>

      <div className="mt-2 px-0.5">
        <p className="truncate text-[13px] text-ink-300">
          {clip.topic || "Untitled moment"}
        </p>
        <p className="mt-0.5 text-[11px] tabular-nums text-ink-600">
          {timecode(clip.start)} – {timecode(clip.end)}
        </p>
      </div>
    </div>
  );
}

export default function Results({
  projectId,
  onOpenClip,
  onNewProject,
}: {
  projectId: string;
  onOpenClip: (clipId: string) => void;
  onNewProject: () => void;
}) {
  const [clips, setClips] = useState<Clip[] | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState(false);
  const [notice, setNotice] = useState<string>("");

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const [c, p] = await Promise.all([
          api.listClips(projectId),
          api.getProject(projectId),
        ]);
        if (!active) return;
        setClips(c);
        setProject(p);
        // Keep polling while renders are still finishing.
        if (c.some((clip) => clip.render_status === "pending" || clip.render_status === "rendering")) {
          window.setTimeout(load, 2500);
        }
      } catch {
        /* leave previous state */
      }
    };
    load();
    return () => {
      active = false;
    };
  }, [projectId]);

  const ready = useMemo(
    () => (clips ?? []).filter((c) => c.render_status === "completed"),
    [clips],
  );

  const runExport = async (ids?: string[]) => {
    setExporting(true);
    setNotice("");
    try {
      const result = await api.prepareExport(projectId, ids);
      setNotice(result.message);
      // Trigger the browser download of the packaged ZIP.
      window.location.href = api.exportUrl(projectId);
    } catch (err) {
      setNotice((err as Error).message);
    } finally {
      setExporting(false);
    }
  };

  if (!clips) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <header className="mb-8 flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <Logo subtitle={project?.source_filename} />
          <h1 className="mt-4 text-xl font-medium tracking-tight text-ink-50">
            {clips.length} clip{clips.length === 1 ? "" : "s"} generated
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button className="btn-ghost" onClick={onNewProject}>
            New video
          </button>
          {selected.size > 0 && (
            <button
              className="btn-secondary"
              disabled={exporting}
              onClick={() => runExport([...selected])}
            >
              Export selected ({selected.size})
            </button>
          )}
          <button
            className="btn-primary"
            disabled={exporting || clips.length === 0}
            onClick={() => runExport()}
          >
            {exporting ? "Packaging…" : "Export all"}
          </button>
        </div>
      </header>

      {project && project.warnings.length > 0 && (
        <div className="mb-8">
          <WarningList warnings={project.warnings} />
        </div>
      )}

      {notice && (
        <p className="mb-6 rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-[13px] text-ink-300">
          {notice}
        </p>
      )}

      {clips.length === 0 ? (
        <EmptyState
          title="No clips were produced"
          detail="No moment in this video was strong enough to stand on its own as a short clip."
          action={
            <button className="btn-secondary" onClick={onNewProject}>
              Try another video
            </button>
          }
        />
      ) : (
        <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-5">
          {clips.map((clip) => (
            <ClipCard
              key={clip.id}
              clip={clip}
              projectId={projectId}
              selected={selected.has(clip.id)}
              onToggle={() =>
                setSelected((prev) => {
                  const next = new Set(prev);
                  next.has(clip.id) ? next.delete(clip.id) : next.add(clip.id);
                  return next;
                })
              }
              onOpen={() => onOpenClip(clip.id)}
            />
          ))}
        </div>
      )}

      {clips.length > 0 && ready.length < clips.length && (
        <p className="mt-8 text-xs text-ink-500">
          {clips.length - ready.length} clip
          {clips.length - ready.length === 1 ? "" : "s"} still rendering or failed.
        </p>
      )}
    </div>
  );
}
