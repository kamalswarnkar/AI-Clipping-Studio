/** Shared formatting helpers. */

export function timecode(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
}

export function duration(seconds: number): string {
  return `${Math.round(seconds)}s`;
}

export function fileSize(bytes: number): string {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard API needs a secure context; fall back to execCommand.
    try {
      const el = document.createElement("textarea");
      el.value = text;
      el.style.position = "fixed";
      el.style.opacity = "0";
      document.body.appendChild(el);
      el.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(el);
      return ok;
    } catch {
      return false;
    }
  }
}

/**
 * CSS aspect-ratio for a project's clips.
 *
 * Clips keep the source shape unless vertical output was asked for, so the
 * player and the grid have to follow the source rather than assume 9:16.
 */
export function clipAspect(project: {
  settings: { vertical: boolean };
  media_info: { width?: number; height?: number };
} | null): string {
  if (!project) return "16 / 9";
  if (project.settings.vertical) return "9 / 16";
  const { width, height } = project.media_info;
  return width && height ? `${width} / ${height}` : "16 / 9";
}
