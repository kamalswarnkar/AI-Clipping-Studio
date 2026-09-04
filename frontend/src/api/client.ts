import type {
  Clip,
  ExportResult,
  Health,
  Project,
  ProjectSettings,
  ProjectStatus,
} from "./types";

const BASE = "/api";

/** Error carrying the backend's actionable detail, not just a status code. */
export class ApiError extends Error {
  status: number;
  stage?: string;
  remedy?: string;

  constructor(message: string, status: number, stage?: string, remedy?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.stage = stage;
    this.remedy = remedy;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(
      "Cannot reach the backend. Is the server running?",
      0,
      "network",
      "Start it with: python -m uvicorn app.main:app --app-dir backend",
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    let stage: string | undefined;
    let remedy: string | undefined;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg)
        detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
      stage = body.stage;
      remedy = body.remedy;
    } catch {
      /* keep the default message */
    }
    throw new ApiError(detail, response.status, stage, remedy);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  listProjects: () => request<Project[]>("/projects"),

  getProject: (id: string) => request<Project>(`/projects/${id}`),

  createProject: (name: string) => {
    const form = new FormData();
    form.append("name", name);
    return request<Project>("/projects", { method: "POST", body: form });
  },

  /** Upload with progress. Uses XHR because fetch cannot report upload progress. */
  uploadVideo: (
    id: string,
    file: File,
    settings: ProjectSettings,
    onProgress?: (fraction: number) => void,
  ): Promise<Project> =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      form.append("clip_count", String(settings.clip_count));
      form.append("min_duration", String(settings.min_duration));
      form.append("max_duration", String(settings.max_duration));
      form.append("vertical", String(settings.vertical));
      form.append("captions", String(settings.captions));
      form.append("smart_reframe", String(settings.smart_reframe));

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/projects/${id}/upload`);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText) as Project);
        } else {
          let detail = `Upload failed (${xhr.status})`;
          try {
            const body = JSON.parse(xhr.responseText);
            if (body.detail) detail = body.detail;
          } catch {
            /* keep default */
          }
          reject(new ApiError(detail, xhr.status));
        }
      };
      xhr.onerror = () =>
        reject(new ApiError("Upload failed: connection lost.", 0));
      xhr.send(form);
    }),

  process: (id: string) =>
    request<{ started: boolean; message: string }>(`/projects/${id}/process`, {
      method: "POST",
    }),

  cancel: (id: string) =>
    request<{ started: boolean; message: string }>(`/projects/${id}/cancel`, {
      method: "POST",
    }),

  status: (id: string) => request<ProjectStatus>(`/projects/${id}/status`),

  listClips: (id: string) => request<Clip[]>(`/projects/${id}/clips`),

  getClip: (projectId: string, clipId: string) =>
    request<Clip>(`/projects/${projectId}/clips/${clipId}`),

  adjustClip: (projectId: string, clipId: string, start: number, end: number) =>
    request<Clip>(`/projects/${projectId}/clips/${clipId}/adjust`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, end }),
    }),

  rerenderClip: (projectId: string, clipId: string) =>
    request<Clip>(`/projects/${projectId}/clips/${clipId}/rerender`, {
      method: "POST",
    }),

  feedback: (projectId: string, clipId: string, event: string, value = "") =>
    request<void>(
      `/projects/${projectId}/clips/${clipId}/feedback?event=${encodeURIComponent(
        event,
      )}&value=${encodeURIComponent(value)}`,
      { method: "POST" },
    ).catch(() => undefined), // telemetry must never interrupt the user

  prepareExport: (projectId: string, clipIds?: string[]) =>
    request<ExportResult>(`/projects/${projectId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_ids: clipIds ?? null }),
    }),

  deleteProject: (id: string) =>
    request<void>(`/projects/${id}`, { method: "DELETE" }),

  // Direct URLs for media elements and downloads.
  videoUrl: (projectId: string, clipId: string) =>
    `${BASE}/projects/${projectId}/clips/${clipId}/video`,
  thumbnailUrl: (projectId: string, clipId: string) =>
    `${BASE}/projects/${projectId}/clips/${clipId}/thumbnail`,
  downloadUrl: (projectId: string, clipId: string) =>
    `${BASE}/projects/${projectId}/clips/${clipId}/download`,
  exportUrl: (projectId: string) => `${BASE}/projects/${projectId}/export`,
};
