export interface ProjectSettings {
  clip_count: number;
  min_duration: number;
  max_duration: number;
  vertical: boolean;
  captions: boolean;
  smart_reframe: boolean;
  vocabulary: string;
}

export interface MediaInfo {
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  has_audio?: boolean;
  size_bytes?: number;
}

export interface PipelineWarning {
  stage: string;
  message: string;
  severity: "info" | "warning";
}

export interface PipelineErrorDetail {
  message: string;
  stage: string;
  remedy: string;
}

export interface Project {
  id: string;
  name: string;
  source_filename: string;
  status: string;
  created_at: string;
  updated_at: string;
  settings: ProjectSettings;
  media_info: MediaInfo;
  global_context: string;
  warnings: PipelineWarning[];
  error: PipelineErrorDetail | null;
  clip_count: number;
}

export interface Job {
  id: string;
  type: string;
  label: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped" | "cancelled";
  progress: number;
  message: string;
  error: string | null;
  attempts: number;
  duration_seconds: number | null;
}

export interface ProjectStatus {
  id: string;
  status: string;
  overall_progress: number;
  current_stage: string;
  jobs: Job[];
  warnings: PipelineWarning[];
  error: PipelineErrorDetail | null;
  clips_ready: number;
  clips_total: number;
}

export interface Clip {
  id: string;
  index: number;
  name: string;
  start: number;
  end: number;
  duration: number;
  topic: string;
  transcript: string;
  reason: string;
  analysis_notes: string;
  context_dependency: string;
  context: string;
  standalone: boolean;
  best_hook: string;
  hooks: { category: string; text: string; rank: number }[];
  caption: string;
  score: number;
  speakers: string[];
  render_status: "pending" | "rendering" | "completed" | "failed";
  render_error: string | null;
  has_video: boolean;
  has_thumbnail: boolean;
  breakdown: Record<string, number>;
}

export interface ProviderHealth {
  provider: string;
  model: string;
  available: boolean;
  reason: string;
}

export interface Health {
  status: "ok" | "degraded" | "error";
  ffmpeg: { available: boolean; ffmpeg?: string; ffprobe?: string; error?: string };
  providers: Record<string, ProviderHealth>;
  settings: Record<string, unknown>;
}

export interface ExportResult {
  ready: boolean;
  filename: string;
  clip_count: number;
  message: string;
}
