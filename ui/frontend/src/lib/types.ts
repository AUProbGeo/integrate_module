/** Shared API types mirroring ui/backend responses. */

export type FileClass = 'PRIOR' | 'DATA' | 'POSTERIOR' | 'UNKNOWN' | 'UNREADABLE';

export interface H5FileInfo {
  name: string;
  size_mb: number;
  mtime: number;
  class: FileClass;
}

export interface H5Node {
  name: string;
  path: string;
  kind: 'group' | 'dataset';
  attrs: Record<string, unknown>;
  shape?: number[];
  dtype?: string;
  size_mb?: number;
  children?: H5Node[];
}

export interface H5Summary {
  class: FileClass;
  file: string;
  n_realizations?: number;
  n_points?: number;
  models?: Array<{ id: string; shape?: number[]; name?: string; is_discrete?: boolean } | string>;
  data?: Array<{ id: string; shape: number[] }>;
  datasets?: Array<{ id: string; noise_model: string; name: string; keys: string[]; shape?: number[]; n_used?: number }>;
  has_forward_data?: boolean;
  root_attrs?: Record<string, unknown>;
  f5_data?: string;
  f5_prior?: string;
  inv_time?: number;
  t?: StatRange;
  ev?: StatRange;
  chi2?: StatRange;
  n_unique?: StatRange;
  [k: string]: unknown;
}

export interface StatRange {
  min: number;
  max: number;
  mean: number;
  n: number;
}

export interface JobProgress {
  current: number;
  total: number;
  info: { phase?: string; status?: string; total_points?: number };
}

export type JobStatus = 'pending' | 'running' | 'done' | 'error' | 'cancelled';

export interface Job {
  id: string;
  kind: string;
  params: Record<string, unknown>;
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  ended_at: number | null;
  progress: JobProgress;
  result: { f_post_h5?: string };
  error: string | null;
  logs?: string[];
}

export interface LLMConfig {
  configured: boolean;
  provider: 'claude' | 'ollama' | null;
  model: string | null;
}

export interface OllamaModels {
  running: boolean;
  models: string[];
}

export interface PriorModelInfo {
  im: number;
  name: string;
  kind: string;
  depth_min: number;
  depth_max: number;
  n_layers: number;
  classes?: Array<{ id: number; name: string }> | null;
}

export interface PriorModelsResponse {
  f_prior_h5: string;
  models: PriorModelInfo[];
  describe: string;
}

export interface QueryTranslation {
  query_dict: Record<string, unknown>;
  interpretation: string;
  system_prompt: string;
}

export interface QueryResult {
  kind: 'probability' | 'percentile';
  n_locations: number;
  mean_probability?: number;
  percentiles?: number[];
  figures: string[];
}

export interface PostStats {
  index: number[];
  n_points: number;
  ev?: (number | null)[];
  ev_post?: (number | null)[];
  t?: (number | null)[];
  chi2?: (number | null)[];
  n_unique?: (number | null)[];
  utmx?: (number | null)[];
  utmy?: (number | null)[];
}
