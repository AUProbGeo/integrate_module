import type { H5FileInfo, H5Node, H5Summary, Job, LLMConfig, OllamaModels, PostStats, PriorModelsResponse, QueryResult, QueryTranslation, VolumeGrowResponse, VolumeProbResponse, VolumeVolumesResponse } from './types';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* non-JSON error */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listFiles: () => req<{ workspace: string; files: H5FileInfo[] }>('/api/files'),
  fileTree: (name: string) => req<H5Node & { truncated?: boolean }>(`/api/files/${encodeURIComponent(name)}/tree`),
  fileSummary: (name: string) => req<H5Summary>(`/api/files/${encodeURIComponent(name)}/summary`),

  listJobs: () => req<{ jobs: Job[] }>('/api/jobs'),
  getJob: (id: string) => req<Job>(`/api/jobs/${id}`),
  startRejection: (params: Record<string, unknown>) =>
    req<Job>('/api/jobs/rejection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  stopJob: (id: string) => req<Job>(`/api/jobs/${id}/stop`, { method: 'POST' }),

  llmConfig: () => req<LLMConfig>('/api/query/config'),
  ollamaModels: () => req<OllamaModels>('/api/query/ollama-models'),
  queryModels: (name: string) =>
    req<PriorModelsResponse>(`/api/query/models?f=${encodeURIComponent(name)}`),
  systemPrompt: (name: string) =>
    req<{ system_prompt: string }>(`/api/query/system-prompt?f=${encodeURIComponent(name)}`),
  translateQuery: (params: {
    f: string;
    text: string;
    provider?: 'claude' | 'ollama';
    api_key?: string;
    model?: string;
    system_prompt?: string;
  }) =>
    req<QueryTranslation>('/api/query/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  evaluateQuery: (params: {
    f: string;
    text: string;
    query_dict: Record<string, unknown>;
    interpretation?: string;
  }) =>
    req<QueryResult>('/api/query/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),

  postStats: (name: string) => req<PostStats>(`/api/results/${encodeURIComponent(name)}/stats`),
  profileUrl: (name: string, im: number) =>
    `/api/results/${encodeURIComponent(name)}/profile.png?im=${im}`,
  volumeProb: (params: {
    f: string;
    query_dict: Record<string, unknown>;
    geo?: { hull_ratio: number; edge_buffer: number | null; cell_area_k: number; elong_max: number | null };
  }) =>
    req<VolumeProbResponse>('/api/volume/prob', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  volumeGrow: (params: {
    f: string;
    p: number[];
    p_min: number;
    x_center?: number | null;
    y_center?: number | null;
    max_area_m2?: number | null;
    geo?: { hull_ratio: number; edge_buffer: number | null; cell_area_k: number; elong_max: number | null };
  }) =>
    req<VolumeGrowResponse>('/api/volume/grow', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  volumeVolumes: (params: {
    f: string;
    query_dict: Record<string, unknown>;
    areas: Array<{ name: string; indices: number[] }>;
    text?: string;
    interpretation?: string;
    geo?: { hull_ratio: number; edge_buffer: number | null; cell_area_k: number; elong_max: number | null };
  }) =>
    req<VolumeVolumesResponse>('/api/volume/volumes', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
};
