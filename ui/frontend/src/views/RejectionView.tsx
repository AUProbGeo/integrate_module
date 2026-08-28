/** Rejection sampling view: parameter form, run control, live monitor, results. */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Play, Square, ChevronDown, ChevronRight, Terminal } from 'lucide-react';
import { api } from '../lib/api';
import { useJobs } from '../lib/jobs';
import type { H5FileInfo, H5Summary, Job, PostStats } from '../lib/types';
import { Button, Card, Field, FileClassBadge, ProgressBar, Select, StatusBadge, TextInput } from '../components/ui';
import { Sparkline } from '../components/Sparkline';

const PHASE_LABELS: Record<string, string> = {
  initializing: 'Initializing',
  generating: 'Generating',
  computing: 'Computing',
  sampling: 'Sampling',
  saving: 'Saving',
  post_processing: 'Post-processing',
  completed: 'Completed',
};

function parseIntList(text: string): number[] | undefined {
  const t = text.trim();
  if (!t) return undefined;
  const out: number[] = [];
  for (const part of t.split(',')) {
    const m = part.trim().match(/^(-?\d+)\s*-\s*(-?\d+)$/);
    if (m) {
      const [a, b] = [+m[1], +m[2]];
      for (let i = a; i <= b; i++) out.push(i);
    } else if (part.trim() !== '') {
      out.push(+part.trim());
    }
  }
  return out.length ? out : undefined;
}

function useElapsed(job: Job | undefined): number {
  const [, tick] = useState(0);
  useEffect(() => {
    if (job?.status !== 'running') return;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [job?.status]);
  if (!job?.started_at) return 0;
  return Math.max(0, (job.ended_at ?? Date.now() / 1000) - job.started_at);
}

function fmtElapsed(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}

// ---------------------------------------------------------------------------

function RejectionForm({ files, onStarted }: { files: H5FileInfo[]; onStarted: (id: string) => void }) {
  const priors = files.filter((f) => f.class === 'PRIOR');
  const datas = files.filter((f) => f.class === 'DATA');

  const [fPrior, setFPrior] = useState('');
  const [fData, setFData] = useState('');
  const [fPost, setFPost] = useState('');
  const [nUse, setNUse] = useState('');
  const [nr, setNr] = useState('1000');
  const [autoT, setAutoT] = useState(true);
  const [tBase, setTBase] = useState('1');
  const [ncpu, setNcpu] = useState('0');
  const [backend, setBackend] = useState('numpy');
  const [parallel, setParallel] = useState(true);
  const [idUse, setIdUse] = useState('');
  const [ipRange, setIpRange] = useState('');
  const [updatePostStat, setUpdatePostStat] = useState(true);
  const [normLik, setNormLik] = useState(false);
  const [useNBest, setUseNBest] = useState('');
  const [tNAbove, setTNAbove] = useState('');
  const [tPAcc, setTPAcc] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const [priorSummary, setPriorSummary] = useState<H5Summary | null>(null);
  const [dataSummary, setDataSummary] = useState<H5Summary | null>(null);

  useEffect(() => {
    setPriorSummary(fPrior ? null : null);
    if (fPrior) api.fileSummary(fPrior).then(setPriorSummary).catch(() => setPriorSummary(null));
  }, [fPrior]);
  useEffect(() => {
    if (fData) api.fileSummary(fData).then(setDataSummary).catch(() => setDataSummary(null));
    else setDataSummary(null);
  }, [fData]);

  const canRun = fPrior && fData && !busy;
  const noForwardData = priorSummary && priorSummary.class === 'PRIOR' && priorSummary.has_forward_data === false;

  const start = async () => {
    setBusy(true);
    setFormError(null);
    try {
      const params: Record<string, unknown> = {
        f_prior_h5: fPrior,
        f_data_h5: fData,
        autoT: autoT ? 1 : 0,
        parallel,
        updatePostStat,
        normalize_likelihood: normLik,
        backend,
      };
      if (fPost.trim()) params.f_post_h5 = fPost.trim();
      if (nUse.trim()) params.N_use = +nUse;
      if (nr.trim()) params.nr = +nr;
      if (!autoT) params.T_base = +tBase;
      if (+ncpu > 0) params.Ncpu = +ncpu;
      if (useNBest.trim()) params.use_N_best = +useNBest;
      if (tNAbove.trim()) params.T_N_above = +tNAbove;
      if (tPAcc.trim()) params.T_P_acc_level = +tPAcc;
      const ids = parseIntList(idUse);
      if (ids) params.id_use = ids;
      const ips = parseIntList(ipRange);
      if (ips) params.ip_range = ips;

      const job = await api.startRejection(params);
      onStarted(job.id);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title="Rejection sampling">
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Prior file (PRIOR.h5)">
            <Select value={fPrior} onChange={(e) => setFPrior(e.target.value)}>
              <option value="">Select prior…</option>
              {priors.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
            </Select>
          </Field>
          <Field label="Data file (DATA.h5)">
            <Select value={fData} onChange={(e) => setFData(e.target.value)}>
              <option value="">Select data…</option>
              {datas.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
            </Select>
          </Field>
        </div>

        {priorSummary && priorSummary.class === 'PRIOR' && (
          <div className="rounded-lg border border-edge bg-panel-2 px-3 py-2 text-xs text-muted">
            {priorSummary.n_realizations?.toLocaleString()} realizations ·{' '}
            {Array.isArray(priorSummary.models) && priorSummary.models.map((m) => (typeof m === 'string' ? m : `${m.id} ${m.name || ''}`)).join(' · ')}
            {' — '}
            {noForwardData ? (
              <span className="text-warn">⚠ no forward data; this prior cannot be used for inversion</span>
            ) : (
              <span className="text-accent">✓ forward data present</span>
            )}
          </div>
        )}
        {dataSummary && (
          <div className="rounded-lg border border-edge bg-panel-2 px-3 py-2 text-xs text-muted">
            {dataSummary.n_points?.toLocaleString()} data points ·{' '}
            {dataSummary.datasets?.map((d) => `${d.id} (${d.noise_model}${d.shape ? `, ${d.shape.join('×')}` : ''})`).join(' · ')}
          </div>
        )}

        <Field label="Output file" hint="Leave empty for automatic naming">
          <TextInput value={fPost} onChange={(e) => setFPost(e.target.value)} placeholder="POST_….h5 (auto)" />
        </Field>

        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Prior samples (N_use)" hint="Empty = all">
            <TextInput value={nUse} onChange={(e) => setNUse(e.target.value)} placeholder={priorSummary?.n_realizations?.toString() ?? 'all'} inputMode="numeric" />
          </Field>
          <Field label="Posterior samples / point (nr)">
            <TextInput value={nr} onChange={(e) => setNr(e.target.value)} inputMode="numeric" />
          </Field>
          <Field label="CPU cores" hint="0 = auto">
            <TextInput value={ncpu} onChange={(e) => setNcpu(e.target.value)} inputMode="numeric" />
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <label className="inline-flex items-center gap-2">
            <input type="checkbox" checked={autoT} onChange={(e) => setAutoT(e.target.checked)} className="accent-emerald-500" />
            Auto temperature (autoT)
          </label>
          {!autoT && (
            <Field label="T base">
              <TextInput value={tBase} onChange={(e) => setTBase(e.target.value)} className="w-24" inputMode="decimal" />
            </Field>
          )}
          <label className="inline-flex items-center gap-2">
            <input type="checkbox" checked={parallel} onChange={(e) => setParallel(e.target.checked)} className="accent-emerald-500" />
            Parallel
          </label>
          <Field label="Backend">
            <Select value={backend} onChange={(e) => setBackend(e.target.value)} className="w-32">
              <option value="numpy">numpy</option>
              <option value="jax">jax</option>
            </Select>
          </Field>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Data ids to use" hint="e.g. 1 or 1,2 — empty = all">
            <TextInput value={idUse} onChange={(e) => setIdUse(e.target.value)} placeholder="all" />
          </Field>
          <Field
            label="Data points to invert"
            hint={dataSummary?.n_points ? `0 – ${dataSummary.n_points - 1}; e.g. 0-9,20 — empty = all` : 'e.g. 0-9,20 — empty = all'}
          >
            <TextInput value={ipRange} onChange={(e) => setIpRange(e.target.value)} placeholder="all" />
          </Field>
        </div>

        <button
          type="button"
          onClick={() => setAdvanced(!advanced)}
          className="flex items-center gap-1 text-xs font-semibold tracking-wide text-muted uppercase hover:text-fg"
        >
          {advanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />} Advanced
        </button>
        {advanced && (
          <div className="grid gap-4 border-l-2 border-edge pl-4 sm:grid-cols-3">
            <Field label="use_N_best" hint="0 = disabled">
              <TextInput value={useNBest} onChange={(e) => setUseNBest(e.target.value)} placeholder="0" inputMode="numeric" />
            </Field>
            <Field label="T_N_above">
              <TextInput value={tNAbove} onChange={(e) => setTNAbove(e.target.value)} placeholder="10" inputMode="numeric" />
            </Field>
            <Field label="T_P_acc_level">
              <TextInput value={tPAcc} onChange={(e) => setTPAcc(e.target.value)} placeholder="0.2" inputMode="decimal" />
            </Field>
            <label className="inline-flex items-center gap-2 text-sm">
              <input type="checkbox" checked={updatePostStat} onChange={(e) => setUpdatePostStat(e.target.checked)} className="accent-emerald-500" />
              Compute posterior stats
            </label>
            <label className="inline-flex items-center gap-2 text-sm">
              <input type="checkbox" checked={normLik} onChange={(e) => setNormLik(e.target.checked)} className="accent-emerald-500" />
              Normalize likelihood
            </label>
          </div>
        )}

        {formError && <div className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">{formError}</div>}

        <Button onClick={start} disabled={!canRun || !!noForwardData} className="w-full justify-center py-2.5 text-base">
          <Play size={16} /> {busy ? 'Starting…' : 'Run inversion'}
        </Button>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

function JobMonitor({ job }: { job: Job }) {
  const { logs } = useJobs();
  const elapsed = useElapsed(job);
  const logRef = useRef<HTMLDivElement>(null);
  const [stopping, setStopping] = useState(false);

  const lines = logs[job.id] ?? job.logs ?? [];
  const { current, total, info } = job.progress;
  const phase = PHASE_LABELS[info.phase ?? ''] ?? info.phase ?? job.status;

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [lines.length]);

  return (
    <Card
      title="Run monitor"
      actions={
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-muted">{fmtElapsed(elapsed)}</span>
          <StatusBadge value={job.status} />
          {job.status === 'running' && (
            <Button
              variant="danger"
              disabled={stopping}
              onClick={async () => { setStopping(true); await api.stopJob(job.id); }}
            >
              <Square size={13} /> Stop
            </Button>
          )}
        </div>
      }
    >
      <div className="space-y-3">
        <div className="flex items-baseline justify-between text-sm">
          <span className="font-medium text-info">{phase}</span>
          <span className="font-mono text-xs text-muted">
            {total > 0 ? `${current} / ${total} data points` : '\u00A0'}
          </span>
        </div>
        <ProgressBar current={current} total={total} />
        <div className="text-xs text-muted">{info.status}</div>

        <div ref={logRef} className="h-56 overflow-auto rounded-lg border border-edge bg-bg p-3 font-mono text-[11px] leading-relaxed text-fg/80">
          {lines.length === 0 ? (
            <div className="flex h-full items-center justify-center gap-2 text-muted/50">
              <Terminal size={14} /> waiting for output…
            </div>
          ) : (
            lines.map((l, i) => <div key={i} className="whitespace-pre-wrap">{l}</div>)
          )}
        </div>

        {job.error && (
          <pre className="max-h-40 overflow-auto rounded-lg border border-danger/40 bg-danger/10 p-3 text-xs text-danger">{job.error}</pre>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

function ResultsPanel({ job }: { job: Job }) {
  const fPost = job.result.f_post_h5;
  const [summary, setSummary] = useState<H5Summary | null>(null);
  const [stats, setStats] = useState<PostStats | null>(null);

  useEffect(() => {
    if (!fPost) return;
    api.fileSummary(fPost).then(setSummary).catch(() => {});
    api.postStats(fPost).then(setStats).catch(() => {});
  }, [fPost]);

  if (!fPost) return null;
  const fileName = fPost.split('/').pop() ?? fPost;

  const modelIds = useMemo(() => {
    if (!summary?.models) return [1];
    return (summary.models as Array<{ id?: string } | string>).map((m, i) =>
      typeof m === 'string' ? i + 1 : +(m.id?.slice(1) ?? i + 1),
    );
  }, [summary]);

  return (
    <div className="space-y-4">
      <Card
        title={<>Results <span className="ml-2 font-mono text-[11px] normal-case text-accent">{fileName}</span></>}
        actions={<FileClassBadge value="POSTERIOR" />}
      >
        {summary && (
          <div className="mb-4 grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-muted sm:grid-cols-4">
            <div>{summary.n_points?.toLocaleString()} points</div>
            <div>{summary.n_realizations?.toLocaleString()} samples / point</div>
            {summary.inv_time !== undefined && <div>{Number(summary.inv_time).toFixed(1)} s runtime</div>}
            {summary.t && <div>T ∈ [{Number(summary.t.min).toFixed(1)}, {Number(summary.t.max).toFixed(1)}]</div>}
          </div>
        )}
        {stats && (
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {stats.chi2 && <Sparkline data={stats.chi2} label="χ² per point" stroke="var(--color-warn)" />}
            {stats.t && <Sparkline data={stats.t} label="Temperature T" stroke="var(--color-info)" />}
            {stats.ev && <Sparkline data={stats.ev} label="Log evidence (EV)" />}
            {stats.n_unique && <Sparkline data={stats.n_unique} label="Unique samples" stroke="#c084fc" />}
          </div>
        )}
      </Card>

      {modelIds.map((im) => (
        <Card key={im} title={`Posterior profile — M${im}`}>
          <img
            src={`${api.profileUrl(fileName, im)}&t=${job.ended_at ?? 0}`}
            alt={`Posterior profile M${im}`}
            className="w-full rounded-lg bg-white"
            loading="lazy"
          />
        </Card>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------

export function RejectionView({ files }: { files: H5FileInfo[] }) {
  const { jobs, connected } = useJobs();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const rejectionJobs = useMemo(
    () => Object.values(jobs).filter((j) => j.kind === 'rejection'),
    [jobs],
  );
  const running = rejectionJobs.find((j) => j.status === 'running');
  const selected = (selectedId && jobs[selectedId]) || running || rejectionJobs[0];

  const lastDone = rejectionJobs.find((j) => j.status === 'done' && j.result.f_post_h5);
  const resultsJob = selected?.status === 'done' && selected.result.f_post_h5 ? selected : lastDone;

  return (
    <div className="grid gap-4 xl:grid-cols-[420px_1fr]">
      <div>
        <RejectionForm files={files} onStarted={setSelectedId} />
        {rejectionJobs.length > 0 && (
          <div className="mt-4">
            <Card title="History">
              <div className="space-y-1">
                {rejectionJobs.slice(0, 8).map((j) => (
                  <button
                    key={j.id}
                    type="button"
                    onClick={() => setSelectedId(j.id)}
                    className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs hover:bg-panel-2 ${
                      selected?.id === j.id ? 'bg-panel-2 ring-1 ring-accent/30' : ''
                    }`}
                  >
                    <StatusBadge value={j.status} />
                    <span className="min-w-0 flex-1 truncate font-mono text-muted">
                      {(j.params.f_post_h5 as string) || (j.result.f_post_h5?.split('/').pop() ?? j.id)}
                    </span>
                    <span className="text-muted/60">{j.started_at ? new Date(j.started_at * 1000).toLocaleTimeString() : ''}</span>
                  </button>
                ))}
              </div>
            </Card>
          </div>
        )}
      </div>

      <div className="space-y-4">
        {!connected && (
          <div className="rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
            Live connection lost — reconnecting…
          </div>
        )}
        {selected ? <JobMonitor job={selected} /> : (
          <Card>
            <div className="py-8 text-center text-sm text-muted">
              Configure the inversion and press <span className="text-accent font-semibold">Run inversion</span>.
            </div>
          </Card>
        )}
        {resultsJob && <ResultsPanel job={resultsJob} />}
      </div>
    </div>
  );
}
