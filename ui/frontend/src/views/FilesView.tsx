/** Files view: workspace .h5 browser with detail inspector. */

import { useEffect, useState } from 'react';
import { Database, FolderOpen, RefreshCw, Upload } from 'lucide-react';
import { api } from '../lib/api';
import type { H5FileInfo, H5Node, H5Summary } from '../lib/types';
import { Button, Card, FileClassBadge } from '../components/ui';

function formatMb(mb: number): string {
  return mb >= 1000 ? `${(mb / 1000).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

function TreeNode({ node, depth }: { node: H5Node; depth: number }) {
  const [open, setOpen] = useState(depth < 1);
  const attrs = Object.entries(node.attrs ?? {});
  return (
    <div style={{ paddingLeft: depth * 14 }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left font-mono text-xs hover:bg-panel-2"
      >
        <span className={node.kind === 'group' ? 'text-info' : 'text-warn'}>
          {node.kind === 'group' ? (open ? '▾' : '▸') : '·'}
        </span>
        <span className={node.kind === 'group' ? 'text-fg font-semibold' : 'text-fg/80'}>{node.name}</span>
        {node.shape && <span className="text-muted">[{node.shape.join(' × ')}] {node.dtype}</span>}
        {node.size_mb !== undefined && node.size_mb > 0.01 && (
          <span className="text-muted/60">{node.size_mb.toFixed(1)} MB</span>
        )}
      </button>
      {open && attrs.length > 0 && (
        <div style={{ paddingLeft: 18 }} className="my-0.5 border-l border-edge pl-2">
          {attrs.map(([k, v]) => (
            <div key={k} className="font-mono text-[11px] text-muted">
              <span className="text-accent/80">@{k}</span> = {JSON.stringify(v)}
            </div>
          ))}
        </div>
      )}
      {open && node.children?.map((c) => <TreeNode key={c.path} node={c} depth={depth + 1} />)}
    </div>
  );
}

function Inspector({ name, onClose }: { name: string; onClose: () => void }) {
  const [summary, setSummary] = useState<H5Summary | null>(null);
  const [tree, setTree] = useState<H5Node | null>(null);
  const [tab, setTab] = useState<'summary' | 'tree'>('summary');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSummary(null);
    setTree(null);
    setError(null);
    api.fileSummary(name).then(setSummary).catch((e) => setError(String(e.message ?? e)));
    api.fileTree(name).then(setTree).catch(() => {});
  }, [name]);

  return (
    <Card
      title={<span className="font-mono normal-case">{name}</span>}
      actions={
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={() => setTab('summary')} className={tab === 'summary' ? 'border-accent/50' : ''}>Summary</Button>
          <Button variant="ghost" onClick={() => setTab('tree')} className={tab === 'tree' ? 'border-accent/50' : ''}>Tree</Button>
          <Button variant="ghost" onClick={onClose}>✕</Button>
        </div>
      }
    >
      {error && <div className="text-danger text-sm">{error}</div>}
      {!summary && !error && <div className="text-sm text-muted">Loading…</div>}

      {tab === 'summary' && summary && (
        <div className="space-y-3 text-sm">
          <div className="flex items-center gap-3">
            <FileClassBadge value={summary.class} />
            {summary.n_realizations !== undefined && (
              <span className="text-muted">{summary.n_realizations.toLocaleString()} realizations</span>
            )}
            {summary.n_points !== undefined && (
              <span className="text-muted">{summary.n_points.toLocaleString()} data points</span>
            )}
          </div>

          {summary.class === 'PRIOR' && Array.isArray(summary.models) && (
            <div>
              <div className="mb-1 text-xs font-semibold text-muted uppercase">Model parameters</div>
              {summary.models.map((m) =>
                typeof m === 'string' ? null : (
                  <div key={m.id} className="flex items-baseline gap-2 font-mono text-xs">
                    <span className="text-warn">{m.id}</span>
                    <span>{m.name || '(unnamed)'}</span>
                    <span className="text-muted">{m.shape?.join(' × ')}</span>
                    {m.is_discrete && <span className="text-muted">discrete</span>}
                  </div>
                ),
              )}
              <div className="mt-2 text-xs">
                {summary.has_forward_data
                  ? <span className="text-accent">✓ Contains forward data ({summary.data?.map((d) => d.id).join(', ')}) — usable for inversion</span>
                  : <span className="text-warn">⚠ No forward data — run a forward model before inversion</span>}
              </div>
            </div>
          )}

          {summary.class === 'DATA' && summary.datasets && (
            <div>
              <div className="mb-1 text-xs font-semibold text-muted uppercase">Datasets</div>
              {summary.datasets.map((d) => (
                <div key={d.id} className="flex items-baseline gap-2 font-mono text-xs">
                  <span className="text-warn">{d.id}</span>
                  <span className="text-info">{d.noise_model}</span>
                  <span className="text-muted">{d.shape?.join(' × ')}</span>
                  {d.n_used !== undefined && <span className="text-muted">{d.n_used} used</span>}
                </div>
              ))}
            </div>
          )}

          {summary.class === 'POSTERIOR' && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              {summary.f5_data && <div><span className="text-muted">data:</span> <span className="font-mono">{summary.f5_data}</span></div>}
              {summary.f5_prior && <div><span className="text-muted">prior:</span> <span className="font-mono">{summary.f5_prior}</span></div>}
              {summary.inv_time !== undefined && <div><span className="text-muted">runtime:</span> {Number(summary.inv_time).toFixed(1)} s</div>}
            </div>
          )}
        </div>
      )}

      {tab === 'tree' && tree && (
        <div className="max-h-[50vh] overflow-auto rounded-lg border border-edge bg-bg p-2">
          <TreeNode node={tree} depth={0} />
        </div>
      )}
    </Card>
  );
}

export function FilesView({
  files,
  workspace,
  onRefresh,
  onSelect,
}: {
  files: H5FileInfo[];
  workspace: string;
  onRefresh: () => void;
  onSelect?: (name: string, cls: string) => void;
}) {
  const [inspect, setInspect] = useState<string | null>(null);
  const [filter, setFilter] = useState('');

  const visible = filter ? files.filter((f) => f.class === filter) : files;

  const upload = async (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    await fetch('/api/files/upload', { method: 'POST', body: form });
    onRefresh();
    ev.target.value = '';
  };

  return (
    <div className="space-y-4">
      <Card
        title={<>Workspace files <span className="ml-2 font-mono text-[11px] normal-case text-muted/60">{workspace}</span></>}
        actions={
          <div className="flex items-center gap-2">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="rounded-lg border border-edge bg-panel-2 px-2 py-1.5 text-xs text-fg"
            >
              <option value="">All types</option>
              <option value="PRIOR">PRIOR</option>
              <option value="DATA">DATA</option>
              <option value="POSTERIOR">POSTERIOR</option>
              <option value="UNKNOWN">UNKNOWN</option>
            </select>
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-edge bg-panel-2 px-3 py-1.5 text-sm hover:border-accent/40">
              <Upload size={14} /> Upload
              <input type="file" accept=".h5" className="hidden" onChange={upload} />
            </label>
            <Button variant="ghost" onClick={onRefresh}><RefreshCw size={14} /> Refresh</Button>
          </div>
        }
      >
        {visible.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-10 text-muted">
            <FolderOpen size={32} className="opacity-40" />
            <span className="text-sm">No .h5 files in this workspace</span>
          </div>
        )}
        <div className="divide-y divide-edge/60">
          {visible.map((f) => (
            <div key={f.name} className="flex items-center gap-3 py-2.5">
              <Database size={16} className="shrink-0 text-muted" />
              <button
                type="button"
                onClick={() => setInspect(f.name)}
                className="min-w-0 flex-1 truncate text-left font-mono text-sm hover:text-accent"
              >
                {f.name}
              </button>
              <span className="w-20 text-right text-xs text-muted">{formatMb(f.size_mb)}</span>
              <FileClassBadge value={f.class} />
              {onSelect && (
                <Button variant="ghost" className="px-2 py-1 text-xs" onClick={() => onSelect(f.name, f.class)}>
                  Use
                </Button>
              )}
            </div>
          ))}
        </div>
      </Card>
      {inspect && <Inspector name={inspect} onClose={() => setInspect(null)} />}
    </div>
  );
}
