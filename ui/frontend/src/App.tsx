import { useCallback, useEffect, useMemo, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { Layers, Database, Waves, BarChart3, Activity, FlaskConical, Search, Boxes } from 'lucide-react';
import { api } from './lib/api';
import { JobsProvider, useJobs } from './lib/jobs';
import type { H5FileInfo } from './lib/types';
import { FilesView } from './views/FilesView';
import { QueryView } from './views/QueryView';
import { VolumeView } from './views/VolumeView';
import { RejectionView } from './views/RejectionView';

function SoonView({ name, detail }: { name: string; detail: string }) {
  return (
    <div className="rounded-xl border border-dashed border-edge bg-panel/40 p-10 text-center">
      <FlaskConical size={28} className="mx-auto mb-3 text-muted/50" />
      <div className="text-sm font-semibold">{name}</div>
      <div className="mx-auto mt-1 max-w-md text-xs text-muted">{detail}</div>
    </div>
  );
}

function Shell() {
  const [files, setFiles] = useState<H5FileInfo[]>([]);
  const [workspace, setWorkspace] = useState('');
  const { jobs, connected } = useJobs();

  const reloadFiles = useCallback(async () => {
    try {
      const r = await api.listFiles();
      setFiles(r.files);
      setWorkspace(r.workspace);
    } catch {
      /* backend unreachable */
    }
  }, []);

  useEffect(() => { reloadFiles(); }, [reloadFiles]);

  // A job finishing usually creates a new POST file — refresh the file list.
  const doneIds = useMemo(
    () => Object.values(jobs).filter((j) => j.status === 'done').map((j) => j.id).join(','),
    [jobs],
  );
  useEffect(() => {
    if (doneIds) reloadFiles();
  }, [doneIds, reloadFiles]);

  const runningCount = Object.values(jobs).filter((j) => j.status === 'running').length;

  const nav = [
    { to: '/', label: 'Rejection', icon: <Waves size={16} />, end: true },
    { to: '/files', label: 'Files', icon: <Database size={16} />, end: false },
    { to: '/query', label: 'Query', icon: <Search size={16} />, end: false },
    { to: '/query-volume', label: 'Query Volume', icon: <Boxes size={16} />, end: false },
  ];
  const soon = [
    { to: '/prior', label: 'Prior models', icon: <Layers size={16} /> },
    { to: '/forward', label: 'Forward', icon: <Activity size={16} /> },
    { to: '/plot', label: 'Plotting', icon: <BarChart3 size={16} /> },
  ];

  return (
    <div className="flex h-full">
      <aside className="flex w-52 shrink-0 flex-col border-r border-edge bg-panel">
        <div className="flex items-center gap-2 border-b border-edge px-4 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 font-mono text-sm font-bold text-accent">ig</div>
          <div>
            <div className="text-sm font-bold tracking-wide">INTEGRATE</div>
            <div className="text-[10px] text-muted">probabilistic integration</div>
          </div>
        </div>
        <nav className="flex-1 space-y-1 p-2">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? 'bg-accent/10 font-semibold text-accent' : 'text-muted hover:bg-panel-2 hover:text-fg'
                }`
              }
            >
              {n.icon}
              {n.label}
              {n.to === '/' && runningCount > 0 && (
                <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-accent" />
              )}
            </NavLink>
          ))}
          <div className="px-3 pt-4 pb-1 text-[10px] font-semibold tracking-widest text-muted/60 uppercase">Coming</div>
          {soon.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? 'bg-accent/10 font-semibold text-accent' : 'text-muted/70 hover:bg-panel-2 hover:text-fg'
                }`
              }
            >
              {n.icon}
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-edge px-4 py-3">
          <div className={`flex items-center gap-2 text-[11px] ${connected ? 'text-accent' : 'text-warn'}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-accent' : 'bg-warn animate-pulse'}`} />
            {connected ? 'connected' : 'reconnecting…'}
          </div>
          <div className="mt-1 truncate font-mono text-[10px] text-muted/60" title={workspace}>{workspace}</div>
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-y-auto p-6">
        <Routes>
          <Route path="/" element={<RejectionView files={files} />} />
          <Route path="/files" element={<FilesView files={files} workspace={workspace} onRefresh={reloadFiles} />} />
          <Route path="/query" element={<QueryView files={files} />} />
          <Route path="/query-volume" element={<VolumeView files={files} />} />
          <Route path="/prior" element={<SoonView name="Prior model generation" detail="prior_model_layered(), prior_model_workbench() and prior_model_workbench_direct() — arriving on this framework next." />} />
          <Route path="/forward" element={<SoonView name="Forward modelling" detail="forward_gaaem() — arriving next." />} />
          <Route path="/plot" element={<SoonView name="Plotting" detail="Selected integrate_plot functions — arriving next." />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <JobsProvider>
      <Shell />
    </JobsProvider>
  );
}
