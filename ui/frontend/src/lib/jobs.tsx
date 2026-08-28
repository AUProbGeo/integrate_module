/** Global job state: one WebSocket streams all job events. */

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { api } from './api';
import type { Job } from './types';

interface JobsState {
  jobs: Record<string, Job>;
  logs: Record<string, string[]>;
  connected: boolean;
  refresh: () => Promise<void>;
}

const JobsContext = createContext<JobsState>({ jobs: {}, logs: {}, connected: false, refresh: async () => {} });

const MAX_LOG_LINES = 1000;

type RetryHandle = number;

export function JobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [logs, setLogs] = useState<Record<string, string[]>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const refresh = async () => {
    const { jobs: list } = await api.listJobs();
    setJobs(Object.fromEntries(list.map((j) => [j.id, j])));
  };

  useEffect(() => {
    let closed = false;
    let retry: RetryHandle | undefined;

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/api/ws`);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(connect, 2000);
      };
      ws.onmessage = (msg) => {
        const ev = JSON.parse(msg.data);
        if (ev.type === 'jobs') {
          setJobs(Object.fromEntries((ev.jobs as Job[]).map((j) => [j.id, j])));
        } else if (ev.type === 'job') {
          const j = ev.job as Job;
          setJobs((prev) => ({ ...prev, [j.id]: j }));
        } else if (ev.type === 'progress') {
          const id = ev.job_id as string;
          setJobs((prev) => {
            const job = prev[id];
            if (!job) return prev;
            return {
              ...prev,
              [id]: {
                ...job,
                progress: { current: ev.current, total: ev.total, info: ev.info ?? {} },
              },
            };
          });
        } else if (ev.type === 'log') {
          const id = ev.job_id as string;
          setLogs((prev) => {
            const lines = [...(prev[id] ?? []), ev.line as string];
            return { ...prev, [id]: lines.slice(-MAX_LOG_LINES) };
          });
        }
      };
    };

    // Initial snapshot in case WS connects after jobs already ran.
    refresh().catch(() => {});
    connect();
    return () => {
      closed = true;
      if (retry !== undefined) window.clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  return (
    <JobsContext.Provider value={{ jobs, logs, connected, refresh }}>
      {children}
    </JobsContext.Provider>
  );
}

export const useJobs = () => useContext(JobsContext);
