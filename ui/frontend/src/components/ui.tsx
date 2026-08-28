/** Small dark-theme primitives shared across views. */

import { type ReactNode } from 'react';
import type { FileClass, JobStatus } from '../lib/types';

export function Card({ title, children, actions }: { title?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <div className="rounded-xl border border-edge bg-panel p-5 shadow-lg shadow-black/20">
      {(title || actions) && (
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold tracking-wide text-muted uppercase">{title}</h3>
          {actions}
        </div>
      )}
      {children}
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[13px] font-medium text-muted">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted/70">{hint}</span>}
    </label>
  );
}

const inputCls =
  'w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 text-sm text-fg placeholder:text-muted/50 outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20 disabled:opacity-50';

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`${inputCls} ${props.className ?? ''}`} />;
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${inputCls} ${props.className ?? ''}`} />;
}

export function Button({
  variant = 'primary',
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' | 'danger' }) {
  const styles = {
    primary: 'bg-accent text-black hover:bg-accent/90 font-semibold',
    ghost: 'border border-edge bg-panel-2 text-fg hover:border-accent/40',
    danger: 'border border-danger/40 bg-danger/10 text-danger hover:bg-danger/20',
  }[variant];
  return (
    <button
      {...props}
      className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${props.className ?? ''}`}
    />
  );
}

const CLASS_BADGE: Record<FileClass, string> = {
  PRIOR: 'bg-sky-400/10 text-info border-sky-400/30',
  DATA: 'bg-amber-400/10 text-warn border-amber-400/30',
  POSTERIOR: 'bg-emerald-400/10 text-accent border-emerald-400/30',
  UNKNOWN: 'bg-zinc-400/10 text-muted border-zinc-400/30',
  UNREADABLE: 'bg-red-400/10 text-danger border-red-400/30',
};

export function FileClassBadge({ value }: { value: FileClass }) {
  return (
    <span className={`inline-block rounded-md border px-2 py-0.5 text-[11px] font-semibold tracking-wide ${CLASS_BADGE[value]}`}>
      {value}
    </span>
  );
}

const STATUS_BADGE: Record<JobStatus, string> = {
  pending: 'bg-zinc-400/10 text-muted border-zinc-400/30',
  running: 'bg-sky-400/10 text-info border-sky-400/30',
  done: 'bg-emerald-400/10 text-accent border-emerald-400/30',
  error: 'bg-red-400/10 text-danger border-red-400/30',
  cancelled: 'bg-amber-400/10 text-warn border-amber-400/30',
};

export function StatusBadge({ value }: { value: JobStatus }) {
  return (
    <span className={`inline-block rounded-md border px-2 py-0.5 text-[11px] font-semibold tracking-wide uppercase ${STATUS_BADGE[value]}`}>
      {value}
    </span>
  );
}

export function ProgressBar({ current, total }: { current: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-panel-2">
      <div
        className="h-full rounded-full bg-gradient-to-r from-accent/70 to-accent transition-[width] duration-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
