/** Minimal SVG line/area chart for POST stat series. */

interface Props {
  data: (number | null)[];
  height?: number;
  stroke?: string;
  label?: string;
}

export function Sparkline({ data, height = 72, stroke = 'var(--color-accent)', label }: Props) {
  const values = data.filter((v): v is number => v !== null);
  if (values.length < 2) return <div className="text-xs text-muted">no data</div>;

  const w = 600;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;

  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = v === null ? null : height - 4 - ((v - min) / span) * (height - 8);
    return { x, y };
  });

  let d = '';
  let pen = false;
  for (const p of pts) {
    if (p.y === null) {
      pen = false;
      continue;
    }
    d += `${pen ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`;
    pen = true;
  }

  return (
    <div>
      {label && <div className="mb-1 text-[11px] font-medium tracking-wide text-muted uppercase">{label}</div>}
      <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className="w-full" style={{ height }}>
        <path d={d} fill="none" stroke={stroke} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="mt-0.5 flex justify-between font-mono text-[10px] text-muted/70">
        <span>{min.toPrecision(3)}</span>
        <span>{max.toPrecision(3)}</span>
      </div>
    </div>
  );
}
