/** Query Volume view: probability map → interactive region growing → volumes. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { api } from '../lib/api';
import type { GeoParams, H5FileInfo, PriorModelsResponse, QueryTranslation, VolumeGrowResponse, VolumeProbResponse, VolumeVolumesResponse } from '../lib/types';
import { Button, Card, Field, Select, TextInput } from '../components/ui';
import { LLMSection } from '../components/LLMSection';
import { useLlm } from '../lib/useLlm';

const DEFAULT_GEO: GeoParams = { hull_ratio: 0.10, edge_buffer: null, cell_area_k: 6.0, elong_max: 4.0 };

interface AreaState {
  name: string;
  center: [number, number];
  params: { p_min: number; max_area_m2: number | null };
  geo: GeoParams;
  indices: number[];
  area_m2: number;
  polygon: [number, number][];
}

function ModelTable({ data }: { data: PriorModelsResponse }) {
  if (!data.models.length) {
    return <div className="text-xs text-muted">No prior models found in the prior file.</div>;
  }
  return (
    <table className="w-full text-left text-sm">
      <thead>
        <tr className="border-b border-edge text-xs text-muted">
          <th className="py-1.5 pr-3">im</th>
          <th className="py-1.5 pr-3">Name</th>
          <th className="py-1.5 pr-3">Kind</th>
          <th className="py-1.5">Depth (m)</th>
        </tr>
      </thead>
      <tbody>
        {data.models.map((m) => (
          <tr key={m.im} className="border-b border-edge/40 align-top">
            <td className="py-1.5 pr-3 font-mono text-xs">{m.im}</td>
            <td className="py-1.5 pr-3">{m.name}</td>
            <td className="py-1.5 pr-3 text-xs text-muted">{m.kind}</td>
            <td className="py-1.5 text-xs text-muted">
              {m.depth_min.toFixed(1)} – {m.depth_max.toFixed(1)} m
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** matplotlib "hot" colormap, sampled at t in [0, 1] (piecewise-linear R/G/B
 *  segments straight from matplotlib's `_hot_data`). */
function hotRGB(t: number): [number, number, number] {
  const seg = (v: number, a: number, b: number) => (v <= a ? 0 : v >= b ? 1 : (v - a) / (b - a));
  const r = 0.0416 + (1 - 0.0416) * seg(t, 0, 0.365079);
  const g = seg(t, 0.365079, 0.746032);
  const b = seg(t, 0.746032, 1);
  return [r, g, b];
}

/** matplotlib `cmap='hot_r'`: reversed hot — P=0 → white, P=1 → near-black. */
function hotR(t: number): string {
  const u = Math.max(0, Math.min(1, t));
  const [r, g, b] = hotRGB(1 - u);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/** CSS gradient string mirroring the hot_r colorbar (left = 0, right = 1). */
const HOT_R_GRADIENT = Array.from({ length: 9 }, (_, i) => hotR(i / 8)).join(', ');

/** Region outline: a wide white stroke with a narrow black stroke on top, so it
 *  stays readable over any hot_r shade (white → black). `dashed` marks the live
 *  preview; solid marks committed areas. */
function HaloPolygon({ points, dashed }: { points: string; dashed?: boolean }) {
  return (
    <>
      <polygon points={points} fill="none" stroke="#ffffff" strokeWidth={4} strokeLinejoin="round" />
      <polygon
        points={points}
        fill="none"
        stroke="#000000"
        strokeWidth={1.75}
        strokeLinejoin="round"
        strokeDasharray={dashed ? '6 4' : undefined}
      />
    </>
  );
}

function HaloCircle({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  return (
    <>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#ffffff" strokeWidth={4} />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#000000" strokeWidth={1.75} />
    </>
  );
}

function ProbabilityMap({ map, areas, preview, pendingCenter, onPointClick }: {
  map: VolumeProbResponse;
  areas: AreaState[];
  preview: VolumeGrowResponse | null;
  pendingCenter: [number, number] | null;
  onPointClick: (pt: [number, number]) => void;
}) {
  const W = 900, H = 700, PAD = 40;
  const { x, y, p, good } = map;
  // Equal-aspect fit (sounding coords are meter-scale UTM):
  const x0 = Math.min(...x), x1 = Math.max(...x);
  const y0 = Math.min(...y), y1 = Math.max(...y);
  const sx = (W - 2 * PAD) / Math.max(1e-9, x1 - x0);
  const sy = (H - 2 * PAD) / Math.max(1e-9, y1 - y0);
  const s = Math.min(sx, sy);
  const px = (v: number) => PAD + (v - x0) * s;
  const py = (v: number) => H - PAD - (v - y0) * s;   // flip Y

  const ring = (pts: [number, number][]) =>
    pts.map(([cx, cy]) => `${px(cx).toFixed(1)},${py(cy).toFixed(1)}`).join(' ');

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * W;
    const my = ((e.clientY - rect.top) / rect.height) * H;
    const cx = x0 + (mx - PAD) / s;
    const cy = y0 + (H - PAD - my) / s;
    // nearest KEPT sounding (matches find_coherent_area's seed resolution):
    let best = -1, bd = Infinity;
    for (let i = 0; i < x.length; i++) {
      if (!good[i]) continue;
      const d = (x[i] - cx) ** 2 + (y[i] - cy) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0) onPointClick([x[best], y[best]]);
  };

  return (
    <div className="overflow-x-auto">
      <svg width={W} height={H} className="rounded-lg border border-edge bg-white cursor-crosshair text-neutral-500"
           onClick={handleClick}>
        <polygon points={ring(map.boundary)} fill="none" stroke="currentColor" strokeOpacity={0.6} strokeDasharray="4 3" />
        {x.map((vx, i) => (
          <circle key={i} cx={px(vx)} cy={py(y[i])} r={2.2}
                  fill={good[i] ? hotR(p[i]) : '#9ca3af'}
                  opacity={good[i] ? 1 : 0.55} />
        ))}
        {areas.map((a) => (
          <HaloPolygon key={a.name} points={ring(a.polygon)} />
        ))}
        {preview && <HaloPolygon points={ring(preview.polygon)} dashed />}
        {pendingCenter && (
          <HaloCircle cx={px(pendingCenter[0])} cy={py(pendingCenter[1])} r={9} />
        )}
        {areas.map((a) => (
          <HaloCircle key={'c' + a.name} cx={px(a.center[0])} cy={py(a.center[1])} r={6} />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          P = 0
          <span
            className="inline-block h-3 w-40 rounded border border-edge"
            style={{ background: `linear-gradient(to right, ${HOT_R_GRADIENT})` }}
          />
          1 <span className="text-muted/70">(hot_r)</span>
        </span>
        <span>{map.n} soundings — grey = edge-affected (dropped), dashed grey = survey outline,
          black/white outline = region (dashed = live preview, solid = added areas).
          Click a sounding to set the region center.</span>
      </div>
    </div>
  );
}

export function VolumeView({ files }: { files: H5FileInfo[] }) {
  const llm = useLlm();

  const [filterText, setFilterText] = useState('');
  const [selected, setSelected] = useState('');
  const [modelInfo, setModelInfo] = useState<PriorModelsResponse | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // A. probability map
  const [probText, setProbText] = useState('');
  const [probJson, setProbJson] = useState('');
  const probJsonRef = useRef('');
  const [probJsonEdited, setProbJsonEdited] = useState(false);
  const [probInterp, setProbInterp] = useState<string | null>(null);
  const [map, setMap] = useState<VolumeProbResponse | null>(null);
  const probTranslation = useRef<QueryTranslation | null>(null);

  // B. region growing
  const [geo, setGeo] = useState<GeoParams>(DEFAULT_GEO);
  const [pMin, setPMin] = useState(0.2);
  const [maxArea, setMaxArea] = useState('');
  const [pendingCenter, setPendingCenter] = useState<[number, number] | null>(null);
  const [areas, setAreas] = useState<AreaState[]>([]);
  const [preview, setPreview] = useState<VolumeGrowResponse | null>(null);
  const [growInfo, setGrowInfo] = useState<string | null>(null);

  // C. volumes
  const [thickText, setThickText] = useState('');
  const [thickJson, setThickJson] = useState('');
  const thickJsonRef = useRef('');
  const [, setThickJsonEdited] = useState(false);
  const [thickInterp, setThickInterp] = useState<string | null>(null);
  const [volumes, setVolumes] = useState<VolumeVolumesResponse | null>(null);
  const thickTranslation = useRef<QueryTranslation | null>(null);

  const [busyProb, setBusyProb] = useState(false);
  const [busyGrow, setBusyGrow] = useState(false);
  const [busyVol, setBusyVol] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [probJsonError, setProbJsonError] = useState<string | null>(null);
  const [thickJsonError, setThickJsonError] = useState<string | null>(null);
  const [thickWarning, setThickWarning] = useState<string | null>(null);

  const posteriors = useMemo(() => files.filter((f) => f.class === 'POSTERIOR'), [files]);
  const filtered = useMemo(
    () => posteriors.filter((f) => !filterText || f.name.includes(filterText)),
    [posteriors, filterText],
  );

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    api
      .queryModels(selected)
      .then((r) => !cancelled && setModelInfo(r))
      .catch((e) => !cancelled && setModelsError((e as Error).message));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const resetWorkflow = () => {
    setProbText('');
    setProbJson('');
    probJsonRef.current = '';
    setProbJsonEdited(false);
    setProbInterp(null);
    probTranslation.current = null;
    setMap(null);
    setPreview(null);
    setPendingCenter(null);
    setAreas([]);
    setGrowInfo(null);
    setThickText('');
    setThickJson('');
    thickJsonRef.current = '';
    setThickJsonEdited(false);
    setThickInterp(null);
    thickTranslation.current = null;
    setVolumes(null);
    setError(null);
    setProbJsonError(null);
    setThickJsonError(null);
    setThickWarning(null);
  };

  const setJsonText = (
    v: string,
    setJson: (s: string) => void,
    ref: React.MutableRefObject<string>,
    setEdited: (b: boolean) => void,
  ) => {
    setJson(v);
    ref.current = v;
    setEdited(true);
  };

  // ---- A: translate (skip when hand-edited) → /prob ----
  const runProb = async () => {
    setBusyProb(true);
    setError(null);
    setMap(null);
    setPreview(null);
    setPendingCenter(null);
    setAreas([]);
    setGrowInfo(null);
    setVolumes(null);
    try {
      let dict: Record<string, unknown>;
      try {
        dict = JSON.parse(probJsonRef.current || probJson || '{}') as Record<string, unknown>;
      } catch (e) {
        setProbJsonError((e as Error).message);
        return;
      }
      setProbJsonError(null);
      if (!probJsonEdited) {
        const tr = await api.translateQuery({ f: selected, text: probText, ...llm.llmParams() });
        probTranslation.current = tr;
        setProbInterp(tr.interpretation);
        const json = JSON.stringify(tr.query_dict, null, 2);
        setProbJson(json);
        probJsonRef.current = json;
        setProbJsonEdited(false);
        dict = tr.query_dict;
        if ('metric' in dict) {
          throw new Error('The LLM produced a percentile query — step A needs a probability question (e.g. "what is the probability that …").');
        }
      }
      const m = await api.volumeProb({ f: selected, query_dict: dict, geo });
      setMap(m);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyProb(false);
    }
  };

  // ---- B: grow one preview area ----
  const runGrow = useCallback(async () => {
    if (!map || !pendingCenter) return;
    setBusyGrow(true);
    setError(null);
    try {
      const max = parseFloat(maxArea);
      const gp = await api.volumeGrow({
        f: selected,
        p: map.p,
        p_min: pMin,
        x_center: pendingCenter[0],
        y_center: pendingCenter[1],
        max_area_m2: maxArea.trim() && isFinite(max) ? max : null,
        geo,
      });
      setPreview(gp);
      setGrowInfo(
        `Center snapped to sounding #${gp.seed} at (${gp.center[0].toFixed(0)}, ${gp.center[1].toFixed(0)}), P = ${gp.p_seed?.toFixed(3) ?? 'n/a'} — ` +
        `${gp.n_soundings} soundings, area ${gp.area_m2.toExponential(3)} m²`,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyGrow(false);
    }
  }, [map, pendingCenter, pMin, maxArea, geo, selected]);

  // Auto-grow: re-run whenever the center or the P_MIN / MAX_AREA_M2 knobs
  // change (grow is cheap). Debounced so typing a value doesn't spam requests.
  useEffect(() => {
    if (!map || !pendingCenter) return;
    const t = setTimeout(() => { void runGrow(); }, 300);
    return () => clearTimeout(t);
  }, [runGrow, map, pendingCenter]);

  const addArea = () => {
    if (!preview) return;
    const max = parseFloat(maxArea);
    const next: AreaState = {
      name: `Area ${areas.length + 1}`,
      center: preview.center,
      params: { p_min: pMin, max_area_m2: maxArea.trim() && isFinite(max) ? max : null },
      geo,
      indices: preview.indices,
      area_m2: preview.area_m2,
      polygon: preview.polygon,
    };
    setAreas([...areas, next]);
    setPreview(null);
    setPendingCenter(null);
    setGrowInfo(null);
  };

  const removeArea = (name: string) => {
    setAreas(areas.filter((a) => a.name !== name));
    setVolumes(null);
  };

  // ---- C: translate thickness question → compute volumes ----
  const translateThick = async () => {
    setBusyVol(true);
    setError(null);
    setThickWarning(null);
    try {
      const tr = await api.translateQuery({ f: selected, text: thickText, ...llm.llmParams() });
      thickTranslation.current = tr;
      setThickInterp(tr.interpretation);
      const json = JSON.stringify(tr.query_dict, null, 2);
      setThickJson(json);
      thickJsonRef.current = json;
      setThickJsonEdited(false);
      if (!('metric' in tr.query_dict)) {
        setThickWarning("The LLM produced a probability query — rephrase as a percentile question (e.g. 'P5/P50/P95 of …').");
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyVol(false);
    }
  };

  const runVolumes = async () => {
    setBusyVol(true);
    setError(null);
    setVolumes(null);
    try {
      let dict: Record<string, unknown>;
      try {
        dict = JSON.parse(thickJsonRef.current || thickJson) as Record<string, unknown>;
      } catch (e) {
        setThickJsonError((e as Error).message);
        return;
      }
      setThickJsonError(null);
      if (!('metric' in dict)) {
        setThickWarning('The query JSON has no "metric" key — compute volumes needs a percentile query (e.g. {"metric": …, "percentiles": [5, 50, 95]}).');
        return;
      }
      setThickWarning(null);
      const v = await api.volumeVolumes({
        f: selected,
        query_dict: dict,
        areas: areas.map((a) => ({ name: a.name, indices: a.indices })),
        text: thickText,
        interpretation: thickInterp ?? undefined,
        geo,
      });
      setVolumes(v);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyVol(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <div>
        <h2 className="text-lg font-bold">Query Volume</h2>
        <p className="mt-0.5 text-xs text-muted">
          Compute a probability map, grow coherent areas interactively, and compute probabilistic
          volumes (P5/P50/P95) per area — translated by an LLM.
        </p>
      </div>
      <Card title="LLM">
        <LLMSection llm={llm} />
      </Card>

      {llm.ready && (
        <>
          <Card title="Posterior File">
            {!posteriors.length ? (
              <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
                No posterior HDF5 files found in the workspace (files must contain an i_use
                dataset).
              </div>
            ) : (
              <div className="space-y-3">
                <Field label="Filter filenames:">
                  <TextInput
                    value={filterText}
                    onChange={(e) => setFilterText(e.target.value)}
                    placeholder="e.g. DAUGAARD"
                  />
                </Field>
                <Field label="Select posterior file:">
                  <Select
                    value={selected}
                    onChange={(e) => {
                      setSelected(e.target.value);
                      setModelInfo(null);
                      setModelsError(null);
                      resetWorkflow();
                    }}
                  >
                    <option value="">— choose —</option>
                    {filtered.map((f) => (
                      <option key={f.name} value={f.name}>
                        {f.name}
                      </option>
                    ))}
                  </Select>
                </Field>
                {selected && !filtered.some((f) => f.name === selected) && (
                  <div className="text-xs text-warn">No posterior files match '{filterText}'.</div>
                )}
                {modelsError && (
                  <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
                    {modelsError}
                  </div>
                )}
                {modelInfo && (
                  <div className="text-xs text-muted">
                    Prior file: <code>{modelInfo.f_prior_h5}</code>
                  </div>
                )}
              </div>
            )}
          </Card>

          {modelInfo && (
            <Card title="Available Prior Models">
              <ModelTable data={modelInfo} />
              {modelInfo.describe && (
                <details className="mt-3 rounded-lg border border-edge bg-panel-2 px-3 py-2">
                  <summary className="cursor-pointer text-xs font-semibold text-muted">
                    Model parameters (classes, names)
                  </summary>
                  <pre className="mt-2 max-h-72 overflow-auto text-xs whitespace-pre-wrap text-fg">
                    {modelInfo.describe}
                  </pre>
                </details>
              )}
            </Card>
          )}

          {modelInfo && (
            <Card title="A. Probability map">
              <div className="space-y-4">
                <Field label="Enter probability query in plain English:">
                  <textarea
                    value={probText}
                    onChange={(e) => setProbText(e.target.value)}
                    rows={3}
                    placeholder="e.g. What is the probability that cumulative raw material (sand and gravel) thickness exceeds 10 m within the top 30 m?"
                    className="w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 text-sm text-fg outline-none transition placeholder:text-muted/50 focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                  />
                </Field>
                <Button onClick={runProb} disabled={busyProb || !probText.trim() || !selected}>
                  <Search size={14} />
                  {busyProb ? 'Running…' : probJsonEdited ? 'Run probability query (edited JSON)' : 'Run probability query'}
                </Button>
                {probInterp && (
                  <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-sm text-info">
                    <span className="font-semibold">Interpretation:</span> {probInterp}
                  </div>
                )}
                <details className="rounded-lg border border-edge bg-panel-2 px-3 py-2">
                  <summary className="cursor-pointer text-xs font-semibold text-muted">
                    Query JSON (optional advanced view — edit and run again to apply)
                  </summary>
                  <div className="mt-2 space-y-2">
                    <textarea
                      value={probJson}
                      onChange={(e) => setJsonText(e.target.value, setProbJson, probJsonRef, setProbJsonEdited)}
                      rows={12}
                      spellCheck={false}
                      className="w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 font-mono text-xs text-fg outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                    />
                    {probJsonError && (
                      <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                        Invalid JSON: {probJsonError}
                      </div>
                    )}
                    <Button onClick={runProb} disabled={busyProb || !selected}>
                      <Search size={14} />
                      {busyProb ? 'Running…' : 'Run probability query'}
                    </Button>
                  </div>
                </details>
                {error && (
                  <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs whitespace-pre-wrap text-danger">
                    {error}
                  </div>
                )}
                {map && (
                  <div className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-sm text-accent">
                    Done. Mean probability: {map.mean_probability.toFixed(3)}
                    {'  |  '}
                    N soundings: {map.n}
                    {'  |  '}
                    Edge-affected dropped: {map.n_dropped}
                  </div>
                )}
              </div>
            </Card>
          )}

          {map && (
            <Card title="B. Interactive region growing">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Field label="P_MIN (inclusion cutoff):">
                    <TextInput type="number" step={0.01} min={0} max={1} value={pMin}
                               onChange={(e) => setPMin(parseFloat(e.target.value) || 0)} />
                  </Field>
                  <Field label="MAX_AREA_M2 (empty = no cap):">
                    <TextInput value={maxArea} onChange={(e) => setMaxArea(e.target.value)} placeholder="e.g. 1000000" />
                  </Field>
                </div>
                <details className="rounded-lg border border-edge bg-panel-2 px-3 py-2">
                  <summary className="cursor-pointer text-xs font-semibold text-muted">
                    Advanced geometry (editing these requires re-running step A)
                  </summary>
                  <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Field label="hull_ratio:">
                      <TextInput type="number" step={0.05} min={0} max={1} value={geo.hull_ratio}
                                 onChange={(e) => { setGeo({ ...geo, hull_ratio: parseFloat(e.target.value) || 0 }); setMap(null); setAreas([]); setPreview(null); }} />
                    </Field>
                    <Field label="edge_buffer (empty = auto):">
                      <TextInput value={geo.edge_buffer ?? ''} onChange={(e) => { setGeo({ ...geo, edge_buffer: e.target.value.trim() === '' ? null : parseFloat(e.target.value) || null }); setMap(null); setAreas([]); setPreview(null); }} />
                    </Field>
                    <Field label="cell_area_k:">
                      <TextInput type="number" step={0.5} value={geo.cell_area_k}
                                 onChange={(e) => { setGeo({ ...geo, cell_area_k: parseFloat(e.target.value) || 6 }); setMap(null); setAreas([]); setPreview(null); }} />
                    </Field>
                    <Field label="elong_max (empty = skip):">
                      <TextInput value={geo.elong_max ?? ''} onChange={(e) => { setGeo({ ...geo, elong_max: e.target.value.trim() === '' ? null : parseFloat(e.target.value) || null }); setMap(null); setAreas([]); setPreview(null); }} />
                    </Field>
                  </div>
                  <div className="mt-2 text-xs text-warn">
                    Editing geometry clears the map and defined areas — re-run step A to rebuild.
                  </div>
                </details>
                <ProbabilityMap
                  map={map}
                  areas={areas}
                  preview={preview}
                  pendingCenter={pendingCenter}
                  onPointClick={setPendingCenter}
                />
                <div className="flex flex-wrap items-center gap-3">
                  <Button variant="ghost" onClick={() => void runGrow()} disabled={busyGrow || !pendingCenter}>
                    {busyGrow ? 'Growing…' : 'Update preview'}
                  </Button>
                  <Button onClick={addArea} disabled={!preview}>
                    Add area
                  </Button>
                  <span className="text-xs text-muted">
                    {pendingCenter
                      ? `Center: (${pendingCenter[0].toFixed(0)}, ${pendingCenter[1].toFixed(0)}) — grows automatically as you change P_MIN / MAX_AREA_M2 or click a new point`
                      : 'Click a sounding on the map to set the region center.'}
                  </span>
                </div>
                {growInfo && <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-xs text-info">{growInfo}</div>}
                {areas.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-edge text-xs text-muted">
                          <th className="py-1.5 pr-3">Area</th>
                          <th className="py-1.5 pr-3">Center</th>
                          <th className="py-1.5 pr-3">P_MIN</th>
                          <th className="py-1.5 pr-3">Area (m²)</th>
                          <th className="py-1.5 pr-3">Soundings</th>
                          <th className="py-1.5"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {areas.map((a) => (
                          <tr key={a.name} className="border-b border-edge/40">
                            <td className="py-1.5 pr-3">{a.name}</td>
                            <td className="py-1.5 pr-3 font-mono text-xs">
                              ({a.center[0].toFixed(0)}, {a.center[1].toFixed(0)})
                            </td>
                            <td className="py-1.5 pr-3 text-xs">{a.params.p_min}</td>
                            <td className="py-1.5 pr-3 text-xs text-muted">{a.area_m2.toExponential(3)}</td>
                            <td className="py-1.5 pr-3 text-xs text-muted">{a.indices.length}</td>
                            <td className="py-1.5 text-right">
                              <Button variant="danger" onClick={() => removeArea(a.name)}>Remove</Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </Card>
          )}

          {map && areas.length > 0 && (
            <Card title="C. Volume computation">
              <div className="space-y-4">
                <Field label="Thickness property (percentile question):">
                  <textarea
                    value={thickText}
                    onChange={(e) => setThickText(e.target.value)}
                    rows={3}
                    placeholder="e.g. thickness of raw material lying under no more than 5 m of overburden and no deeper than 50 m."
                    className="w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 text-sm text-fg outline-none transition placeholder:text-muted/50 focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                  />
                </Field>
                <div className="flex flex-wrap items-center gap-3">
                  <Button onClick={translateThick} disabled={busyVol || !thickText.trim() || !selected}>
                    {busyVol ? 'Translating…' : 'Translate thickness query'}
                  </Button>
                  <Button onClick={runVolumes} disabled={busyVol || !thickJson.trim()}>
                    {busyVol ? 'Computing…' : 'Compute volumes'}
                  </Button>
                </div>
                {thickInterp && (
                  <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-sm text-info">
                    <span className="font-semibold">Interpretation:</span> {thickInterp}
                  </div>
                )}
                <details className="rounded-lg border border-edge bg-panel-2 px-3 py-2">
                  <summary className="cursor-pointer text-xs font-semibold text-muted">
                    Query JSON (optional advanced view — edit and compute again to apply)
                  </summary>
                  <div className="mt-2 space-y-2">
                    <textarea
                      value={thickJson}
                      onChange={(e) => setJsonText(e.target.value, setThickJson, thickJsonRef, setThickJsonEdited)}
                      rows={12}
                      spellCheck={false}
                      className="w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 font-mono text-xs text-fg outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                    />
                    {thickJsonError && (
                      <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                        Invalid JSON: {thickJsonError}
                      </div>
                    )}
                    <Button onClick={runVolumes} disabled={busyVol || !thickJson.trim()}>
                      Compute volumes
                    </Button>
                  </div>
                </details>
                {thickWarning && (
                  <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
                    {thickWarning}
                  </div>
                )}
                {volumes && (
                  <>
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-edge text-xs text-muted">
                          <th className="py-1.5 pr-3">Area</th>
                          <th className="py-1.5 pr-3">Soundings</th>
                          {volumes.percentiles.map((p) => (
                            <th key={p} className="py-1.5 pr-3">P{p} (m³)</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {volumes.areas.map((a) => (
                          <tr key={a.name} className="border-b border-edge/40">
                            <td className="py-1.5 pr-3">{a.name}</td>
                            <td className="py-1.5 pr-3 text-xs text-muted">{a.n_soundings}</td>
                            {volumes.percentiles.map((p, k) => (
                              <td key={p} className="py-1.5 pr-3 font-mono text-xs">
                                {volumes.areas.find((x) => x.name === a.name)!.volumes[k].toExponential(3)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <img
                      src={`data:image/png;base64,${volumes.figure}`}
                      alt="Volume per grown area"
                      className="w-full max-w-2xl rounded-lg border border-edge"
                    />
                  </>
                )}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}