/** Query view: natural-language posterior queries translated by an LLM. */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { api } from '../lib/api';
import type { H5FileInfo, PriorModelsResponse, QueryResult, QueryTranslation } from '../lib/types';
import { Button, Card, Field, Select, TextInput } from '../components/ui';
import { LLMSection } from '../components/LLMSection';
import { useLlm } from '../lib/useLlm';

function ModelTable({ data }: { data: PriorModelsResponse }) {
  if (!data.models.length) {
    return <div className="text-xs text-muted">No prior models found in the prior file.</div>;
  }
  return (
    <table className="w-full text-left text-sm">
      <thead>
        <tr className="border-b border-edge text-xs tracking-wide text-muted uppercase">
          <th className="py-1.5 pr-3">im</th>
          <th className="py-1.5 pr-3">Name</th>
          <th className="py-1.5 pr-3">Type</th>
          <th className="py-1.5">Depth range</th>
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

function QueryResultPanel({ result }: { result: QueryResult }) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-sm text-accent">
        Done.{' '}
        {result.kind === 'probability' ? (
          <>
            Mean probability: {result.mean_probability?.toFixed(3)}
            {'  |  '}
          </>
        ) : (
          <>Percentiles ({result.percentiles?.join(', ')}) computed{'  |  '}</>
        )}
        N locations: {result.n_locations}
      </div>
      {result.figures.map((b64, i) => (
        <img
          key={i}
          src={`data:image/png;base64,${b64}`}
          alt={`Query result figure ${i + 1}`}
          className="w-full max-w-3xl rounded-lg border border-edge"
        />
      ))}
    </div>
  );
}

export function QueryView({ files }: { files: H5FileInfo[] }) {
  const llm = useLlm();

  const [filterText, setFilterText] = useState('');
  const [selected, setSelected] = useState('');
  const [modelInfo, setModelInfo] = useState<PriorModelsResponse | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [translation, setTranslation] = useState<QueryTranslation | null>(null);
  const [jsonText, setJsonText] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [jsonOpen, setJsonOpen] = useState(false);
  const [jsonEdited, setJsonEdited] = useState(false);
  const [sysPrompt, setSysPrompt] = useState('');
  const jsonTextRef = useRef('');

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
    api
      .systemPrompt(selected)
      .then((r) => !cancelled && setSysPrompt(r.system_prompt))
      .catch(() => !cancelled && setSysPrompt(''));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const translate = async (params: {
    f: string;
    text: string;
    provider?: string;
    api_key?: string;
    model?: string;
    system_prompt?: string;
  }) => {
    const tr = await api.translateQuery(params);
    setTranslation(tr);
    const json = JSON.stringify(tr.query_dict, null, 2);
    setJsonText(json);
    jsonTextRef.current = json;   // always see the latest value from any async handler
    setJsonEdited(false);
  };

  const evaluate = async () => {
    let dict: Record<string, unknown>;
    try {
      dict = JSON.parse(jsonTextRef.current || jsonText) as Record<string, unknown>;
    } catch (e) {
      setJsonError((e as Error).message);
      return null;
    }
    setJsonError(null);
    return await api.evaluateQuery({
      f: selected,
      text,
      query_dict: dict,
      interpretation: translation?.interpretation,
    });
  };

  const runQuery = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const params = {
        f: selected,
        text,
        ...llm.llmParams(),
        ...(sysPrompt.trim() ? { system_prompt: sysPrompt } : {}),
      };
      if (jsonEdited) {
        // JSON panel is open with hand edits: re-use them, skip re-translation.
        setResult(await evaluate());
      } else {
        await translate(params);
        setResult(await evaluate());
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <div>
        <h2 className="text-lg font-bold">Query Tool</h2>
        <p className="mt-0.5 text-xs text-muted">
          Compute per-data-point probabilities that posterior realizations satisfy a plain-English
          geological query, translated by an LLM.
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
                    placeholder="e.g. SDR"
                  />
                </Field>
                <Field label="Select posterior file:">
                  <Select
                    value={selected}
                    onChange={(e) => {
                      setSelected(e.target.value);
                      setModelInfo(null);
                      setModelsError(null);
                      setResult(null);
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
            <Card title="Query">
              <div className="space-y-4">
                <details className="rounded-lg border border-edge bg-panel-2 px-3 py-2">
                  <summary className="cursor-pointer text-xs font-semibold text-muted">
                    System prompt (advanced — rarely changed)
                  </summary>
                  <textarea
                    value={sysPrompt}
                    onChange={(e) => setSysPrompt(e.target.value)}
                    rows={10}
                    spellCheck={false}
                    className="mt-2 w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 font-mono text-xs text-fg outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                  />
                </details>
                <Field label="Enter query in plain English:">
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows={3}
                    placeholder="e.g. What is the probability that cumulative clay thickness exceeds 10 m within 0 to 30 m depth?"
                    className="w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 text-sm text-fg outline-none transition placeholder:text-muted/50 focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                  />
                </Field>
                <Button onClick={runQuery} disabled={busy || !text.trim() || !selected}>
                  <Search size={14} />
                  {busy ? 'Running…' : 'Run query'}
                </Button>
                {error && (
                  <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs whitespace-pre-wrap text-danger">
                    {error}
                  </div>
                )}
                {translation && (
                  <>
                    <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-sm text-info">
                      <span className="font-semibold">Interpretation:</span> {translation.interpretation}
                    </div>
                    <details
                      className="rounded-lg border border-edge bg-panel-2 px-3 py-2"
                      open={jsonOpen}
                      onToggle={(e) => setJsonOpen((e.target as HTMLDetailsElement).open)}
                    >
                      <summary className="cursor-pointer text-xs font-semibold text-muted">
                        Query JSON (optional advanced view — edit and run again to apply)
                      </summary>
                      <div className="mt-2 space-y-2">
                        <textarea
                          value={jsonText}
                          onChange={(e) => {
                            setJsonText(e.target.value);
                            jsonTextRef.current = e.target.value;
                            setJsonEdited(true);
                          }}
                          rows={12}
                          spellCheck={false}
                          className="w-full rounded-lg border border-edge bg-panel-2 px-3 py-2 font-mono text-xs text-fg outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
                        />
                        {jsonError && (
                          <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                            Invalid JSON: {jsonError}
                          </div>
                        )}
                        <Button onClick={runQuery} disabled={busy || !!jsonError}>
                          <Search size={14} />
                          {busy ? 'Running…' : 'Run query'}
                        </Button>
                      </div>
                    </details>
                  </>
                )}
                {result && <QueryResultPanel result={result} />}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
