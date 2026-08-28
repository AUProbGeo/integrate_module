/** Query view: natural-language posterior queries translated by an LLM. */

import { useEffect, useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import { api } from '../lib/api';
import type { H5FileInfo, LLMConfig, OllamaModels, PriorModelsResponse, QueryResult } from '../lib/types';
import { Button, Card, Field, Select, TextInput } from '../components/ui';

const OLLAMA_DEFAULT_MODEL = 'ollama_chat/qwen3:latest';

function LLMSection({
  llm,
  provider,
  setProvider,
  apiKey,
  setApiKey,
  ollamaModel,
  setOllamaModel,
  ollamaList,
}: {
  llm: LLMConfig | null;
  provider: 'claude' | 'ollama';
  setProvider: (p: 'claude' | 'ollama') => void;
  apiKey: string;
  setApiKey: (k: string) => void;
  ollamaModel: string;
  setOllamaModel: (m: string) => void;
  ollamaList: OllamaModels | null;
}) {
  const serverHasClaude = !!(llm?.configured && llm.provider === 'claude');
  const serverHasOllama = !!(llm?.configured && llm.provider === 'ollama');
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4">
        <span className="text-[13px] font-medium text-muted">LLM provider:</span>
        {(['claude', 'ollama'] as const).map((p) => (
          <label key={p} className="flex cursor-pointer items-center gap-1.5 text-sm">
            <input
              type="radio"
              name="llm-provider"
              checked={provider === p}
              onChange={() => setProvider(p)}
              className="accent-accent"
            />
            {p === 'claude' ? 'Claude' : 'Ollama'}
          </label>
        ))}
      </div>
      {provider === 'claude' ? (
        serverHasClaude ? (
          <div className="text-xs text-muted">
            Using Claude (<code>{llm?.model}</code>) with the API key configured on the server.
          </div>
        ) : (
          <Field label="Anthropic API key:">
            <TextInput
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-..."
              autoComplete="off"
            />
          </Field>
        )
      ) : ollamaList?.running && ollamaList.models.length ? (
        <Field label="Ollama model (local server):">
          <Select value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)}>
            {!ollamaList.models.some((m) => `ollama_chat/${m}` === ollamaModel) && (
              <option value={ollamaModel}>{ollamaModel}</option>
            )}
            {ollamaList.models.map((m) => (
              <option key={m} value={`ollama_chat/${m}`}>
                {m}
              </option>
            ))}
          </Select>
        </Field>
      ) : (
        <Field
          label="Ollama model:"
          hint={ollamaList ? 'No local Ollama server detected — entering a free-form model id.' : 'Checking for a local Ollama server…'}
        >
          <TextInput
            value={ollamaModel}
            onChange={(e) => setOllamaModel(e.target.value)}
            placeholder="ollama_chat/llama3:latest"
          />
        </Field>
      )}
      {provider === 'ollama' && serverHasOllama && (
        <div className="text-xs text-muted">Using the API key configured on the server.</div>
      )}
      {provider === 'claude' && !serverHasClaude && !apiKey.trim() && (
        <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-xs text-info">
          Enter an API key or select Ollama to continue.
        </div>
      )}
    </div>
  );
}

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
      <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-sm text-info">
        <span className="font-semibold">Interpretation:</span> {result.interpretation}
      </div>
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
      <details className="rounded-lg border border-edge bg-panel-2 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-muted">Query JSON</summary>
        <pre className="mt-2 max-h-72 overflow-auto text-xs text-fg">
          {JSON.stringify(result.query_dict, null, 2)}
        </pre>
      </details>
      <details className="rounded-lg border border-edge bg-panel-2 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-muted">System Prompt</summary>
        <pre className="mt-2 max-h-72 overflow-auto text-xs whitespace-pre-wrap text-fg">
          {result.system_prompt}
        </pre>
      </details>
    </div>
  );
}

export function QueryView({ files }: { files: H5FileInfo[] }) {
  const [llm, setLlm] = useState<LLMConfig | null>(null);
  const [provider, setProvider] = useState<'claude' | 'ollama'>('claude');
  const [apiKey, setApiKey] = useState('');
  const [ollamaModel, setOllamaModel] = useState(OLLAMA_DEFAULT_MODEL);
  const [ollamaList, setOllamaList] = useState<OllamaModels | null>(null);

  const [filterText, setFilterText] = useState('');
  const [selected, setSelected] = useState('');
  const [modelInfo, setModelInfo] = useState<PriorModelsResponse | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);

  useEffect(() => {
    api
      .llmConfig()
      .then((cfg) => {
        setLlm(cfg);
        if (cfg.configured && cfg.provider) setProvider(cfg.provider);
        if (cfg.configured && cfg.provider === 'ollama' && cfg.model) setOllamaModel(cfg.model);
      })
      .catch(() => setLlm({ configured: false, provider: null, model: null }));
  }, []);

  // Probe the local Ollama server the first time Ollama is selected.
  useEffect(() => {
    if (provider !== 'ollama' || ollamaList !== null) return;
    let cancelled = false;
    api
      .ollamaModels()
      .then((r) => {
        if (cancelled) return;
        setOllamaList(r);
        if (r.running && r.models.length) {
          setOllamaModel((cur) =>
            r.models.some((m) => `ollama_chat/${m}` === cur) ? cur : `ollama_chat/${r.models[0]}`,
          );
        }
      })
      .catch(() => !cancelled && setOllamaList({ running: false, models: [] }));
    return () => {
      cancelled = true;
    };
  }, [provider, ollamaList]);

  const llmReady = useMemo(() => {
    if (!llm) return false;
    if (provider === 'claude') {
      return (llm.configured && llm.provider === 'claude') || !!apiKey.trim();
    }
    return !!ollamaModel.trim();
  }, [llm, provider, apiKey, ollamaModel]);

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

  const runQuery = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const params =
        provider === 'claude'
          ? { f: selected, text, provider: 'claude' as const, ...(apiKey.trim() ? { api_key: apiKey } : {}) }
          : { f: selected, text, provider: 'ollama' as const, model: ollamaModel };
      setResult(await api.runQuery(params));
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
        <LLMSection
          llm={llm}
          provider={provider}
          setProvider={setProvider}
          apiKey={apiKey}
          setApiKey={setApiKey}
          ollamaModel={ollamaModel}
          setOllamaModel={setOllamaModel}
          ollamaList={ollamaList}
        />
      </Card>

      {llmReady && (
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
                  {busy ? 'Running — translating with LLM, then evaluating…' : 'Query'}
                </Button>
                {error && (
                  <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs whitespace-pre-wrap text-danger">
                    {error}
                  </div>
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
