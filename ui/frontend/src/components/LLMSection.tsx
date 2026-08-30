/** Provider + API key + model picker, shared by the Query and Query-Volume views.
 *
 *  All state lives in the `useLlm` hook; this component is presentational. When
 *  the server has an LLM configured via env vars it collapses to a one-liner. */

import { Button, Field, Select, TextInput } from './ui';
import { CUSTOM_MODEL, OTHER_ID, PROVIDERS, providerDef } from '../lib/llm';
import type { UseLlm } from '../lib/useLlm';

export function LLMSection({ llm }: { llm: UseLlm }) {
  if (llm.serverConfig?.configured) {
    return (
      <div className="text-xs text-muted">
        Using <code>{llm.serverConfig.model}</code> (configured on the server).
      </div>
    );
  }

  const isOther = llm.providerId === OTHER_ID;
  const isOllama = llm.providerId === 'ollama';
  const def = providerDef(llm.providerId);
  const keyLabel = isOther ? 'API key:' : `${def.label} API key:`;

  return (
    <div className="space-y-3">
      <Field label="LLM provider:">
        <Select value={llm.providerId} onChange={(e) => llm.setProviderId(e.target.value)}>
          {PROVIDERS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
          <option value={OTHER_ID}>Other…</option>
        </Select>
      </Field>

      {isOther && (
        <Field label="Provider id (LiteLLM):" hint="e.g. together_ai, fireworks_ai, vertex_ai">
          <TextInput
            value={llm.otherProvider}
            onChange={(e) => llm.setOtherProvider(e.target.value)}
            placeholder="together_ai"
          />
        </Field>
      )}

      {llm.needsKey && (
        <Field label={keyLabel}>
          <TextInput
            type="password"
            value={llm.apiKey}
            onChange={(e) => llm.setApiKey(e.target.value)}
            placeholder="API key — kept in browser memory, sent per request"
            autoComplete="off"
          />
        </Field>
      )}

      <ModelPicker llm={llm} isOllama={isOllama} />

      {!llm.ready && (
        <div className="rounded-lg border border-info/30 bg-info/10 px-3 py-2 text-xs text-info">
          {llm.needsKey && !llm.apiKey.trim()
            ? 'Enter an API key, then pick a model, to continue.'
            : 'Pick a model to continue.'}
        </div>
      )}
    </div>
  );
}

function ModelPicker({ llm, isOllama }: { llm: UseLlm; isOllama: boolean }) {
  if (llm.needsKey && !llm.apiKey.trim()) {
    return (
      <Field label="Model:" hint="Enter your API key to load the model list.">
        <Select value="" disabled>
          <option value="">—</option>
        </Select>
      </Field>
    );
  }

  if (llm.modelsLoading) {
    return (
      <Field label="Model:" hint="Loading models…">
        <Select value="" disabled>
          <option value="">—</option>
        </Select>
      </Field>
    );
  }

  if (!llm.useCustomModel && llm.models.length > 0) {
    return (
      <Field label="Model:">
        <Select
          value={llm.modelId}
          onChange={(e) => {
            if (e.target.value === CUSTOM_MODEL) llm.setUseCustomModel(true);
            else llm.setModelId(e.target.value);
          }}
        >
          {llm.models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
          <option value={CUSTOM_MODEL}>Custom…</option>
        </Select>
      </Field>
    );
  }

  const hint = llm.modelsError
    ? `Couldn't load models live (${llm.modelsError}). Enter the model id manually.`
    : isOllama
      ? 'No reachable Ollama server — enter a model id (e.g. qwen3:latest).'
      : 'Enter the model id manually.';

  return (
    <div className="space-y-1.5">
      <Field label="Model id:" hint={hint}>
        <TextInput
          value={llm.customModel}
          onChange={(e) => llm.setCustomModel(e.target.value)}
          placeholder={isOllama ? 'qwen3:latest' : 'gpt-4o'}
        />
      </Field>
      <div className="flex gap-2">
        <Button variant="ghost" onClick={llm.reloadModels}>
          Reload list
        </Button>
        {llm.models.length > 0 && (
          <Button variant="ghost" onClick={() => llm.setUseCustomModel(false)}>
            Back to list
          </Button>
        )}
      </div>
    </div>
  );
}
