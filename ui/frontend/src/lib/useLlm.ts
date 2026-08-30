/** Shared LLM provider/key/model state for the Query and Query-Volume views.
 *
 *  Holds provider selection, the API key, and a model list fetched live from the
 *  provider. When the server already has an LLM configured via environment
 *  variables, everything here collapses: `ready` is true and `llmParams()`
 *  returns `{}` so the backend uses its env config. */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import type { LLMConfig } from './types';
import { buildModelString, OTHER_ID, providerDef } from './llm';

export interface LlmParams {
  provider?: string;
  model?: string;
  api_key?: string;
}

export interface UseLlm {
  serverConfig: LLMConfig | null;
  providerId: string;
  setProviderId: (v: string) => void;
  otherProvider: string;
  setOtherProvider: (v: string) => void;
  apiKey: string;
  setApiKey: (v: string) => void;
  modelId: string;
  setModelId: (v: string) => void;
  customModel: string;
  setCustomModel: (v: string) => void;
  useCustomModel: boolean;
  setUseCustomModel: (v: boolean) => void;
  models: string[];
  modelsLive: boolean;
  modelsError: string | null;
  modelsLoading: boolean;
  reloadModels: () => void;
  resolvedProvider: string;
  needsKey: boolean;
  ready: boolean;
  llmParams: () => LlmParams;
}

export function useLlm(): UseLlm {
  const [serverConfig, setServerConfig] = useState<LLMConfig | null>(null);
  const [providerId, setProviderId] = useState('anthropic');
  const [otherProvider, setOtherProvider] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [modelId, setModelId] = useState('');
  const [customModel, setCustomModel] = useState('');
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [modelsLive, setModelsLive] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [reloadTick, setReloadTick] = useState(0);

  const def = providerDef(providerId);
  const isOther = providerId === OTHER_ID;
  const isOllama = providerId === 'ollama';
  const resolvedProvider = isOther ? otherProvider.trim() : def.prefix;
  const needsKey = isOther ? true : def.needsKey;

  useEffect(() => {
    api
      .llmConfig()
      .then(setServerConfig)
      .catch(() => setServerConfig({ configured: false, provider: null, model: null }));
  }, []);

  // Reset the model choice whenever the provider changes.
  useEffect(() => {
    setModelId('');
    setCustomModel('');
    setUseCustomModel(false);
    setModels([]);
    setModelsError(null);
    setModelsLive(false);
  }, [providerId, otherProvider]);

  // Fetch the model list live from the provider. Debounced while the key is typed.
  useEffect(() => {
    if (serverConfig?.configured) return;
    const providerName = isOther ? otherProvider.trim() : providerId;
    if (!providerName) return;
    if (needsKey && !apiKey.trim()) return;

    let cancelled = false;
    const fetchModels = () => {
      setModelsLoading(true);
      api
        .providerModels(providerName, apiKey.trim() || undefined)
        .then((r) => {
          if (cancelled) return;
          setModels(r.models);
          setModelsLive(r.live);
          setModelsError(r.error);
          if (r.live && r.models.length) {
            setUseCustomModel(false);
            setModelId((cur) => (cur && r.models.includes(cur) ? cur : r.models[0]));
          } else {
            setUseCustomModel(true);
          }
        })
        .catch((e) => {
          if (cancelled) return;
          setModels([]);
          setModelsLive(false);
          setModelsError((e as Error).message);
          setUseCustomModel(true);
        })
        .finally(() => {
          if (!cancelled) setModelsLoading(false);
        });
    };

    const delay = needsKey && !isOllama ? 500 : 0;
    const timer = setTimeout(fetchModels, delay);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [serverConfig?.configured, providerId, otherProvider, apiKey, needsKey, isOther, isOllama, reloadTick]);

  const reloadModels = useCallback(() => setReloadTick((n) => n + 1), []);

  const effectiveModelId = useCustomModel ? customModel : modelId;
  const fullModel = buildModelString(resolvedProvider, effectiveModelId);

  const ready = useMemo(() => {
    if (serverConfig?.configured) return true;
    if (!resolvedProvider) return false;
    if (needsKey && !apiKey.trim()) return false;
    return !!fullModel;
  }, [serverConfig?.configured, resolvedProvider, needsKey, apiKey, fullModel]);

  const llmParams = useCallback((): LlmParams => {
    if (serverConfig?.configured) return {};
    return {
      provider: resolvedProvider,
      model: fullModel,
      ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
    };
  }, [serverConfig?.configured, resolvedProvider, fullModel, apiKey]);

  return {
    serverConfig,
    providerId,
    setProviderId,
    otherProvider,
    setOtherProvider,
    apiKey,
    setApiKey,
    modelId,
    setModelId,
    customModel,
    setCustomModel,
    useCustomModel,
    setUseCustomModel,
    models,
    modelsLive,
    modelsError,
    modelsLoading,
    reloadModels,
    resolvedProvider,
    needsKey,
    ready,
    llmParams,
  };
}
