/** LLM provider catalogue + model-string helpers (no React). */

export interface ProviderDef {
  id: string;
  label: string;
  prefix: string;      // LiteLLM route prefix
  needsKey: boolean;
}

export const PROVIDERS: ProviderDef[] = [
  { id: 'openai', label: 'OpenAI', prefix: 'openai', needsKey: true },
  { id: 'anthropic', label: 'Anthropic (Claude)', prefix: 'anthropic', needsKey: true },
  { id: 'gemini', label: 'Google Gemini', prefix: 'gemini', needsKey: true },
  { id: 'groq', label: 'Groq', prefix: 'groq', needsKey: true },
  { id: 'mistral', label: 'Mistral', prefix: 'mistral', needsKey: true },
  { id: 'deepseek', label: 'DeepSeek', prefix: 'deepseek', needsKey: true },
  { id: 'xai', label: 'xAI (Grok)', prefix: 'xai', needsKey: true },
  { id: 'openrouter', label: 'OpenRouter', prefix: 'openrouter', needsKey: true },
  { id: 'ollama', label: 'Ollama (local / remote)', prefix: 'ollama_chat', needsKey: false },
];

export const OTHER_ID = 'other';

/** Sentinel option value that switches the model <Select> to a free-text field. */
export const CUSTOM_MODEL = '__custom__';

export function providerDef(id: string): ProviderDef {
  return (
    PROVIDERS.find((p) => p.id === id) ?? {
      id,
      label: id || 'Custom provider',
      prefix: id,
      needsKey: true,
    }
  );
}

/** Combine a provider prefix and a (possibly already-prefixed) model id into a
 *  full LiteLLM model string, e.g. ("openai", "gpt-4o") -> "openai/gpt-4o". */
export function buildModelString(prefix: string, modelId: string): string {
  const m = modelId.trim();
  if (!m || !prefix) return '';
  if (m.startsWith(`${prefix}/`)) return m;
  return `${prefix}/${m}`;
}
