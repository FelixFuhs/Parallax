# OpenRouter Usage Notes

Verified against the OpenRouter docs and model pages on 2026-03-20.

## Request shape used in this repo

- Endpoint: `POST https://openrouter.ai/api/v1/chat/completions`
- Auth: `Authorization: Bearer $OPENROUTER_API_KEY`
- Local convenience: `openrouter.py` will also load `.env` and `.env.local` from the repo root before checking environment variables
- Models:
  - `openai/gpt-5.4-nano:online`
  - `openai/gpt-5.4:online`
- Request body fields that matter here:
  - `messages`
  - `reasoning: {"effort": "high" | "xhigh"}`
  - `response_format: {"type": "json_object"}`
  - `provider: {"require_parameters": true}` so OpenRouter only routes to providers that support the requested parameters
  - `max_tokens` set high enough to leave room for the final JSON after reasoning
  - `plugins: [{"id": "response-healing"}]` on non-streaming calls to repair malformed JSON when possible

## Important OpenRouter details

- `usage` is returned automatically in responses. Older `usage: {"include": true}` and `stream_options: {"include_usage": true}` flags are deprecated.
- Reasoning tokens are billed as output tokens. The runner estimates cost from prompt tokens plus `completion_tokens + reasoning_tokens`.
- `:online` is a dynamic OpenRouter variant that attaches web results to the prompt. Model pages also show an extra web-search price component, so `_meta.cost_usd` in this repo is an estimate based on the token pricing requested for the project, not a full billing reconciliation.
- For OpenAI reasoning models, `reasoning.effort` uses a fraction of `max_tokens` as the reasoning budget. `max_tokens` must be higher than that budget or the final answer can be truncated.
- `HTTP-Referer` and `X-Title` headers are optional. If you want app attribution on OpenRouter, set them via `OPENROUTER_REFERER` and `OPENROUTER_TITLE`.
- OpenRouter also supports stricter `response_format: {"type": "json_schema"}`. This repo keeps `json_object` because that is the project requirement, but `json_schema` is the stronger option if you later want hard schema enforcement.

## Why `require_parameters` is set

OpenRouter can route the same model through multiple providers. Without `provider.require_parameters: true`, providers that do not support `response_format` can still receive the request and ignore unsupported fields. This repo enables `require_parameters` so JSON mode and other requested parameters are enforced by routing, not left to chance.

## Sources

- [OpenRouter API reference](https://openrouter.ai/docs/api-reference/overview/)
- [Provider routing](https://openrouter.ai/docs/features/provider-routing/)
- [Structured outputs](https://openrouter.ai/docs/features/structured-outputs)
- [Response healing plugin](https://openrouter.ai/docs/guides/features/plugins/response-healing)
- [Usage accounting](https://openrouter.ai/docs/guides/administration/usage-accounting)
- [Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [GPT-5.4 model page](https://openrouter.ai/openai/gpt-5.4)
- [GPT-5.4 Nano model page](https://openrouter.ai/openai/gpt-5.4-nano)
