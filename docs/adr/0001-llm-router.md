# ADR 0001: Route LLM traffic through a runtime LlmRouter

**Status:** Accepted  
**Date:** 2026-08-01

## Context

The TUI and agent loop were started with a single fixed `LlmClient` built once
in `main`. Selecting a model with `/model` only changed `Config.coder_model`;
the backend itself never changed.

That caused a real bug: after picking a cloud model (e.g. an NVIDIA NIM model
like `nvidia/llama-3.3-nemotron-super-49b-v1.5`), the request was still sent to
the local Ollama server, which has no such model. The same problem applied to
plain-name cloud providers such as Ollama Cloud (`nemotron-3-nano:30b`), which
we now want as the default cloud provider using `OLLAMA_API_KEY`.

## Decision

Introduce `LlmRouter` (`src/llm/router.rs`) that holds:

- a **local** backend (`LlmClient::Ollama` or local GGUF engine), and
- a lazily-built **cloud** backend (`LlmClient::Cloud`, OpenAI-compatible),
  created from the models.dev catalog base URL + the configured/environment API
  key via `ProviderStore::resolve_cloud_credentials`.

Routing per model id:

1. provider-qualified ids (`nvidia/…`, `ollama-cloud/…`) → cloud;
2. ids explicitly marked cloud (picked from `/model`'s cloud list, or the model
   set by `--cloud`) → cloud;
3. everything else → local.

`/provider` now rebuilds the cloud backend live; the default provider is
`ollama-cloud`, and `OLLAMA_API_KEY` lives in the git-ignored `.env`. The agent
loop, repl and TUI all use the router, resolving the concrete client for the
planner/coder/summarizer model per call.

## Consequences

- Selecting a cloud model in the TUI now actually reaches the cloud backend.
- Switching providers (`/provider`) works without restarting.
- Local-first behavior is preserved: plain local models still go to Ollama.
- Added unit tests for router routing, provider store, tools, compressor,
  memory, models.dev queries and UI helpers (111 total, all passing).
