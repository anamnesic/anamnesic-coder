# TODO — Harness Gaps vs Claude Code / Gemini Codex (August 2026)

## Context

This document lists gaps between the `anamnesic-coder` harness and the tool-calling
patterns used by Claude Code and Gemini Codex (as of August 2026), organized by
priority and mapped to the six target models:

| Model | Provider | Difficulty |
|---|---|---|
| nvidia/nvidia-nemotron-nano-9b-v2 | NIM | easy |
| deepseek-ai/deepseek-v4-flash | DeepSeek API | medium |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | NIM | medium |
| minimaxai/minimax-m3 | Minimax API | hard |
| z-ai/glm-5.2 | Zhipu API | hard |
| deepseek-ai/deepseek-v4-pro | DeepSeek API | hard |

---

## 1. Cloud Model Benchmarking

The bench harness (`src/bench/model_bench.rs`) only benchmarks local GGUF models
via `Model::load()` + `InferenceEngine`. Cloud models (NIM, DeepSeek, Minimax,
Zhipu) cannot be evaluated.

- [ ] Add `CloudBenchResult` struct that wraps `provider_chain::FallbackChain` for
      cloud model benchmarking
- [ ] Add `benchmark_cloud_model()` function that runs a prompt against a cloud
      provider and measures TPS, latency, cost (from models.dev catalog), and
      output quality
- [ ] Add `CloudBenchCategory` enum (general, coding, reasoning, tool_use) to
      mirror the local bench categories
- [ ] Wire cloud benchmarks into the `Bench` CLI command so `cargo run -- bench
      --category coding --cloud` works
- [ ] Add per-model cost estimation using `models_dev::CloudMatch` pricing data
- [ ] Add context window utilization tracking (input tokens + output tokens vs
      model's context limit)

---

## 2. Tool Calling — Core Support

Neither Claude Code nor Gemini Codex ship as text-only generators. Both use
structured tool calls (function calling / tool use) as the primary agent
mechanism. The current harness has no tool calling at all.

- [ ] Define a `ToolDef` struct (name, description, input_schema: JSON Schema)
      in a new `src/agent/tools.rs` module
- [ ] Add `tool_use` field to `LlmClient::generate()` / `LlmClient::chat()` that
      accepts a `Vec<ToolDef>` and passes it as `tools` in the OpenAI-compatible
      request body for cloud providers
- [ ] Add `tool_use` support for the Ollama `/api/chat` endpoint (Ollama supports
      tool calls via the `tools` field in the chat request)
- [ ] Add `tool_use` support for NIM `/v1/chat/completions` (already OpenAI-compatible)
- [ ] Parse tool call responses from cloud providers (OpenAI format:
      `response.choices[0].message.tool_calls[]`; Anthropic format:
      `content[].tool_use`)
- [ ] Parse tool call responses from Ollama (returns `tool_calls` in message)
- [ ] Implement tool result formatting: convert execution results back to the
      provider-specific format (`tool` role for OpenAI, `tool_result` block for
      Anthropic, `function_call` response for Ollama)

---

## 3. Parallel Tool Calls

Claude Code and Gemini Codex both support emitting multiple tool calls in a
single response and executing them concurrently.

- [ ] Detect multiple tool calls in a single model response
- [ ] Execute independent tool calls concurrently using `tokio::join!` or
      `futures::future::join_all`
- [ ] Handle partial failures (some tools succeed, some fail) — return all
      results to the model including error details
- [ ] Add `parallel_tool_calls` config option (default: `true`) to `Config`
- [ ] Add dependency analysis: if tool B's input depends on tool A's output,
      execute sequentially within the same turn
- [ ] Add timeout per tool call (default 30s) to prevent hanging on slow tools

---

## 4. Agent Loop with Tool Use

Claude Code's core loop is: `LLM → tool_calls → execute → results → LLM → ...`
until `stop_reason === "end_turn"`. The current agent loop has no tool use cycle.

- [ ] Refactor `run_agent_loop()` to support a tool-use iteration:
      1. Send prompt + tools to model
      2. If model returns tool calls, execute them and feed results back
      3. Repeat until model produces final text (no more tool calls)
      4. Apply plan/act/verify on the final result
- [ ] Add `max_tool_iterations` config (default 10) to prevent infinite loops
- [ ] Add `tool_choice` support: `auto` (default), `none` (disable tools),
      `required` (force at least one tool call), and specific tool name
- [ ] Add tool call logging with timing (LLM inference + tool execution) for
      per-turn latency breakdown

---

## 5. MCP (Model Context Protocol) Integration

Claude Code uses MCP as its primary extension mechanism for dynamic tool
discovery. Gemini Codex also supports MCP (remote MCP servers).

- [ ] Add MCP client module (`src/agent/mcp.rs`) that connects to MCP servers
      via stdio or SSE transport
- [ ] Implement tool discovery: `tools/list` request to MCP server, merge
      discovered tools into the agent's available tool set
- [ ] Implement tool execution via MCP: `tools/call` request with tool name +
      arguments
- [ ] Add MCP server configuration to `Config` (list of MCP server commands)
- [ ] Add `supports_parallel_tool_calls` flag per MCP server (matching Codex
      CLI's per-server parallelism config)
- [ ] Cache MCP tool definitions and refresh on session restart (cache TTL)

---

## 6. Structured Output / JSON Mode

Claude Code and Gemini Codex both support forcing JSON-structured responses,
which is critical for reliable tool call parameter generation.

- [ ] Add `response_format` parameter to `LlmClient::generate()` and `.chat()`
      supporting: `text` (default), `json_object`, `json_schema`
- [ ] For OpenAI-compatible providers (NIM, DeepSeek), pass
      `response_format: { type: "json_schema", json_schema: { ... } }`
- [ ] For Ollama, pass `format: { type: "json" }` in the chat request
- [ ] Add `json_schema` validation of model responses before feeding to tool
      executor
- [ ] Add fallback: if JSON parsing fails, retry with explicit instruction to
      return valid JSON

---

## 7. Provider Infrastructure Gaps

Several providers needed for the target models are missing from the current
infrastructure.

- [ ] Add `minimax` to `default_base()` in `src/providers/verify.rs`
      (endpoint: `https://api.minimax.chat/v1`)
- [ ] Add `z-ai` to `default_base()` in `src/providers/verify.rs`
      (endpoint: `https://api.z.ai/v1`)
- [ ] Add `nvidia` NIM provider to `provider_chain.rs` with configurable base_url
      (currently hardcoded to `https://integrate.api.nvidia.com`)
- [ ] Add `DeepSeekProvider` struct implementing `CompletionProvider` with
      DeepSeek API base URL and key handling
- [ ] Add `MinimaxProvider` struct implementing `CompletionProvider` with
      Minimax API base URL and key handling
- [ ] Add `ZhipuProvider` struct implementing `CompletionProvider` with
      Zhipu API base URL and key handling
- [ ] Update `build_default_chain()` to accept a list of providers from config
      instead of hardcoding NIM + Ollama
- [ ] Add provider capability tagging (tool_call, reasoning, coding, vision) so
      the harness can select appropriate models per task type

---

## 8. Evaluation Metrics (Beyond TPS)

The current bench only measures tokens/second. Claude Code and Gemini Codex
harnesses measure code quality, correctness, and tool-use accuracy.

- [ ] Add code quality score: run `cargo check` / `cargo test` on generated code
      and record pass/fail
- [ ] Add tool-use accuracy metric: compare model's tool calls against expected
      calls for a given task (precision, recall, F1)
- [ ] Add multi-turn task evaluation: run a task that requires 3+ tool calls and
      measure success rate
- [ ] Add latency percentiles (p50, p95, p99) for inference + tool execution
- [ ] Add cost-per-task metric using models.dev pricing data
- [ ] Add output correctness check: compare generated code output against
      expected output for a given test case
- [ ] Add context efficiency metric: output tokens / input tokens ratio

---

## 9. Agent Loop Integration with Provider Chain

The `FallbackChain` in `provider_chain.rs` is defined but never used by the
agent loop (`src/agent/loop.rs`). The agent loop uses `LlmClient` directly.

- [ ] Integrate `FallbackChain` into the agent loop so that if one cloud provider
      fails (rate limited, credit exhausted, transient error), the loop
      automatically falls back to the next provider
- [ ] Add per-provider rate limiting using `TokenBucket` with model-specific RPM
      limits (NIM models have different rate limits than DeepSeek/Minimax/Zhipu)
- [ ] Add provider health tracking: if a provider fails N times in a row,
      temporarily deprioritize it in the chain
- [ ] Add cost-aware routing: prefer cheaper providers when they can handle the
      task, fall back to more capable (expensive) models for hard tasks

---

## 10. Missing Prompt Templates

The `prompts/` directory has four empty template files (0 bytes):
`coder.txt`, `fixer.txt`, `planner.txt`, `system.txt`.

- [ ] Populate `prompts/system.txt` with a system prompt that includes tool-use
      instructions and MCP awareness
- [ ] Populate `prompts/planner.txt` with a planner prompt that can generate
      tool-use plans (not just file operations)
- [ ] Populate `prompts/coder.txt` with a coder prompt that includes tool-use
      for code generation and verification
- [ ] Populate `prompts/fixer.txt` with a fixer prompt for the verify/fix loop
      that can use tools to diagnose and fix failures

---

## 11. Configuration Gaps

- [ ] Add `tool_use` boolean field to `Config` (default: `true`)
- [ ] Add `max_tool_iterations` field to `Config` (default: 10)
- [ ] Add `parallel_tool_calls` boolean field to `Config` (default: `true`)
- [ ] Add `tool_timeout_secs` field to `Config` (default: 30)
- [ ] Add `mcp_servers` field to `Config` (list of MCP server configs)
- [ ] Add `cloud_providers` field to `Config` (list of cloud providers to
      benchmark/test, with model, RPM, and API key references)
- [ ] Add per-model rate limit config (RPM) to `Config`
- [ ] Add `evaluation_metrics` boolean field to `Config` to enable/disable
      quality scoring during benchmarks

---

## 12. Testing & Validation

- [ ] Write integration test that runs a multi-turn task with tool calls against
      a mock provider
- [ ] Write test for parallel tool call execution with mixed success/failure
- [ ] Write test for provider fallback chain with simulated rate-limit errors
- [ ] Write test for MCP tool discovery and execution
- [ ] Add benchmark for the six target models (requires API keys):
      nemotron-nano-9b-v2, deepseek-v4-flash, nemotron-super-49b-v1.5,
      minimax-m3, glm-5.2, deepseek-v4-pro
- [ ] Add regression test that ensures code quality score doesn't degrade
      after harness changes