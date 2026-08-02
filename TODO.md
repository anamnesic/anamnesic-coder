# TODO — Remaining Harness Gaps (August 2026)

## Current baseline

The harness now has an OpenAI-compatible NIM client, a primary `LLM → tool →
result → LLM` loop, file read/write/search tools, an allowlisted command tool,
and a planner fallback. The NIM loop was validated with
`nvidia/llama-3.3-nemotron-super-49b-v1.5` executing `cargo check`.

## P0 — Safety and reliability

- [ ] Add a timeout and process-group kill for every `run_command` invocation.
- [ ] Replace the prefix-based command allowlist with parsed executable + args;
      reject shell operators, redirects, substitutions and chained commands by default.
- [ ] Add a per-tool approval policy: read/search automatic; write and commands
      configurable as ask/allow/deny.
- [ ] Add a workspace diff summary and explicit commit/rollback workflow before
      ending an edit task.
- [ ] Preserve tool-call IDs and handle malformed/provider-specific arguments
      without silently treating them as a final response.
- [ ] Make maximum tool iterations, output caps and timeouts configurable.
- [ ] Add tests for path traversal, symlink escape and command-injection bypasses.

## P1 — Agent quality

- [ ] Execute independent tool calls concurrently and return ordered results;
      keep dependent calls sequential.
- [ ] Add `tool_choice` (`auto`, `none`, `required`, named tool) and tool
      capability filtering per model.
- [ ] Feed tool output, changed-file summaries and test failures into a bounded
      repair loop rather than returning immediately after a tool-use turn.
- [ ] Add patch/edit tools (unified diff or targeted replacement) to avoid
      rewriting whole files for small changes.
- [ ] Add repository discovery tools: list tree, git diff/status and diagnostics.
- [ ] Move prompts into the existing `prompts/` files and version/test them.
- [ ] Add structured planner output with a correctly serialized JSON-schema
      response format; retain Markdown fallback only as a recovery path.

## P1 — Provider and routing

- [ ] Use `FallbackChain` in the primary tool loop, not only the unused
      alternate loop.
- [ ] Make provider/model/base URL/RPM configuration data-driven; remove the
      hard-coded cloud benchmark model list.
- [ ] Implement provider health checks, retry classification, `Retry-After`
      handling and circuit breaking.
- [ ] Record NIM reasoning tokens, finish reason, token usage and request IDs.
- [ ] Add capability tests for tool calls, JSON mode and streaming per model.
- [ ] Complete native adapters/default endpoints for DeepSeek, Minimax and Z.ai.

## P2 — Benchmark harness

- [ ] Split local and cloud benchmark result files; do not overwrite local JSON
      when `bench --cloud` runs.
- [ ] Measure TTFT, end-to-end latency, output tokens from API `usage`, TPS,
      p50/p95/p99, errors and retry rate. Do not estimate tokens by words.
- [ ] Add fixed coding, reasoning, tool-use and multi-turn benchmark suites.
- [ ] Score task success with isolated test fixtures, not generation speed alone.
- [ ] Track cost per task and context-window utilization from provider metadata.
- [ ] Persist raw, redacted request/response metadata for reproducible runs.
- [ ] Benchmark all target NIM/provider models with repeat counts and a warm-up.

## P2 — Ecosystem and validation

- [ ] Add an MCP client (stdio and HTTP/SSE), discovery, execution and server
      lifecycle/configuration.
- [ ] Add integration tests using a mock OpenAI-compatible provider for:
      multi-turn tool use, malformed calls, timeouts, fallback and partial failure.
- [ ] Add an end-to-end sandbox fixture that verifies an edit, test and repair
      sequence without touching the host workspace.
- [ ] Add regression snapshots for prompts, plans and tool transcripts.
- [ ] Add CI for formatting, clippy, unit/integration tests and benchmark schema
      compatibility.
