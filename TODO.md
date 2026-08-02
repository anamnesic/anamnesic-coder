# TODO — Remaining Harness Gaps (August 2026)

## Current baseline

The harness now has an OpenAI-compatible NIM client, a primary `LLM → tool →
result → LLM` loop, file read/write/search tools, an allowlisted command tool,
and a planner fallback. The typed tool loop is optimized for and smoke-tested with
`z-ai/glm-5.2`, including an atomic edit followed by a passing Cargo gate.

## P0 — Safety and reliability

- [x] Add a timeout and process-group kill for every `run_command` invocation.
- [x] Replace the prefix-based command allowlist with parsed executable + args;
      reject shell operators, redirects, substitutions and chained commands by default.
- [x] Add a per-tool approval policy: read/search automatic; write and commands
      configurable as ask/allow/deny.
- [x] Add a workspace diff summary and explicit commit/rollback workflow before
      ending an edit task.
- [x] Preserve tool-call IDs and handle malformed/provider-specific arguments
      without silently treating them as a final response.
- [x] Make maximum tool iterations, output caps and timeouts configurable.
- [x] Add tests for path traversal, symlink escape and command-injection bypasses.

- [x] Snapshot the workspace per turn and roll back failed/interrupted turns to
      the turn baseline (preserving pre-existing local modifications).
- [x] Route every mutation (typed loop, planner, TUI editor) through the same
      approval + transaction path; a denied action can never be reported as done.

## P1 — Agent quality

- [x] Execute independent tool calls concurrently and return ordered results;
      keep dependent calls sequential.
- [x] Add `tool_choice` (`auto`, `none`, `required`, named tool) and tool
      capability filtering per model.
- [x] Feed tool output, changed-file summaries and test failures into a bounded
      repair loop rather than returning immediately after a tool-use turn.
- [x] Add patch/edit tools (unified diff or targeted replacement) to avoid
      rewriting whole files for small changes.
- [x] Add repository discovery tools: list tree, git diff/status and diagnostics.
- [x] Move prompts into the existing `prompts/` files and version/test them.
- [ ] Add structured planner output with a correctly serialized JSON-schema
      response format; retain Markdown fallback only as a recovery path.

## P1 — Provider and routing

- [x] Unify orchestration: the orphan `FallbackChain` agent path was removed in
      favor of the single typed loop, whose router fallback preserves messages,
      tools, tool-call ids and finish reason.
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
