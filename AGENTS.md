# AGENTS.md

This file lists all the index.md files created in the src directory and its subdirectories, providing a map of where documentation for each module can be found.

## Documentation Index

- [`src/index.md`](src/index.md) - Overview of the src directory
- [`src/agent/index.md`](src/agent/index.md) - Documentation for the agent module
- [`src/compressor/index.md`](src/compressor/index.md) - Documentation for the compressor module
- [`src/config/index.md`](src/config/index.md) - Documentation for the config module
- [`src/hw_recommend/index.md`](src/hw_recommend/index.md) - Documentation for the hardware recommendation module
- [`src/llm/index.md`](src/llm/index.md) - Documentation for the LLM module
- [`src/llm/infer/index.md`](src/llm/infer/index.md) - Documentation for the LLM inference submodule
 - [`src/bench/index.md`](src/bench/index.md) - Documentation for benchmarking utilities
 - [`src/memory/index.md`](src/memory/index.md) - Documentation for memory subsystems
- [`src/repo/index.md`](src/repo/index.md) - Documentation for repository helpers
- [`src/skills/index.md`](src/skills/index.md) - Documentation for the skills system
- [`src/tools/index.md`](src/tools/index.md) - Documentation for tooling helpers
- [`src/types/index.md`](src/types/index.md) - Documentation for shared types
- [`src/terminal/index.md`](src/terminal/index.md) - Documentation for the interactive PTY terminal module

Each index.md file contains a list of files in that directory with brief descriptions of their purpose.

## Architectural Decision Records (ADRs)

- [`docs/adr/0001-llm-router.md`](docs/adr/0001-llm-router.md) — Route LLM traffic through a runtime LlmRouter
- [`docs/adr/0002-harness-parity-glm52.md`](docs/adr/0002-harness-parity-glm52.md) — Harness Parity (GLM-5.2)
- [`docs/adr/0003-resilient-routing.md`](docs/adr/0003-resilient-routing.md) — Resilient Routing: Same-Tier Fallback, Backoff & Error Surfacing
- [`docs/adr/0004-project-context-and-workspace-structure.md`](docs/adr/0004-project-context-and-workspace-structure.md) — Project Context Auto-Loading & Workspace Directory Discovery
- [`docs/adr/0005-token-usage-tracking.md`](docs/adr/0005-token-usage-tracking.md) — Token Usage & Cost Tracking Per Turn
- [`docs/adr/0006-line-range-code-editing.md`](docs/adr/0006-line-range-code-editing.md) — Line-Range Code Editing (`edit_file`)
- [`docs/adr/0007-gguf-safety-and-bounds-checking.md`](docs/adr/0007-gguf-safety-and-bounds-checking.md) — GGUF & Dequantization Memory Safety
- [`docs/adr/0008-robustness-and-floating-point-safety.md`](docs/adr/0008-robustness-and-floating-point-safety.md) — Robustness, Floating Point Safety & Store Key Masking
- [`docs/adr/0009-context-intelligence-and-repo-map.md`](docs/adr/0009-context-intelligence-and-repo-map.md) — Context Intelligence, Calibrated Token Estimation & Repo Map
- [`docs/adr/0010-benchmark-module-refactoring.md`](docs/adr/0010-benchmark-module-refactoring.md) — Benchmark Module Refactoring (Local / Cloud Split)
- [`docs/adr/0011-sub-agent-task-tool.md`](docs/adr/0011-sub-agent-task-tool.md) — Sub-Agent Task Tool (`task`)
- [`docs/adr/0012-mcp-client.md`](docs/adr/0012-mcp-client.md) — MCP Client (Model Context Protocol, stdio)
- [`docs/adr/0013-streaming-tool-call-deltas.md`](docs/adr/0013-streaming-tool-call-deltas.md) — Streaming Tool Call Deltas
- [`docs/adr/0014-circuit-breaker.md`](docs/adr/0014-circuit-breaker.md) — Provider Health Checks & Circuit Breaking
- [`docs/adr/0015-competitive-backlog.md`](docs/adr/0015-competitive-backlog.md) — Competitive Backlog (C1–C9, R1–R3)
- [`docs/gap-analysis-2026-08.md`](docs/gap-analysis-2026-08.md) — 2026 Competitor Gap Analysis Report