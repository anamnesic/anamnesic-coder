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
 - [`src/tools/index.md`](src/tools/index.md) - Documentation for tooling helpers
 - [`src/types/index.md`](src/types/index.md) - Documentation for shared types

Each index.md file contains a list of files in that directory with brief descriptions of their purpose.

## Architectural Decision Records (ADRs)

- [`docs/adr/0001-llm-router.md`](docs/adr/0001-llm-router.md) — Route LLM traffic through a runtime LlmRouter
- [`docs/adr/0002-harness-parity-glm52.md`](docs/adr/0002-harness-parity-glm52.md) — Harness Parity (GLM-5.2)
- [`docs/adr/0003-resilient-routing.md`](docs/adr/0003-resilient-routing.md) — Resilient Routing: Same-Tier Fallback, Backoff & Error Surfacing
- [`docs/adr/0004-project-context-and-workspace-structure.md`](docs/adr/0004-project-context-and-workspace-structure.md) — Project Context Auto-Loading & Workspace Directory Discovery
- [`docs/adr/0005-token-usage-tracking.md`](docs/adr/0005-token-usage-tracking.md) — Token Usage & Cost Tracking Per Turn
- [`docs/adr/0006-line-range-code-editing.md`](docs/adr/0006-line-range-code-editing.md) — Line-Range Code Editing (`edit_file`)
- [`docs/adr/0007-gguf-safety-and-bounds-checking.md`](docs/adr/0007-gguf-safety-and-bounds-checking.md) — GGUF & Dequantization Memory Safety
- [`docs/gap-analysis-2026-08.md`](docs/gap-analysis-2026-08.md) — 2026 Competitor Gap Analysis Report