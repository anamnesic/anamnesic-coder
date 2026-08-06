# anamnesic-coder

Local coding agent — plan → act → verify. Fusion of TinyCoder + llm-on-legacy-gpus.

```
cargo run -- --dir /tmp/project "add user authentication"
```

## Features

### Agent harness (GLM-5.2)

The primary cloud harness is tuned and smoke-tested with `z-ai/glm-5.2` and uses a typed **Observe → Act → Verify → Repair** loop:

- Native OpenAI/Ollama tool calls; `arguments` may be a JSON string or object.
- Bounded repository discovery with `list_tree`, ranged `read_file`, and `search_code`.
- Atomic `replace_exact` edits that reject stale or ambiguous matches; full `write_file` remains available for deliberate replacements.
- Structured `run_tests` for Cargo, pytest, and npm, using exit status and process timeout rather than output substring matching.
- Every mutation invalidates the previous gate. Failed gates enter a bounded repair loop and can never produce a successful `Done` event.
- `git_status`/`git_diff`, UTF-8-safe output caps, per-turn state reset, changed-file and verification events.

Parity features:

- **Workspace transactions.** Each turn snapshots the workspace first. A failed turn is rolled back to the pre-turn state, so pre-existing local modifications are preserved (the baseline is the turn start, not `git HEAD`).
- **Deterministic finalization.** The real filesystem delta — not the model's narration — decides whether a mutation happened. Successful turns report a diff summary; failures report the rollback.
- **Interactive approval.** With `ask`, the TUI shows an approval modal (`a` allow once, `s` allow for session, `d`/`Esc` deny) while rendering and input keep running. Non-interactive sessions fail closed.
- **Parallel read-only tools.** Independent reads run concurrently with bounded fan-out; mutations and commands stay sequential, and results always return in the model's original call order.
- **`tool_choice` + capability filtering.** `auto`/`none`/`required`/named tool are sent to the provider, tools are withheld from models the catalog reports as non tool-calling, and a pending gate forces `required`.
- **Single orchestration path.** The typed loop, planner steps, and the TUI editor all go through the same policy, transaction, and verification guarantees.

Relevant environment controls:

| Variable | Default | Rationale (GLM-5.2 via NIM) |
|---|---:|---|
| `MAX_TOOL_ITERATIONS` | `128` | GLM-5.2 is fast enough for many turns; 128 matches its 128K output budget (one response per iteration) |
| `MAX_RETRIES` | `5` | Model has strong self-correction; 5 attempts balance cost vs. persistence |
| `MAX_TOOL_OUTPUT_BYTES` | `100000` | ~25K tokens at GLM-5.2's tokenizer ratio; fills ≤10% of usable context per call |
| `MAX_PARALLEL_TOOLS` | `4` | Bounded fan-out for read-only tools; prevents thundering herd on NIM |
| `COMMAND_TIMEOUT_SECS` | `600` | 10 min; GLM-5.2 sessions are long-horizon — tests and builds need room |
| `MAX_CONTEXT_TOKENS` | `128000` | NIM-hosted GLM-5.2 effective context; compaction triggers at 80% (~102K) |
| `TRANSACTION_MAX_BYTES` | `67108864` | 64 MB workspace snapshot; sufficient for most real repositories |
| `ROLLBACK_ON_FAILURE` | `true` | Revert workspace when a turn fails or is interrupted |
| `REQUIRE_DIFF_SUMMARY` | `true` | Deterministic workspace diff appended to every completion |
| `WRITE_TOOL_POLICY` | `allow` | `allow`, `ask`, or `deny` file mutations |
| `COMMAND_TOOL_POLICY` | `allow` | `allow`, `ask`, or `deny` commands/tests |

Model output: **16.384 tokens per turn** (NIM provider cap; native GLM-5.2 supports up to 131K). Override per provider if needed.

### 🪨 Caveman Mode

Compress agent output tokens by 40-75%. Agent says _less_, code quality stays same.

| Level | Effect |
|-------|--------|
| `off` | Normal responses |
| `lite` | Drop filler words, pleasantries, hedging |
| `full` | Drop articles, fragments OK, short synonyms |
| `ultra` | Abbreviated prose, → for causality, fragments only |

```
cargo run -- --caveman full "refactor this callback to async/await"
# [CAVEMAN:FULL]
# Callback → async. Wrap in Promise. Remove nested .then().
```

Toggle at runtime in REPL: `/caveman [lite|full|ultra|off|stats]`

System prompts gain caveman suffix, so planner + coder both respond terse.

### 🔧 NTK Output Compression

Semantic compression for tool outputs (inspired by [VALRAW-ALL/ntk](https://github.com/VALRAW-ALL/ntk)).

**L1 — Fast Filter** (`src/compressor/layer1.rs`):

| Stage | Effect |
|-------|--------|
| ANSI strip | Remove escape codes |
| Progress bars | Drop `Compiling`/`Checking`/`Downloading` lines |
| Collapse blanks | Consecutive blank lines → one |
| Template dedup | Normalize timestamps/UUIDs/hex → `<TS>`/`<UUID>`/`<HEX>`, dedup identical templates: `[×N] ...` |
| Stack collapse | ≥3 framework frames → `... N framework frames omitted` |
| Test filter | Drop `test ... ok` lines, keep FAILED |
| Prefix factor | Extract common prefix when ≥80% lines share ≥12 chars |
| Path shrink | Long paths → last 3 segments |
| Token norm | JWT/SHA256 → `<JWT>`/`<SHA256>` |

**L2 — Tokenizer-Aware** (`src/compressor/layer2.rs`):

| Stage | Effect |
|-------|--------|
| Opaque norm | Replace hashes/base64/URLs with placeholders |
| Path shrink | Keep last 3 path segments |
| WS collapse | Normalize indent, collapse mid-line runs |
| Prefix consolidate | Group lines sharing ≥8-char prefix (ratio ≥50%) |

Applied automatically to `run_command`, `run_tests`, `read_file`, `search_code` outputs.

### ☁️ Cloud (NVIDIA NIM)

Run the agent against NVIDIA NIM (OpenAI-compatible) instead of Ollama:

```bash
# Key from https://build.nvidia.com/ — global settings (Claude-style) or env
# `~/.anamnesic/settings.json`:
#   { "env": { "NVIDIA_API_KEY": "nvapi-..." } }
echo "NVIDIA_API_KEY=nvapi-..." >> ~/.anamnesic/settings.json  # or export it

cargo run -- --cloud "add user authentication"
cargo run -- --cloud --cloud-model z-ai/glm-5.2 "explain this code"
```

- `--provider <id>`: any OpenAI-compatible provider from the models.dev catalog (default `nvidia`).
- `--cloud-model <id>`: override model (default `z-ai/glm-5.2`, selected for the best latency/quality result in the local agent benchmark).
- API keys live in `~/.anamnesic/settings.json` (global, Claude Code-style `env` block) or `~/.anamnesic/providers.toml`.
- Key priority: `providers set nvidia <key>` → `~/.anamnesic/settings.json` `env` → process env → project `.env`.

### 💻 Hardware Recommendation

Auto-detect CPU/RAM/GPU and recommend optimal models. Based on [llm-checker](https://github.com/Pavelevich/llm-checker).

```
cargo run -- check
```

Reads `/proc/cpuinfo`, `/proc/meminfo`, sysfs/lspci for GPU. 4D scoring: Quality, Speed, Fit, Context.

### 🏗️ Architecture

```
Task → Planner (JSON plan) → Executor (steps) → Verify → retry if failed
       │                        │
       └── Ollama API ──────────┘
              │
         optional: local GGUF inference engine
```

### Quick Start

```bash
# Requires Ollama running on localhost:11434
cargo run -- --dir ./workspace "initialize a react project"

# Hardware check + model recommendations
cargo run -- check

# Caveman mode
cargo run -- --caveman full "explain this code"

# REPL mode
cargo run
```

### Models

| Role | Default | Config |
|------|---------|--------|
| Planner | `granite3.3:2b` | `PLANNER_MODEL` env |
| Coder | `qwen3:1.7b` | `CODER_MODEL` env |
| Summarizer | `qwen3:0.6b` | `SUMMARIZER_MODEL` env |

### Shell.nix

```bash
nix-shell shell.nix
cargo build
```
