# anamnesic-coder

Local coding agent — plan → act → verify. Fusion of TinyCoder + llm-on-legacy-gpus.

```
cargo run -- --dir /tmp/project "add user authentication"
```

## Features

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
